"""Screen Qwen MLP block cuts and restorations on disjoint arithmetic splits."""
import argparse
import itertools
import json
import random
import statistics
import subprocess
from pathlib import Path


def dataset(seed, count, start, end):
    rng = random.Random(seed)
    problems = rng.sample(list(itertools.combinations(range(start, end), 2)), count)
    rows = []
    for a, b in problems:
        answer = a + b
        label = answer // 10 % 10
        prompt = f"What is the tens digit of {a} + {b}? Reply with only that single digit."
        rows.append((prompt, label, a % 10 + b % 10 >= 10))
    return rows


def mask_command(cuts):
    return "mask " + " ".join(f"{sum(1 << block for block in blocks):x}" for blocks in cuts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--dev", type=int, default=32)
    parser.add_argument("--test", type=int, default=96)
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--matched-test", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    manifest = json.loads((root / "norm.json").read_text())
    removed = [sorted(set(range(36)) - set(manifest["kept_blocks"][str(i)])) for i in range(32)]
    ranking = json.loads((root / "norm-ranking.json").read_text())
    assert [sorted(row[:4]) for row in ranking] == removed
    dev = dataset(43, args.dev, 21, 50)
    test = dataset(44, args.test, 51, 90)
    screen = dev[:min(12, len(dev))]
    filename = "necessity.jsonl" if args.followup else "restoration.jsonl"
    if args.matched_test:
        filename = "matched-test.jsonl"
        used = {row[0] for row in dev}
        test = [row for row in dataset(45, args.dev + args.test, 21, 50) if row[0] not in used][:args.test]
    with (root / filename).open("x") as output, (root / f"{Path(filename).stem}.log").open("w") as log, subprocess.Popen(
        [str(root / "ablate"), str(args.model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=log, text=True, bufsize=1,
    ) as proc:
        assert proc.stdout.readline().strip() == "ready", "model failed to load"

        def evaluate(name, cases, cuts, split, trace=False):
            print(mask_command(cuts), file=proc.stdin, flush=True)
            print("trace on" if trace else "trace off", file=proc.stdin, flush=True)
            expected_hits = 32 if trace else sum(bool(layer) for layer in cuts)
            rows = []
            for prompt, label, carry in cases:
                print(prompt, file=proc.stdin, flush=True)
                traces = {}
                line = proc.stdout.readline().split()
                while line and line[0] == "trace":
                    traces[int(line[1])] = list(map(float, line[2:]))
                    line = proc.stdout.readline().split()
                assert line and line[0] == "score" and int(line[1]) == expected_hits, line
                logp = list(map(float, line[3:]))
                prediction = max(range(10), key=logp.__getitem__)
                row = dict(prompt=prompt, label=label, carry=carry, prediction=prediction,
                           greedy_token=int(line[2]), logp=logp)
                if trace:
                    row["block_activation_norms"] = traces
                rows.append(row)
            accuracy = sum(row["label"] == row["prediction"] for row in rows)
            mean = statistics.mean(row["logp"][row["label"]] for row in rows)
            record = dict(name=name, split=split, cuts=cuts, correct=accuracy,
                          count=len(rows), mean_logp=mean, rows=rows)
            print(json.dumps(record), file=output, flush=True)
            print(f"{split} {name}: {accuracy}/{len(rows)}, log p {mean:.3f}", flush=True)
            return mean

        def random_cut(width, seed):
            rng = random.Random(seed)
            return [rng.sample(range(36), width) for _ in range(32)]

        def restore(layers):
            return [[] if i in layers else blocks for i, blocks in enumerate(removed)]

        empty = [[] for _ in range(32)]
        if args.matched_test:
            records = [json.loads(line) for line in (root / "restoration.jsonl").read_text().splitlines()]
            for row in records:
                if row["split"] == "test":
                    evaluate(row["name"], test, row["cuts"], "matched")
            proc.stdin.close()
            assert proc.wait(timeout=10) == 0
            return

        if args.followup:
            records = [json.loads(line) for line in (root / "restoration.jsonl").read_text().splitlines()]
            target = next(row for row in records if row["name"] == "targeted")
            selected = [i for i in range(32) if not target["cuts"][i]]
            evaluate("baseline", test, empty, "test")
            for layer in selected:
                cuts = [removed[i] if i == layer else [] for i in range(32)]
                control = [[] for _ in range(32)]
                control[layer] = random.Random(layer).sample(
                    sorted(set(range(36)) - set(removed[layer])), len(removed[layer]))
                evaluate(f"knockout-L{layer}", test, cuts, "test")
                evaluate(f"control-L{layer}", test, control, "test")
            proc.stdin.close()
            assert proc.wait(timeout=10) == 0
            return

        evaluate("baseline", dev, empty, "dev", trace=True)
        for width in (1, 2, 4):
            evaluate(f"norm-{width}", dev, [row[:width] for row in ranking], "dev")
            evaluate(f"early-{width}", dev, [list(range(width)) for _ in range(32)], "dev")
            for seed in range(5):
                evaluate(f"random-{width}-{seed}", dev, random_cut(width, seed), "dev")

        evaluate("norm-4", screen, removed, "screen")
        gains = [(evaluate(f"restore-L{i}", screen, restore([i]), "screen"), i) for i in range(32)]
        selected = [i for _, i in sorted(gains, reverse=True)[:4]]
        print(f"Frozen restoration layers {selected}", flush=True)
        evaluate("baseline", test, empty, "test")
        evaluate("norm-4", test, removed, "test")
        evaluate("targeted", test, restore(selected), "test")
        for seed in range(5):
            layers = random.Random(seed + 100).sample(range(32), len(selected))
            evaluate(f"random-restore-{seed}", test, restore(layers), "test")
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0


if __name__ == "__main__":
    main()
