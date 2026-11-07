"""Evaluate the existing compact GGUFs with the restoration study's scorer."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from restoration import dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases = dataset(44, 40, 51, 90)
    with (root / "compact-check.jsonl").open("x") as output:
        for name, model in [("original", args.original), ("norm", root / "norm.gguf"), ("random", root / "random.gguf")]:
            with (root / f"compact-{name}.log").open("w") as log, subprocess.Popen(
                [str(root / "ablate"), str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, text=True, bufsize=1,
            ) as proc:
                assert proc.stdout.readline().strip() == "ready"
                rows = []
                for prompt, label, carry in cases:
                    start = time.monotonic()
                    print(prompt, file=proc.stdin, flush=True)
                    score = proc.stdout.readline().split()
                    elapsed = time.monotonic() - start
                    assert score[:2] == ["score", "0"]
                    logp = list(map(float, score[3:]))
                    prediction = max(range(10), key=logp.__getitem__)
                    rows.append(dict(label=label, carry=carry, logp=logp, prediction=prediction,
                                     greedy_token=int(score[2]), seconds=elapsed))
                status = Path(f"/proc/{proc.pid}/status").read_text().splitlines()
                peak = next(int(line.split()[1]) for line in status if line.startswith("VmHWM:"))
                record = dict(name=name, bytes=model.stat().st_size, peak_rss_kib=peak,
                              median_seconds=statistics.median(row["seconds"] for row in rows),
                              correct=sum(row["prediction"] == row["label"] for row in rows), rows=rows)
                print(json.dumps(record), file=output, flush=True)
                proc.stdin.close()
                assert proc.wait(timeout=10) == 0


if __name__ == "__main__":
    main()
