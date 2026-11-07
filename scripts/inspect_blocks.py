"""Measure how traced MLP activation blocks project through stored weights."""
import argparse
import json
from pathlib import Path
import subprocess

import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    records = [json.loads(line) for line in (root / "restoration.jsonl").read_text().splitlines()]
    target = next(row for row in records if row["name"] == "targeted")
    broken = next(row for row in records if row["name"] == "norm-4" and row["split"] == "test")
    selected = [i for i in range(32) if broken["cuts"][i] and not target["cuts"][i]]
    prompts = next(row["rows"] for row in records if row["name"] == "baseline" and row["split"] == "dev")
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    with (root / "block-contributions.jsonl").open("x") as output, (root / "trace.log").open("w") as log, subprocess.Popen(
        [str(root / "ablate-trace"), str(args.model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=log, text=True, bufsize=1,
    ) as proc:
        assert proc.stdout.readline().strip() == "ready"
        for layer in selected:
            tensor = tensors[f"blk.{layer}.ffn_down.weight"]
            weight = gguf.dequantize(tensor.data, tensor.tensor_type)
            blocks = weight.reshape(weight.shape[0], 36, 256).transpose(1, 0, 2).copy()
            norms = np.linalg.norm(blocks, axis=(1, 2))
            print(f"vector {layer}", file=proc.stdin, flush=True)
            for row in prompts:
                print(row["prompt"], file=proc.stdin, flush=True)
                vector = proc.stdout.readline().split()
                score = proc.stdout.readline().split()
                assert vector[:2] == ["activation", str(layer)] and score[:2] == ["score", "1"]
                activation = np.asarray(vector[2:], dtype=np.float32).reshape(36, 256)
                contribution = (blocks @ activation[..., None]).squeeze(-1)
                print(json.dumps(dict(layer=layer, prompt=row["prompt"], removed=broken["cuts"][layer],
                                      weight_norm=norms.tolist(),
                                      activation_norm=np.linalg.norm(activation, axis=1).tolist(),
                                      contribution_norm=np.linalg.norm(contribution, axis=1).tolist())),
                      file=output, flush=True)
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0


if __name__ == "__main__":
    main()
