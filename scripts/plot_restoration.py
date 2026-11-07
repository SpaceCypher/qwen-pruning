"""Plot the block ablation and restoration results."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo / "data")
    parser.add_argument("--output", type=Path, default=repo / "figures/restoration.png")
    args = parser.parse_args()
    root = args.data_dir
    rows = [json.loads(line) for line in (root / "restoration.jsonl").read_text().splitlines()]
    matched = root / "matched-test.jsonl"
    if matched.exists():
        rows.extend(json.loads(line) for line in matched.read_text().splitlines())
    records = {(row["split"], row["name"]): row for row in rows}

    def accuracy(split, name):
        row = records[split, name]
        return 100 * row["correct"] / row["count"]

    splits = ["test", "matched"] if matched.exists() else ["test"]
    fig, axes = plt.subplots(1, 1 + len(splits), figsize=(5 * (1 + len(splits)), 4),
                            layout="constrained")
    left = axes[0]
    widths = [1, 2, 4]
    for method, label in [("norm", "Lowest norm"), ("early", "Earliest channels")]:
        left.plot([0, *widths], [accuracy("dev", "baseline"), *[
            accuracy("dev", f"{method}-{width}") for width in widths]], "o-", label=label)
    for width in widths:
        values = [accuracy("dev", f"random-{width}-{seed}") for seed in range(5)]
        left.scatter([width] * 5, values, marker="x", color="black",
                     label="Random seeds" if width == 1 else None)
    left.set(xlabel="256-channel blocks removed per layer", ylabel="Digit accuracy (%)",
             title="Development ablations", ylim=(0, 105), xticks=[0, *widths])
    left.legend(frameon=False)
    for axis, split in zip(axes[1:], splits):
        for x, name in enumerate(("baseline", "norm-4", "targeted")):
            axis.scatter(x, accuracy(split, name), s=60, zorder=2)
        axis.scatter([3] * 5, [accuracy(split, f"random-restore-{seed}") for seed in range(5)],
                     marker="x", color="black")
        title = "Held-out larger operands" if split == "test" else "Held-out same-range operands"
        axis.set(xticks=range(4), xticklabels=["Original", "Low norm\ncut", "Targeted\nrestore", "Random\nrestore"],
                 title=title, ylabel="Digit accuracy (%)", ylim=(0, 105))
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
