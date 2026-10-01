"""Show the actual fixed-seed sampling protocol at 256, 512, and 1024 points."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import ModelNet10
from datasets.split import stratified_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/point_count_preview.png")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    fig = plt.figure(figsize=(12, 4.5), layout="constrained")
    for panel, count in enumerate((256, 512, 1024), start=1):
        dataset = ModelNet10(split="train", num_points=count, seed=42)
        labels = [label for _, label in dataset.samples]
        train, _ = stratified_split(labels, 0.2, 42)
        index = next(i for i in train if labels[i] == dataset.class_to_idx["chair"])
        points, _ = dataset[index]
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ax.scatter(*points.T.numpy(), c=points[:, 2].numpy(), cmap="viridis", s=5)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), title=f"{count} points")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-55)
        ax.set_axis_off()
    fig.suptitle(f"{dataset.samples[index][0].name} | same mesh and seed, sampled separately at each N")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
