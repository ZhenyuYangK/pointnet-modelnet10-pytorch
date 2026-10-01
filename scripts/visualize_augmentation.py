"""Show one training chair and three fresh draws of the augmentation preset."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from datasets import ModelNet10
from datasets.augmentation import PointCloudAugmentation
from datasets.split import stratified_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results/figures/augmentation_preview.png")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new --output path")
    dataset = ModelNet10(split="train", num_points=1024, seed=42)
    labels = [label for _, label in dataset.samples]
    train_indices, _ = stratified_split(labels, 0.2, 42)
    index = next(i for i in train_indices if labels[i] == dataset.class_to_idx["chair"])
    points, _ = dataset[index]
    augmentation = PointCloudAugmentation(42, "cpu")
    clouds = torch.cat([points[None], augmentation(points[None].repeat(3, 1, 1))])
    fig = plt.figure(figsize=(14, 4), layout="constrained")
    for panel, cloud in enumerate(clouds):
        ax = fig.add_subplot(1, 4, panel + 1, projection="3d")
        ax.scatter(*cloud.T.numpy(), c=points[:, 2].numpy(), cmap="viridis", s=3)
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), zlim=(-1.4, 1.4),
               xlabel="x", ylabel="y", zlabel="z",
               title="Original" if panel == 0 else f"Augmented draw {panel}")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-55)
    fig.suptitle(f"{dataset.samples[index][0].name}: scale + translation + jitter (same axis limits)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
