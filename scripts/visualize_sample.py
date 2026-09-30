"""Save a normalized point cloud figure; optionally display an interactive window."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

from datasets import ModelNet10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--class-name", default="chair")
    parser.add_argument("--index", type=int, default=0, help="Zero-based index within each class")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-classes", action="store_true", help="Show one sample per class")
    parser.add_argument("--output", type=Path, help="Figure path; default: results/figures/<sample>.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window")
    args = parser.parse_args()
    if args.index < 0 or args.num_points < 2 or args.seed < 0:
        parser.error("Require index >= 0, num-points >= 2, seed >= 0")

    # 默认只保存图片，在没有桌面的环境中也能运行。
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = ModelNet10(root=args.root, split=args.split, num_points=args.num_points, seed=args.seed)
    if not args.all_classes and args.class_name not in dataset.class_to_idx:
        parser.error(f"Unknown class; choose from: {', '.join(dataset.classes)}")
    classes = dataset.classes if args.all_classes else [args.class_name]
    selections = []
    for name in classes:
        indices = [i for i, (_, label) in enumerate(dataset.samples)
                   if label == dataset.class_to_idx[name]]
        if args.index >= len(indices):
            parser.error(f"{name} has {len(indices)} samples; index must be smaller")
        selections.append(indices[args.index])

    fig = plt.figure(figsize=(18, 8) if args.all_classes else (8, 7), layout="constrained")
    for panel, index in enumerate(selections, start=1):
        points, label = dataset[index]
        points = points.numpy()
        path, _ = dataset.samples[index]
        ax = fig.add_subplot(2, 5, panel, projection="3d") if args.all_classes else fig.add_subplot(projection="3d")
        # 颜色仅表示 z 坐标，不是模型预测；三个坐标轴保持相同尺度。
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2],
                   cmap="viridis", s=3 if args.all_classes else 6, alpha=0.8)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-55)
        ax.set_title(f"{path.stem}\nlabel={label} ({dataset.classes[label]})")
        print(f"Sample: {path.name}; shape={list(points.shape)}; label={label}")

    fig.suptitle(f"ModelNet10 {args.split} | {args.num_points} surface points | seed={args.seed}")
    name = f"modelnet10_{args.split}_classes" if args.all_classes else dataset.samples[selections[0]][0].stem
    output = args.output or PROJECT_ROOT / "results" / "figures" / f"{name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Figure saved: {output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
