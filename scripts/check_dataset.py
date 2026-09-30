"""Check real ModelNet10 samples and one DataLoader batch per split."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets import ModelNet10


# 官方划分的样本数，同时用来检查类别编号有没有发生变化。
EXPECTED_COUNTS = {
    "train": dict(zip(
        ("bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"),
        (106, 515, 889, 200, 200, 465, 200, 680, 392, 344),
    )),
    "test": dict(zip(
        ("bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"),
        (50, 100, 100, 86, 86, 100, 86, 100, 100, 100),
    )),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_split(args: argparse.Namespace, split: str) -> dict:
    dataset = ModelNet10(split=split, root=args.root, num_points=args.num_points, seed=args.seed)
    require(dataset.classes == sorted(EXPECTED_COUNTS[split]), "Unexpected class names")
    counts = Counter(dataset.classes[label] for _, label in dataset.samples)
    require(dict(counts) == EXPECTED_COUNTS[split], f"Unexpected {split} counts: {dict(counts)}")
    print(f"{split}: {len(dataset)} samples; class_to_idx={dataset.class_to_idx}", flush=True)

    # 默认每个类别检查一个模型；--all 会逐一检查整个划分。
    first_by_label = {}
    for index, (_, label) in enumerate(dataset.samples):
        first_by_label.setdefault(label, index)
    indices = list(range(len(dataset))) if args.all else list(first_by_label.values())
    max_center_error = 0.0
    max_radius_error = 0.0
    for checked, index in enumerate(indices, start=1):
        path, expected_label = dataset.samples[index]
        try:
            points, label = dataset[index]
            require(tuple(points.shape) == (args.num_points, 3), "Unexpected point shape")
            require(points.dtype == torch.float32, "Points must be float32")
            require(bool(torch.isfinite(points).all()), "Non-finite point coordinate")
            require(isinstance(label, int) and label == expected_label, "Incorrect label")
            center_error = points.mean(dim=0).abs().max().item()
            radius_error = abs(points.norm(dim=1).max().item() - 1.0)
            require(center_error < 1e-5, f"Cloud is not centered: {center_error}")
            require(radius_error < 1e-5, f"Cloud does not have unit radius: {radius_error}")
            max_center_error = max(max_center_error, center_error)
            max_radius_error = max(max_radius_error, radius_error)
        except Exception as error:
            raise RuntimeError(f"Failed sample: {path}") from error
        if args.all and checked % 500 == 0:
            print(f"  checked {checked}/{len(indices)}", flush=True)

    # 固定 seed 后，同一个模型应得到完全相同的点云。
    repeatable = torch.equal(dataset[0][0], dataset[0][0])
    require(repeatable, "Point sampling is not repeatable for a fixed seed")

    # DataLoader 将单个 [N, 3] 点云堆叠成 [B, N, 3]，整数标签变成 [B]。
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed),
    )
    batch_points, batch_labels = next(iter(loader))
    size = min(args.batch_size, len(dataset))
    require(tuple(batch_points.shape) == (size, args.num_points, 3), "Incorrect batch shape")
    require(tuple(batch_labels.shape) == (size,), "Incorrect label batch shape")
    require(batch_labels.dtype == torch.int64, "Batched labels must be int64")
    require(bool(torch.isfinite(batch_points).all()), "Batch has non-finite coordinates")
    require(bool(((batch_labels >= 0) & (batch_labels < 10)).all()), "Invalid batch labels")

    print(f"  PASS: {len(indices)} samples; batch points={list(batch_points.shape)}, "
          f"labels={list(batch_labels.shape)}; fixed seed repeatable", flush=True)
    return {
        "dataset_size": len(dataset), "class_counts": dict(counts),
        "class_to_idx": dataset.class_to_idx, "checked_samples": len(indices),
        "max_center_error": max_center_error, "max_radius_error": max_radius_error,
        "fixed_seed_repeatable": repeatable, "batch_points_shape": list(batch_points.shape),
        "batch_labels_shape": list(batch_labels.shape), "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Check every mesh; default: one per class")
    parser.add_argument("--output", type=Path, help="Optionally save a JSON check report")
    args = parser.parse_args()
    if args.num_points < 2 or args.batch_size < 1 or args.num_workers < 0 or args.seed < 0:
        parser.error("Require num-points >= 2, batch-size >= 1, num-workers >= 0, seed >= 0")

    start = time.perf_counter()
    splits = ("train", "test") if args.split == "both" else (args.split,)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "all" if args.all else "one_per_class",
        "num_points": args.num_points, "seed": args.seed,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "splits": {split: check_split(args, split) for split in splits},
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report saved: {args.output}")
    print(f"All requested checks passed in {report['elapsed_seconds']} seconds.")


if __name__ == "__main__":
    main()
