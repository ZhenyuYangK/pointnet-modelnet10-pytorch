"""Train PointNet with a reproducible held-out split of official training data."""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import ModelNet10
from datasets.split import CachedPointCloudSubset, limit_per_class, stratified_split
from models.pointnet import PointNetClassifier
from utils.seed import seed_everything
from utils.training import evaluate, train_one_epoch
from utils.visualization import save_training_curves

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-per-class", type=int, default=0, help="Debug subset limit; 0 uses all training samples")
    parser.add_argument("--val-per-class", type=int, default=0, help="Debug subset limit; 0 uses all validation samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run-name", help="Unique run name; default includes a timestamp")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_points < 2:
        parser.error("Require epochs >= 1, batch-size >= 2, num-points >= 2")
    if min(args.num_workers, args.train_per_class, args.val_per_class) < 0 or args.cpu_threads < 1:
        parser.error("Workers/subset limits must be nonnegative; cpu-threads must be positive")
    if not 0 < args.val_fraction < 1 or not 0 <= args.dropout < 1:
        parser.error("Require 0 < val-fraction < 1 and 0 <= dropout < 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Require finite positive learning-rate and seed in [0, 2**32)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("run-name must contain only letters, numbers, underscores and hyphens")
    return args


def save_json(path, content):
    # 同一次运行按 epoch 更新；临时文件写完后替换，避免中途留下半个 JSON。
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_name = args.run_name or now.strftime("train_%Y%m%d_%H%M%S_%f")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_name
    metrics_path = PROJECT_ROOT / "results" / "metrics" / f"{run_name}.json"
    split_path = PROJECT_ROOT / "results" / "metrics" / f"{run_name}_split.json"
    figure_path = PROJECT_ROOT / "results" / "figures" / f"{run_name}.png"
    if any(path.exists() for path in (checkpoint_dir, metrics_path, split_path, figure_path)):
        raise SystemExit(f"Run {run_name} already exists; choose a new name")

    start = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    # 只打开官方 train 文件夹；validation 来自这里，官方 test 不参与训练和选模型。
    dataset = ModelNet10(split="train", root=args.root, num_points=args.num_points, seed=args.seed)
    labels = [label for _, label in dataset.samples]
    train_indices, val_indices = stratified_split(labels, args.val_fraction, args.seed)
    used_train = limit_per_class(train_indices, labels, args.train_per_class, args.seed + 1)
    used_val = limit_per_class(val_indices, labels, args.val_per_class, args.seed + 2)
    if set(train_indices) & set(val_indices) or len(train_indices) + len(val_indices) != len(dataset):
        raise RuntimeError("Training/validation split is not disjoint and exhaustive")

    def paths(indices):
        return [str(dataset.samples[index][0].relative_to(dataset.root)) for index in indices]

    manifest = {
        "source": "official ModelNet10 train only", "seed": args.seed,
        "val_fraction": args.val_fraction, "class_to_idx": dataset.class_to_idx,
        "train": paths(train_indices), "validation": paths(val_indices),
        "used_train": paths(used_train), "used_validation": paths(used_val),
    }
    train_data = CachedPointCloudSubset(dataset, used_train)
    val_data = CachedPointCloudSubset(dataset, used_val)
    # 只在尾批恰好为 1 时丢弃它，避免分类层 BatchNorm 报错。
    drop_last = len(train_data) % args.batch_size == 1
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=drop_last,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    model_kwargs = {"num_classes": len(dataset.classes), "dropout": args.dropout}
    model = PointNetClassifier(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"optimizer": "Adam", "point_sampling": "fixed seed per mesh, cached in RAM",
                   "augmentation": False, "drop_last": drop_last})
    checkpoint_dir.mkdir(parents=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(split_path, manifest)
    history = []
    report = {
        "run_name": run_name, "started_at": now.isoformat(), "status": "running",
        "purpose": "small subset smoke run" if args.train_per_class or args.val_per_class else "full training run",
        "device": device, "torch_version": str(torch.__version__), "config": config,
        "split_counts": {"train": len(train_indices), "validation": len(val_indices)},
        "used_counts": {"train": len(used_train), "validation": len(used_val)},
        "used_class_counts": {
            "train": dict(Counter(dataset.classes[labels[i]] for i in used_train)),
            "validation": dict(Counter(dataset.classes[labels[i]] for i in used_val)),
        },
        "split_manifest": str(split_path.relative_to(PROJECT_ROOT)),
        "official_test_used": False, "history": history,
        "best_selection": "highest validation accuracy; lowest validation loss breaks ties",
        "best_epoch": None, "best_validation": None,
        "checkpoint": str((checkpoint_dir / "best_model.pth").relative_to(PROJECT_ROOT)),
        "figure": str(figure_path.relative_to(PROJECT_ROOT)),
    }
    save_json(metrics_path, report)
    print(f"Run: {run_name}; device={device}; official-train split={len(train_indices)}/{len(val_indices)}", flush=True)
    print(f"Using train={len(train_data)}, validation={len(val_data)}; batch={args.batch_size}; points={args.num_points}; dropout={args.dropout}", flush=True)
    print("First epoch samples meshes; later epochs reuse cached point clouds.", flush=True)

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                            "elapsed_seconds": round(time.perf_counter() - epoch_start, 2)})
            best = report["best_validation"]
            if best is None or (val_metrics["accuracy"], -val_metrics["loss"]) > (best["accuracy"], -best["loss"]):
                report["best_epoch"], report["best_validation"] = epoch, val_metrics
                torch.save({
                    "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "model_kwargs": model_kwargs, "epoch": epoch, "validation": val_metrics,
                    "config": config, "class_to_idx": dataset.class_to_idx,
                    "split_manifest": str(split_path.relative_to(PROJECT_ROOT)),
                }, checkpoint_dir / "best_model.pth")
            save_json(metrics_path, report)
            print(f"Epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}; "
                  f"val loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.1%}", flush=True)

        # 从磁盘创建新模型，验证保存的是最佳 epoch，且固定验证集结果可复现。
        saved = torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=True)
        restored = PointNetClassifier(**saved["model_kwargs"]).to(device)
        restored.load_state_dict(saved["model_state_dict"])
        restored_metrics = evaluate(restored, val_loader, criterion, device)
        best = report["best_validation"]
        if saved["epoch"] != report["best_epoch"] or restored_metrics["accuracy"] != best["accuracy"]:
            raise RuntimeError("Best checkpoint does not reproduce its validation result")
        if not math.isclose(restored_metrics["loss"], best["loss"], rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("Validation loss changed after loading the best checkpoint")
        save_training_curves(history, figure_path, run_name)
        report.update({"status": "completed", "checkpoint_reload_verified": True,
                       "reloaded_validation": restored_metrics,
                       "elapsed_seconds": round(time.perf_counter() - start, 2)})
        save_json(metrics_path, report)
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        save_json(metrics_path, report)
        raise
    print(f"PASS: {args.epochs} epochs; best epoch={report['best_epoch']}; checkpoint reload matched")
    print(f"Metrics: {metrics_path}\nSplit: {split_path}\nFigure: {figure_path}")


if __name__ == "__main__":
    main()
