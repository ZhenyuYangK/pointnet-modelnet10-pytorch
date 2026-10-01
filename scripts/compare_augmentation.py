"""Compare completed baseline and augmentation runs on the same validation set."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New comparison JSON path")
    args = parser.parse_args()
    figure = args.output.with_suffix(".png")
    if args.output.exists() or figure.exists():
        parser.error("Comparison output already exists; choose a new path")
    baseline, augmented = [json.loads(path.read_text()) for path in (args.baseline, args.augmented)]
    for report in (baseline, augmented):
        if report["status"] != "completed":
            parser.error("Both training runs must be completed")
    for key in ("device", "device_name", "torch_version", "best_selection"):
        if baseline[key] != augmented[key]:
            parser.error(f"Unmatched experiment environment or selection rule: {key}")
    keys = ("epochs", "batch_size", "num_points", "learning_rate", "dropout", "val_fraction",
            "train_per_class", "val_per_class", "seed", "optimizer", "point_sampling",
            "drop_last", "num_workers", "cpu_threads")
    for key in keys:
        if baseline["config"][key] != augmented["config"][key]:
            parser.error(f"Unmatched training configuration: {key}")
    if baseline["config"]["augmentation"] or not augmented["config"]["augmentation"]:
        parser.error("Expected an unaugmented baseline and an augmented run")
    manifests = [json.loads((PROJECT_ROOT / r["split_manifest"]).read_text())
                 for r in (baseline, augmented)]
    if manifests[0] != manifests[1]:
        parser.error("Training/validation manifests differ")
    # 模型、原始点云和划分代码相同，训练入口和增强模块的变化是本实验因素。
    for name in ("models/pointnet.py", "datasets/modelnet10.py", "datasets/split.py", "utils/seed.py"):
        if baseline["source_files_sha256"][name] != augmented["source_files_sha256"][name]:
            parser.error(f"Source differs: {name}")
    rows = []
    for label, report in (("Baseline", baseline), ("Augmented", augmented)):
        best = report["best_validation"]
        rows.append({"label": label, "run_name": report["run_name"],
                     "best_epoch": report["best_epoch"], "validation": best,
                     "correct": round(best["accuracy"] * best["samples"]),
                     "elapsed_seconds": report["elapsed_seconds"],
                     "checkpoint": report["checkpoint"],
                     "checkpoint_sha256": report["checkpoint_sha256"]})
    delta = 100 * (rows[1]["validation"]["accuracy"] - rows[0]["validation"]["accuracy"])
    comparison = {"scope": "best held-out validation result; no new official test evaluation",
                  "matched_configuration": {key: baseline["config"][key] for key in keys},
                  "same_split": True, "augmentation": augmented["config"]["augmentation"],
                  "runs": rows, "accuracy_delta_percentage_points": delta,
                  "limitations": ["One seed only", "Combined augmentation; no individual attribution",
                                  "Training accuracy uses different input distributions"],
                  "figure": str(figure)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for label, report in (("Baseline", baseline), ("Scale + shift + jitter", augmented)):
        history = report["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [100 * row["validation"]["accuracy"] for row in history], label=label)
        axes[1].plot(epochs, [row["validation"]["loss"] for row in history], label=label)
    axes[0].set(ylabel="Validation accuracy (%)", ylim=(0, 100))
    axes[1].set(ylabel="Validation cross-entropy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Same 798 validation objects, seed 42, 1024 points")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    for row in rows:
        print(f"{row['label']}: epoch {row['best_epoch']}, "
              f"{row['correct']}/{row['validation']['samples']}, "
              f"accuracy={row['validation']['accuracy']:.2%}")
    print(f"Augmented minus baseline: {delta:+.2f} percentage points")


if __name__ == "__main__":
    main()
