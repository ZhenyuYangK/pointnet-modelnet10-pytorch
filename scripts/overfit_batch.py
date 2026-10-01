"""Memorize one fixed, class-balanced training batch to check the training loop."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from datasets import ModelNet10
from models.pointnet import PointNetClassifier
from utils.seed import seed_everything


def fixed_batch(dataset: ModelNet10, samples_per_class: int, seed: int):
    """Choose an equal number of meshes per class, then read each mesh once."""
    generator = torch.Generator().manual_seed(seed)
    points, labels, samples = [], [], []
    for name, label in dataset.class_to_idx.items():
        candidates = [i for i, (_, item_label) in enumerate(dataset.samples) if item_label == label]
        if samples_per_class > len(candidates):
            raise ValueError(f"{name} has only {len(candidates)} training samples")
        choices = torch.randperm(len(candidates), generator=generator)[:samples_per_class]
        for choice in choices.tolist():
            index = candidates[choice]
            cloud, actual_label = dataset[index]
            points.append(cloud)
            labels.append(actual_label)
            samples.append({
                "dataset_index": index,
                "path": str(dataset.samples[index][0].relative_to(dataset.root)),
                "class": name, "label": actual_label,
            })
    return torch.stack(points), torch.tensor(labels, dtype=torch.long), samples


@torch.no_grad()
def evaluate_batch(model, points, labels, criterion):
    """Evaluate the SAME training batch; this is not validation/test accuracy."""
    model.eval()
    logits = model(points)
    loss = criterion(logits, labels)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite loss while evaluating the fixed batch")
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss.item(), accuracy, logits


def save_curve(history: list[dict], output: Path, run_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    steps = [row["step"] for row in history]
    trained = history[1:]
    axes[0].plot([row["step"] for row in trained], [row["train_loss"] for row in trained],
                 label="Train mode, before update", alpha=0.8)
    axes[0].plot(steps, [row["eval_loss"] for row in history], label="Eval mode, after update")
    axes[0].set(xlabel="Optimizer steps", ylabel="Cross-entropy loss", title="Fixed-batch loss")
    axes[1].plot([row["step"] for row in trained], [100 * row["train_accuracy"] for row in trained],
                 label="Train mode, before update", alpha=0.8)
    axes[1].plot(steps, [100 * row["eval_accuracy"] for row in history], label="Eval mode, after update")
    axes[1].set(xlabel="Optimizer steps", ylabel="Accuracy (%)", ylim=(-2, 102),
                title="Accuracy on the same training samples")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"{run_name}\nTraining-loop diagnostic; no held-out evaluation", fontsize=11)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run-name", help="Unique output name; default includes local timestamp")
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.num_points < 2 or args.steps < 1 or args.cpu_threads < 1:
        parser.error("Require samples-per-class >= 1, num-points >= 2, steps >= 1, cpu-threads >= 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Require a finite positive learning-rate and seed in [0, 2**32)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_name = args.run_name or now.strftime("overfit_%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        parser.error("run-name must contain only letters, numbers, underscores and hyphens")

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_name
    metrics_path = PROJECT_ROOT / "results" / "metrics" / f"{run_name}.json"
    figure_path = PROJECT_ROOT / "results" / "figures" / f"{run_name}.png"
    if checkpoint_dir.exists() or metrics_path.exists() or figure_path.exists():
        parser.error(f"Run {run_name} already exists; choose a new run-name")
    checkpoint_dir.mkdir(parents=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    dataset = ModelNet10(split="train", root=args.root, num_points=args.num_points, seed=args.seed)
    points, labels, samples = fixed_batch(dataset, args.samples_per_class, args.seed)
    points, labels = points.to(device), labels.to(device)
    # 诊断时关闭 Dropout，固定样本和点云，排除数据增强与随机屏蔽特征的影响。
    model_kwargs = {"num_classes": len(dataset.classes), "dropout": 0.0}
    model = PointNetClassifier(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_loss, initial_accuracy, _ = evaluate_batch(model, points, labels, criterion)
    history = [{"step": 0, "train_loss": None, "train_accuracy": None,
                "eval_loss": initial_loss, "eval_accuracy": initial_accuracy}]
    print(f"Run: {run_name}; device={device}; fixed batch={list(points.shape)}; dropout=0", flush=True)
    print(f"Step 0: eval loss={initial_loss:.6f}, accuracy={initial_accuracy:.1%}", flush=True)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)  # 清除上一步留下的梯度。
        logits = model(points)                # 预测这一批物体的类别分数。
        loss = criterion(logits, labels)      # 用真实标签衡量预测误差。
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()                       # 计算参数的梯度。
        optimizer.step()                      # 根据梯度更新参数。
        train_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        eval_loss, eval_accuracy, final_logits = evaluate_batch(model, points, labels, criterion)
        history.append({"step": step, "train_loss": loss.item(), "train_accuracy": train_accuracy,
                        "eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"Step {step}: train loss={loss.item():.6f}, accuracy={train_accuracy:.1%}; "
                  f"eval loss={eval_loss:.6f}, accuracy={eval_accuracy:.1%}", flush=True)

    # 一并保存固定点云，后续能在不重新采样的情况下复查这次诊断。
    checkpoint_path = checkpoint_dir / "model.pth"
    config = {
        "seed": args.seed, "samples_per_class": args.samples_per_class,
        "batch_size": len(labels), "num_points": args.num_points, "steps": args.steps,
        "learning_rate": args.learning_rate, "optimizer": "Adam", "dropout": 0.0,
        "data_augmentation": False, "pooling": "max", "cpu_threads": args.cpu_threads,
    }
    torch.save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_kwargs": model_kwargs, "config": config,
        "class_to_idx": dataset.class_to_idx, "samples": samples,
        "points": points.cpu(), "labels": labels.cpu(),
    }, checkpoint_path)
    # 用新模型加载权重，再计算输出，检查保存的权重是否可用。
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = PointNetClassifier(**saved["model_kwargs"]).to(device)
    restored.load_state_dict(saved["model_state_dict"])
    restored_loss, restored_accuracy, restored_logits = evaluate_batch(
        restored, saved["points"], saved["labels"], criterion,
    )
    torch.testing.assert_close(restored_logits, final_logits, rtol=1e-5, atol=1e-6)

    passed = eval_accuracy == 1.0 and eval_loss < 0.05 and eval_loss < initial_loss
    report = {
        "run_name": run_name, "started_at": now.isoformat(),
        "purpose": "Fixed training batch memorization; not validation or test performance",
        "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "config": config,
        "class_to_idx": dataset.class_to_idx, "samples": samples,
        "initial_eval_loss": initial_loss, "initial_eval_accuracy": initial_accuracy,
        "final_eval_loss": eval_loss, "final_eval_accuracy": eval_accuracy,
        "checkpoint_reload_verified": True,
        "reloaded_eval_loss": restored_loss, "reloaded_eval_accuracy": restored_accuracy,
        "acceptance": {"required_eval_accuracy": 1.0, "required_eval_loss_below": 0.05,
                       "loss_must_decrease": True, "passed": passed},
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "figure": str(figure_path.relative_to(PROJECT_ROOT)),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    save_curve(history, figure_path, run_name)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: fixed-batch eval accuracy={eval_accuracy:.1%}, loss={eval_loss:.6f}; checkpoint reload matched")
    print(f"Metrics: {metrics_path}\nFigure: {figure_path}\nCheckpoint: {checkpoint_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
