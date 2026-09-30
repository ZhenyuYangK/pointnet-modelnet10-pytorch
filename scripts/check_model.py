"""Check PointNet shapes, point-order invariance and gradients using real data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import ModelNet10
from models.pointnet import PointNetClassifier


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_device(points: torch.Tensor, labels: torch.Tensor, device: str, seed: int) -> dict:
    torch.manual_seed(seed)
    model = PointNetClassifier().to(device)
    points, labels = points.to(device), labels.to(device)

    # eval 关闭 Dropout，并让 BatchNorm 使用已记录的统计值。
    # 这样才能单独比较“点的顺序”是否影响输出。
    model.eval()
    with torch.no_grad():
        logits = model(points)
        require(tuple(logits.shape) == (points.shape[0], 10), "Wrong real-batch output shape")
        require(bool(torch.isfinite(logits).all()), "Non-finite logits")
        order = torch.randperm(points.shape[1], device=device)
        shuffled_logits = model(points[:, order, :])
        torch.testing.assert_close(logits, shuffled_logits, rtol=1e-5, atol=1e-6)
        permutation_error = (logits - shuffled_logits).abs().max().item()

        shapes = []
        for batch_size, num_points in ((1, 256), (2, 1024)):
            example = torch.randn(batch_size, num_points, 3, device=device)
            output = model(example)
            require(tuple(output.shape) == (batch_size, 10), "Wrong variable-size output shape")
            require(bool(torch.isfinite(output).all()), "Non-finite variable-size output")
            shapes.append({"input": list(example.shape), "output": list(output.shape)})

    # backward 只检查梯度能否传回模型参数；这里没有 optimizer.step，不更新参数。
    model.train()
    model.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(points), labels)
    require(bool(torch.isfinite(loss)), "Non-finite classification loss")
    loss.backward()
    squared_gradient_norm = 0.0
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None, f"No gradient for {name}")
        require(bool(torch.isfinite(parameter.grad).all()), f"Non-finite gradient for {name}")
        squared_gradient_norm += parameter.grad.square().sum().item()
    require(squared_gradient_norm > 0, "All gradients are zero")

    print(f"{device}: PASS input={list(points.shape)} -> output={list(logits.shape)}", flush=True)
    print(f"  point-order max difference={permutation_error:.3g}; variable point counts and single-sample eval passed")
    print(f"  backward passed; diagnostic loss={loss.item():.6f} (untrained model)", flush=True)
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(points.shape), "output_shape": list(logits.shape),
        "permutation_max_abs_difference": permutation_error,
        "variable_shape_checks": shapes,
        "diagnostic_loss": loss.item(), "gradient_l2_norm": squared_gradient_norm ** 0.5,
        "all_parameter_gradients_finite": True, "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", choices=("all", "cpu", "cuda"), default="all",
                        help="Default: CPU and CUDA if available")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optionally save a JSON check report")
    args = parser.parse_args()
    if args.batch_size < 2 or args.num_points < 2 or args.seed < 0:
        parser.error("Require batch-size >= 2, num-points >= 2, seed >= 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    devices = [args.device] if args.device != "all" else ["cpu"]
    if args.device == "all" and torch.cuda.is_available():
        devices.append("cuda")

    start = time.perf_counter()
    dataset = ModelNet10(split="train", root=args.root, num_points=args.num_points, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    points, labels = next(iter(loader))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "PointNetClassifier (shared MLP + max pooling, no T-Net)",
        "torch_version": torch.__version__, "seed": args.seed,
        "data_split": "train", "class_to_idx": dataset.class_to_idx,
        "checks": [check_device(points, labels, device, args.seed) for device in devices],
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report saved: {args.output}")
    print("All model checks passed. No model training has been performed.")


if __name__ == "__main__":
    main()
