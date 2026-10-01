"""Evaluate one preselected PointNet checkpoint on official ModelNet10 test data."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import ModelNet10
from models.pointnet import PointNetClassifier
from utils.metrics import classification_metrics
from utils.seed import seed_everything
from utils.visualization import save_confusion_matrix, save_prediction_examples

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = {"bathtub": 50, "bed": 100, "chair": 100, "desk": 86, "dresser": 86,
                   "monitor": 100, "night_stand": 86, "sofa": 100, "table": 100, "toilet": 100}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def predict(model, dataset, loader, device):
    model.eval()
    loss_sum = 0.0
    predictions, examples = [], {True: [], False: []}
    seen_example_classes = {True: set(), False: set()}
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch_index, (points, labels) in enumerate(loader, start=1):
        logits = model(points.to(device))
        loss = criterion(logits, labels.to(device))
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite test output")
        loss_sum += loss.item()
        confidence, predicted_labels = logits.softmax(dim=1).max(dim=1)
        for local_index, (true_label, predicted_label, score) in enumerate(zip(
            labels.tolist(), predicted_labels.cpu().tolist(), confidence.cpu().tolist(),
        )):
            index = len(predictions)
            path, expected_label = dataset.samples[index]
            if true_label != expected_label:
                raise RuntimeError("Prediction order does not match the test file list")
            row = {
                "index": index, "path": str(path.relative_to(dataset.root)),
                "true_label": true_label, "true_class": dataset.classes[true_label],
                "predicted_label": predicted_label, "predicted_class": dataset.classes[predicted_label],
                "confidence": score,
            }
            predictions.append(row)
            correct = true_label == predicted_label
            # 各取至多三个不同真实类别的首个正确／错误例子，不按置信度挑选。
            if len(examples[correct]) < 3 and true_label not in seen_example_classes[correct]:
                examples[correct].append({**row, "name": path.stem, "points": points[local_index].clone()})
                seen_example_classes[correct].add(true_label)
        if batch_index % 5 == 0 or batch_index == len(loader):
            print(f"Evaluated {len(predictions)}/{len(dataset)} test models", flush=True)
    if len(predictions) != len(dataset):
        raise RuntimeError("Not all test samples were evaluated")
    return predictions, loss_sum / len(dataset), examples[True] + examples[False]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--training-report", type=Path, help="Default: infer report from checkpoint directory name")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run-name", help="Unique result name; default includes timestamp")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        parser.error("Require batch-size >= 1, num-workers >= 0, cpu-threads >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_name = args.run_name or now.strftime("test_%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        parser.error("run-name must contain only letters, numbers, underscores and hyphens")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    metrics_path = PROJECT_ROOT / "results" / "metrics" / f"{run_name}.json"
    csv_path = PROJECT_ROOT / "results" / "metrics" / f"{run_name}_predictions.csv"
    matrix_path = PROJECT_ROOT / "results" / "figures" / f"{run_name}_confusion_matrix.png"
    examples_path = PROJECT_ROOT / "results" / "figures" / f"{run_name}_examples.png"
    if any(path.exists() for path in (metrics_path, csv_path, matrix_path, examples_path)):
        parser.error(f"Run {run_name} already exists; choose a new name")

    start = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    checkpoint_hash = file_hash(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    num_points, seed = saved["config"]["num_points"], saved["config"]["seed"]
    seed_everything(seed)
    report_path = args.training_report or PROJECT_ROOT / "results" / "metrics" / f"{args.checkpoint.parent.name}.json"
    if args.training_report and not report_path.is_file():
        parser.error(f"Training report does not exist: {report_path}")
    provenance_verified = False
    if report_path.is_file():
        training_report = json.loads(report_path.read_text())
        if training_report["status"] != "completed" or training_report["best_epoch"] != saved["epoch"]:
            raise ValueError("Checkpoint is not the selected best epoch of a completed training run")
        expected_hash = training_report.get("checkpoint_sha256")
        if expected_hash and expected_hash != checkpoint_hash:
            raise ValueError("Checkpoint hash does not match the selected training artifact")
        provenance_verified = expected_hash == checkpoint_hash

    dataset = ModelNet10(split="test", root=args.root, num_points=num_points, seed=seed)
    counts = dict(Counter(dataset.classes[label] for _, label in dataset.samples))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected official test class counts: {counts}")
    if dataset.class_to_idx != saved["class_to_idx"]:
        raise ValueError("Dataset and checkpoint have different class mappings")
    model = PointNetClassifier(**saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, generator=torch.Generator().manual_seed(seed))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": run_name, "started_at": now.isoformat(), "status": "running",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": saved["epoch"], "training_report": str(report_path) if report_path.is_file() else None,
        "checkpoint_matches_training_report_hash": provenance_verified,
        "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "data_split": "official test",
        "num_points": num_points, "seed": seed, "batch_size": args.batch_size,
        "num_workers": args.num_workers, "class_to_idx": dataset.class_to_idx,
        "protocol": "one fixed surface sample per mesh; no augmentation or voting; checkpoint chosen on validation",
        "source_files_sha256": {name: file_hash(PROJECT_ROOT / name) for name in
                                ("test.py", "utils/metrics.py", "models/pointnet.py", "datasets/modelnet10.py")},
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checkpoint epoch={saved['epoch']}; device={device}; test samples={len(dataset)}; points={num_points}; seed={seed}", flush=True)
    try:
        predictions, loss, examples = predict(model, dataset, loader, device)
        metrics = classification_metrics([row["true_label"] for row in predictions],
                                         [row["predicted_label"] for row in predictions], len(dataset.classes))
        for item in metrics["per_class"]:
            item["class"] = dataset.classes[item["label"]]
        for name, value in model.state_dict().items():
            if not torch.equal(before[name], value.detach().cpu()):
                raise RuntimeError(f"Evaluation modified model state: {name}")
        if file_hash(args.checkpoint) != checkpoint_hash:
            raise RuntimeError("Checkpoint file changed during evaluation")
        with csv_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        save_confusion_matrix(metrics["confusion_matrix"], dataset.classes, matrix_path,
                              f"ModelNet10 test | epoch {saved['epoch']} | accuracy {metrics['overall_accuracy']:.2%}")
        save_prediction_examples(examples, examples_path, "Official test examples | title: green=correct, red=incorrect\nPoint color indicates z coordinate")
        confusions = [
            {"true_class": dataset.classes[row], "predicted_class": dataset.classes[column], "count": count}
            for row, values in enumerate(metrics["confusion_matrix"])
            for column, count in enumerate(values) if row != column and count > 0
        ]
        confusions.sort(key=lambda item: (-item["count"], item["true_class"], item["predicted_class"]))
        result.update({
            "status": "completed", "metrics": {**metrics, "loss": loss},
            "top_confusions": confusions[:10], "model_state_unchanged": True, "checkpoint_file_unchanged": True,
            "predictions_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_figure": str(matrix_path.relative_to(PROJECT_ROOT)),
            "examples_figure": str(examples_path.relative_to(PROJECT_ROOT)),
            "example_selection": "first correct and incorrect sample per true class, at most three classes each, in dataset order",
            "example_paths": [row["path"] for row in examples],
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        })
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    except Exception as error:
        result.update({"status": "failed", "error": str(error)})
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
        raise
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4%} ({metrics['correct']}/{metrics['total']})")
    print(f"Mean class accuracy: {metrics['mean_class_accuracy']:.4%}; test loss: {loss:.6f}")
    for item in metrics["per_class"]:
        print(f"  {item['class']}: {item['correct']}/{item['support']} = {item['accuracy']:.2%}")
    print(f"Model state and checkpoint unchanged.\nMetrics: {metrics_path}\nPredictions: {csv_path}\nMatrix: {matrix_path}\nExamples: {examples_path}")


if __name__ == "__main__":
    main()
