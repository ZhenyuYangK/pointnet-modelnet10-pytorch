"""Summarize a completed point-count suite together with its fixed 1024 baseline."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    csv_path = args.output.with_suffix(".csv")
    require(not any(p.exists() for p in (args.output, csv_path, args.figure)), "Choose new output paths")
    suite = read_json(args.suite)
    require(suite["status"] == "completed", "Suite has not completed")
    trains = {1024: read_json(suite["baseline_training"])}
    evaluations = {1024: read_json(suite["baseline_test"])}
    for step in suite["steps"]:
        require(step["status"] == "completed", "A suite step is incomplete")
        target = trains if step["kind"] == "train" else evaluations
        target[step["num_points"]] = read_json(step["report"])
    require(set(trains) == set(evaluations) == {256, 512, 1024}, "Expected 256/512/1024")
    baseline, baseline_test = trains[1024], evaluations[1024]
    manifest = read_json(baseline["split_manifest"])
    settings = ("epochs", "batch_size", "learning_rate", "dropout", "val_fraction", "seed",
                "train_per_class", "val_per_class", "num_workers", "cpu_threads", "optimizer",
                "point_sampling", "augmentation", "drop_last")
    paths_and_labels = None
    rows = []
    for n in sorted(trains):
        train, evaluation = trains[n], evaluations[n]
        require(train["status"] == evaluation["status"] == "completed", "Run incomplete")
        require(train["config"]["num_points"] == evaluation["num_points"] == n, "Point count mismatch")
        require(read_json(train["split_manifest"]) == manifest, "Data split differs")
        for key in settings:
            require(train["config"][key] == baseline["config"][key], f"Training setting differs: {key}")
        for key in ("device", "device_name", "torch_version", "best_selection"):
            require(train[key] == baseline[key], f"Training environment/selection differs: {key}")
        for name in ("models/pointnet.py", "datasets/modelnet10.py", "datasets/split.py", "utils/seed.py"):
            require(train["source_files_sha256"][name] == baseline["source_files_sha256"][name], f"Source differs: {name}")
        for key in ("seed", "data_split", "protocol", "class_to_idx", "device", "device_name", "torch_version", "batch_size", "num_workers"):
            require(evaluation[key] == baseline_test[key], f"Test setting differs: {key}")
        require(evaluation["source_files_sha256"] == baseline_test["source_files_sha256"], "Test source differs")
        require(evaluation["checkpoint_sha256"] == train["checkpoint_sha256"], "Test checkpoint hash differs")
        require(evaluation["checkpoint_epoch"] == train["best_epoch"], "Test epoch differs")
        with (ROOT / evaluation["predictions_csv"]).open(newline="") as source:
            predictions = list(csv.DictReader(source))
        current = {p["path"]: p["true_label"] for p in predictions}
        require(len(current) == len(predictions) == evaluation["metrics"]["total"] == 908, "Test sample count differs")
        require(sum(p["true_label"] == p["predicted_label"] for p in predictions) == evaluation["metrics"]["correct"], "Prediction count differs from metrics")
        require(paths_and_labels is None or current == paths_and_labels, "Test object list differs")
        paths_and_labels = current
        history = train["history"]
        require(len(history) == train["config"]["epochs"] == 50, "Expected 50 epochs")
        require(all(h["train"]["samples"] == 3193 and h["validation"]["samples"] == 798 for h in history), "Epoch sample counts differ")
        metrics = evaluation["metrics"]
        rows.append({"num_points": n, "train_run": train["run_name"], "test_run": evaluation["run_name"],
                     "best_epoch": train["best_epoch"], "validation_accuracy": train["best_validation"]["accuracy"],
                     "test_accuracy": metrics["overall_accuracy"], "test_correct": metrics["correct"],
                     "test_mean_class_accuracy": metrics["mean_class_accuracy"], "test_loss": metrics["loss"],
                     "training_total_seconds": train["elapsed_seconds"],
                     "first_epoch_seconds": history[0]["elapsed_seconds"],
                     "mean_epoch_2_to_50_seconds": mean(h["elapsed_seconds"] for h in history[1:]),
                     "test_total_seconds": evaluation["elapsed_seconds"],
                     "checkpoint": train["checkpoint"], "checkpoint_sha256": train["checkpoint_sha256"]})
    selected = max(rows, key=lambda row: (row["validation_accuracy"], -trains[row["num_points"]]["best_validation"]["loss"]))
    result = {"suite": str(args.suite), "factor": "num_points", "same_train_validation_test_objects": True,
              "matched_training_settings": {key: baseline["config"][key] for key in settings},
              "sampling_note": suite["sampling_note"], "runs": rows,
              "selected_by_validation_num_points": selected["num_points"],
              "per_class": {n: evaluations[n]["metrics"]["per_class"] for n in evaluations},
              "limitations": ["One seed only; clouds are not nested", "Historical 1024 run reused",
                              "Timing includes data loading; epoch timing includes both training and validation",
                              "No repeated performance benchmark or statistical significance claim"],
              "figure": str(args.figure)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    counts = [row["num_points"] for row in rows]
    for key, label in (("validation_accuracy", "Best validation"), ("test_accuracy", "Official test")):
        axes[0].plot(counts, [100 * row[key] for row in rows], marker="o", label=label)
    axes[0].set(ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0].legend()
    axes[1].plot(counts, [row["training_total_seconds"] for row in rows], marker="o")
    axes[1].set(ylabel="Total training run (seconds)")
    axes[2].plot(counts, [row["mean_epoch_2_to_50_seconds"] for row in rows], marker="o")
    axes[2].set(ylabel="Mean epoch 2-50 (seconds)")
    for axis in axes:
        axis.set_xticks(counts)
        axis.set_xlabel("Points per object")
        axis.grid(alpha=0.25)
    fig.suptitle("Point count comparison | no augmentation | seed 42 | timing is observational")
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2))
    print(f"Configuration selected by validation: {selected['num_points']} points")


if __name__ == "__main__":
    main()
