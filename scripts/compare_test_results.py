"""Compare saved official test predictions under an identical sampling protocol."""

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_predictions(report):
    with (PROJECT_ROOT / report["predictions_csv"]).open(newline="") as source:
        rows = list(csv.DictReader(source))
    indexed = {row["path"]: row for row in rows}
    if len(indexed) != len(rows) or len(rows) != report["metrics"]["total"]:
        raise ValueError("Prediction count or unique sample count does not match report")
    correct = sum(row["true_label"] == row["predicted_label"] for row in rows)
    if correct != report["metrics"]["correct"]:
        raise ValueError("CSV correct count does not match report")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    baseline, candidate = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    for report in (baseline, candidate):
        if report["status"] != "completed":
            parser.error("Both evaluations must be completed")
    for key in ("data_split", "num_points", "seed", "class_to_idx", "protocol"):
        if baseline[key] != candidate[key]:
            parser.error(f"Evaluation protocol differs: {key}")
    for key in ("test.py", "datasets/modelnet10.py", "utils/metrics.py"):
        if baseline["source_files_sha256"][key] != candidate["source_files_sha256"][key]:
            parser.error(f"Evaluation source differs: {key}")
    old, new = read_predictions(baseline), read_predictions(candidate)
    if old.keys() != new.keys():
        parser.error("Evaluated sample lists differ")
    groups = {"both_correct": [], "corrected": [], "regressed": [], "both_wrong": []}
    for path, row in old.items():
        other = new[path]
        if row["true_label"] != other["true_label"]:
            parser.error(f"True label differs for {path}")
        before = row["true_label"] == row["predicted_label"]
        after = other["true_label"] == other["predicted_label"]
        group = ("both_correct" if after else "regressed") if before else ("corrected" if after else "both_wrong")
        groups[group].append(path)
    per_class = []
    for first, second in zip(baseline["metrics"]["per_class"], candidate["metrics"]["per_class"]):
        if first["class"] != second["class"] or first["support"] != second["support"]:
            parser.error("Per-class entries differ")
        per_class.append({"class": first["class"], "support": first["support"],
                          "baseline_accuracy": first["accuracy"], "candidate_accuracy": second["accuracy"],
                          "correct_delta": second["correct"] - first["correct"],
                          "accuracy_delta_percentage_points": 100 * (second["accuracy"] - first["accuracy"])})
    result = {
        "baseline_report": str(args.baseline), "candidate_report": str(args.candidate),
        "same_sample_list_and_protocol": True, "total": len(old),
        "baseline_metrics": baseline["metrics"], "candidate_metrics": candidate["metrics"],
        "accuracy_delta_percentage_points": 100 * (candidate["metrics"]["overall_accuracy"] - baseline["metrics"]["overall_accuracy"]),
        "paired_counts": {key: len(value) for key, value in groups.items()},
        "paired_paths": groups, "per_class_comparison": per_class,
        "interpretation": "Descriptive comparison of these fixed checkpoints; one training seed only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("total", "accuracy_delta_percentage_points", "paired_counts", "per_class_comparison")}, indent=2))


if __name__ == "__main__":
    main()
