"""Classification metrics with rows=true, columns=predicted confusion counts."""

import numpy as np


def classification_metrics(targets, predictions, num_classes):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if num_classes < 1 or targets.ndim != 1 or predictions.shape != targets.shape or targets.size == 0:
        raise ValueError("Require nonempty, equally sized label vectors and positive num_classes")
    if not np.issubdtype(targets.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Labels must be integers")
    if np.any(targets < 0) or np.any(targets >= num_classes) or np.any(predictions < 0) or np.any(predictions >= num_classes):
        raise ValueError("Label is outside the class range")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    support, correct = matrix.sum(axis=1), matrix.diagonal()
    per_class = [
        {"label": label, "support": int(support[label]), "correct": int(correct[label]),
         "accuracy": float(correct[label] / support[label]) if support[label] else None}
        for label in range(num_classes)
    ]
    observed = [item["accuracy"] for item in per_class if item["accuracy"] is not None]
    return {
        "total": int(targets.size), "correct": int(correct.sum()),
        "overall_accuracy": float(correct.sum() / targets.size),
        "mean_class_accuracy": float(np.mean(observed)),
        "per_class": per_class, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_orientation": "rows=true, columns=predicted",
    }
