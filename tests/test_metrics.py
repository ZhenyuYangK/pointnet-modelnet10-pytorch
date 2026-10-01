"""Check metric definitions using asymmetric, imbalanced examples."""

import unittest

from utils.metrics import classification_metrics


class MetricChecks(unittest.TestCase):
    def test_confusion_orientation_and_macro_vs_overall_accuracy(self):
        result = classification_metrics([0, 0, 0, 1, 2], [0, 0, 1, 2, 2], 3)
        self.assertEqual(result["confusion_matrix"], [[2, 1, 0], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["total"], 5)
        self.assertAlmostEqual(result["overall_accuracy"], 3 / 5)
        self.assertAlmostEqual(result["mean_class_accuracy"], (2 / 3 + 0 + 1) / 3)

    def test_absent_class_is_not_counted_as_perfect_or_zero(self):
        result = classification_metrics([0, 0], [0, 1], 3)
        self.assertIsNone(result["per_class"][1]["accuracy"])
        self.assertAlmostEqual(result["mean_class_accuracy"], 0.5)

    def test_invalid_labels_are_rejected(self):
        for targets, predictions in (([], []), ([0], [3]), ([0], [-1]), ([0], [0, 1]), ([0.5], [0])):
            with self.subTest(targets=targets, predictions=predictions):
                with self.assertRaises(ValueError):
                    classification_metrics(targets, predictions, 3)


if __name__ == "__main__":
    unittest.main()
