"""Checks for data leakage, validation isolation and batched metric weighting."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datasets.split import limit_per_class, stratified_split
from utils.training import evaluate, train_one_epoch


class TrainingChecks(unittest.TestCase):
    def test_split_is_repeatable_disjoint_and_covers_every_class(self):
        labels = [label for label in range(10) for _ in range(13)]
        train, val = stratified_split(labels, 0.2, 42)
        self.assertEqual((train, val), stratified_split(labels, 0.2, 42))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(len(labels))))
        self.assertEqual({labels[i] for i in train}, set(range(10)))
        self.assertEqual({labels[i] for i in val}, set(range(10)))
        self.assertNotEqual(val, stratified_split(labels, 0.2, 43)[1])

    def test_debug_subsets_stay_inside_the_original_splits(self):
        labels = [label for label in range(10) for _ in range(20)]
        train, val = stratified_split(labels, 0.2, 42)
        small_train = limit_per_class(train, labels, 8, 43)
        small_val = limit_per_class(val, labels, 2, 44)
        self.assertEqual(len(small_train), 80)
        self.assertEqual(len(small_val), 20)
        self.assertTrue(set(small_train) <= set(train))
        self.assertTrue(set(small_val) <= set(val))
        self.assertFalse(set(small_train) & set(small_val))

    def test_validation_weights_partial_batch_and_does_not_change_model(self):
        # 最后一个样本故意预测错误，使错误的“按 batch 平均”算法暴露出来。
        points = torch.tensor([[10., 0.], [0., 10.], [10., 0.]])
        labels = torch.tensor([0, 1, 1])
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[1].weight.copy_(torch.eye(2))
        model.eval()
        with torch.no_grad():
            expected_loss = nn.CrossEntropyLoss()(model(points), labels).item()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        model.train()
        result = evaluate(model, DataLoader(TensorDataset(points, labels), batch_size=2), nn.CrossEntropyLoss(), "cpu")
        self.assertAlmostEqual(result["loss"], expected_loss, places=5)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["samples"], 3)
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_training_updates_weights_and_counts_samples(self):
        points = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.]])
        labels = torch.tensor([0, 0, 1, 1])
        model = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        result = train_one_epoch(model, DataLoader(TensorDataset(points, labels), batch_size=2),
                                 optimizer, nn.CrossEntropyLoss(), "cpu")
        self.assertEqual(result["samples"], 4)
        self.assertTrue(model.training)
        self.assertGreater(model.weight.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
