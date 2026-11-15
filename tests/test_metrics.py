"""Checks for metrics."""
import torch

from src.evaluate import batch_accuracy, mean_last_fraction


def test_batch_accuracy():
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [5.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    assert abs(batch_accuracy(logits, targets) - 2 / 3) < 1e-6


def test_mean_last_fraction():
    values = list(range(10))
    assert mean_last_fraction(values, 0.1) == 9.0
    assert mean_last_fraction(values, 0.3) == 8.0