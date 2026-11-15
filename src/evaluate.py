"""Metrics.

Status: first executable version.
Online accuracy of a task = fraction of training examples classified correctly by the
model's prediction made BEFORE the update on that batch, over the task's (last) epoch.
"""
import numpy as np


def batch_accuracy(logits, targets):
    """Fraction of rows whose argmax equals the target."""
    return (logits.argmax(dim=1) == targets).float().mean().item()


def mean_last_fraction(values, fraction=0.1):
    """Mean of the last `fraction` of a per-task list (the paper's headline summary)."""
    n = max(1, int(round(len(values) * fraction)))
    return float(np.mean(values[-n:]))