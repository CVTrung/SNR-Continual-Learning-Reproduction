"""Checks for the inactivity counters and Transformer diagnostics, hand-computed values."""
import math

import torch

from src.diagnostics import InactivityTracker, attention_entropy


def test_counter_update_matches_hand_computation():
    tracker = InactivityTracker([3])
    tracker.counters[0] = torch.tensor([10, 10, 10])
    acts = torch.zeros(4, 3)
    acts[1, 1] = 1.0   # neuron 1 fires on example 1 only
    acts[3, 2] = 2.0   # neuron 2 fires on the last example
    tracker.update([acts])
    # neuron 0 silent all 4 examples: 10 + 4; neuron 1: 2 silent after firing; neuron 2: 0
    assert tracker.counters[0].tolist() == [14, 2, 0]


def test_dead_fraction_uses_threshold():
    tracker = InactivityTracker([4])
    tracker.counters[0] = torch.tensor([0, 999, 1000, 5000])
    assert tracker.dead_fraction(threshold=1000) == 0.5


def test_step_unit_counts_steps_over_tokens():
    tracker = InactivityTracker([2], unit="step")
    acts = torch.zeros(2, 5, 2)
    acts[1, 3, 0] = 1.0                    # unit 0 fires on one token
    for _ in range(5):
        tracker.update([acts])
    assert tracker.counters[0].tolist() == [0, 5]
    assert tracker.dead_count(threshold=5) == 1


def test_reset_units_clears_counters():
    tracker = InactivityTracker([3])
    tracker.counters[0] = torch.tensor([7, 8, 9])
    tracker.reset_units(0, torch.tensor([1]))
    assert tracker.counters[0].tolist() == [7, 0, 9]


def test_attention_entropy_uniform():
    probs = torch.full((1, 1, 4, 4), 0.25)
    assert abs(attention_entropy(probs) - math.log(4)) < 1e-6
