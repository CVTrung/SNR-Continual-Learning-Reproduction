"""Checks for the inactivity counter, with hand-computed expected values."""
import torch

from src.diagnostics import InactivityTracker


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