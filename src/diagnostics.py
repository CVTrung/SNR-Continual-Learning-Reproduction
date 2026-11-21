"""Inactivity counters and Transformer diagnostics.

Status: executable.
InactivityTracker counts, per unit, consecutive time without firing:
- unit="example" (MLP, Figure 1): examples inside a batch are treated as sequential, so a
  neuron that fires on example k of a batch of B ends the batch with a counter of B - 1 - k.
  A neuron is "dead" when its counter is >= 1000 examples.
- unit="step" (Transformer, Figure 3): a unit fires on a step if it is active on any token of
  the batch. The paper counts units inactive for 5 consecutive steps.
These are diagnostics, separate from any reset criterion. When a method resets a unit, its
counter is cleared through reset_units(), so a fresh unit is never reported as dead.
"""
import torch

from src.methods.resets import fired_per_step


class InactivityTracker:
    def __init__(self, layer_sizes, device="cpu", unit="example"):
        if unit not in ("example", "step"):
            raise ValueError(f"unknown unit: {unit}")
        self.unit = unit
        self.counters = [torch.zeros(n, dtype=torch.long, device=device) for n in layer_sizes]

    @torch.no_grad()
    def update(self, activations):
        """activations: list of (B, H) or (B, T, H) post-ReLU tensors, one per layer."""
        for i, acts in enumerate(activations):
            if self.unit == "step":
                fired = fired_per_step(acts)
                self.counters[i] = torch.where(fired, torch.zeros_like(self.counters[i]),
                                               self.counters[i] + 1)
                continue
            acts = acts.reshape(-1, acts.shape[-1])
            batch = acts.shape[0]
            fired = acts > 0
            any_fired = fired.any(dim=0)
            last = batch - 1 - torch.argmax(fired.flip(0).int(), dim=0)
            self.counters[i] = torch.where(any_fired, batch - 1 - last, self.counters[i] + batch)

    def reset_units(self, layer_index, idx):
        """Clear the counters of units that an intervention has just re-initialized."""
        self.counters[layer_index][idx] = 0

    def dead_count(self, threshold=1000):
        return sum(int((c >= threshold).sum()) for c in self.counters)

    def dead_fraction(self, threshold=1000):
        """Fraction of all tracked units whose counter is >= threshold."""
        total = sum(c.numel() for c in self.counters)
        return self.dead_count(threshold) / total


def attention_entropy(probs):
    """Mean Shannon entropy (nats) of attention rows; probs: (B, heads, T, T)."""
    p = probs.clamp(min=1e-12)
    return float(-(probs * p.log()).sum(dim=-1).mean())


@torch.no_grad()
def transformer_norms(model):
    """Frobenius norms of the Q, K and FFN weights (Figure 4)."""
    b = model.block
    ffn = torch.sqrt(b.fc1.weight.pow(2).sum() + b.fc2.weight.pow(2).sum())
    return {"q_norm": float(b.q.weight.norm()), "k_norm": float(b.k.weight.norm()),
            "ffn_norm": float(ffn)}
