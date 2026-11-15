"""Per-neuron inactivity counters and the dead-neuron fraction.

Status: first executable version.
Counter semantics: for every hidden neuron, the number of CONSECUTIVE training examples,
counted up to the end of the latest batch, on which its activation was 0. Examples inside a
batch are treated as sequential, so a neuron that fires on example k of a batch of B examples
ends the batch with a counter of B - 1 - k. A neuron is called inactive ("dead") when its
counter is at least `threshold` (1000 in the paper's Figure 1).
Decision to record in docs/reproduction_notes.md: counters are updated per example order
within a batch (exact), not per batch.
"""
import torch


class InactivityTracker:
    def __init__(self, layer_sizes, device="cpu"):
        self.counters = [torch.zeros(n, dtype=torch.long, device=device) for n in layer_sizes]

    @torch.no_grad()
    def update(self, activations):
        """activations: list of (B, H) post-ReLU tensors, one per hidden layer."""
        for i, acts in enumerate(activations):
            batch = acts.shape[0]
            fired = acts > 0                                   # (B, H)
            any_fired = fired.any(dim=0)                       # (H,)
            last = batch - 1 - torch.argmax(fired.flip(0).int(), dim=0)  # last example that fired
            self.counters[i] = torch.where(any_fired, batch - 1 - last, self.counters[i] + batch)

    def dead_fraction(self, threshold=1000):
        """Fraction of all hidden neurons whose counter is >= threshold."""
        dead = sum((c >= threshold).sum().item() for c in self.counters)
        total = sum(c.numel() for c in self.counters)
        return dead / total