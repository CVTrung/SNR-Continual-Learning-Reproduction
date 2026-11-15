"""Self-Normalized Resets (SNR) for the MLP.

Status: first executable version, implemented from the paper's Algorithm 1 and Section 2.
Rule: reset neuron i when (1 - p_i)^a_i <= eta, where a_i is the number of consecutive
examples on which it did not fire and p_i = 1 / m_i is the firing probability implied by
the mean inter-firing time m_i under a geometric distribution.
TEAM DECISIONS (the paper does not specify them; log in docs/reproduction_notes.md):
- m_i is an exponential moving average (factor `beta`) of COMPLETED inter-firing intervals,
  not a fixed-length trailing window (a time window would forget dead neurons).
- the first observed interval sets m_i directly; before it m_i = m_init.
- examples inside a batch are treated as sequential.
- reset: input weights and bias redrawn from nn.Linear's default range, output weights set
  to 0, and the Adam moments of exactly those entries set to 0.
Use as the `on_step` hook of src/train.py: snr(model, activations).
"""
import math

import torch


class SNR:
    def __init__(self, model, optimizer, eta=0.00125, beta=0.1, m_init=10_000.0):
        sizes = [layer.out_features for layer in model.hidden_layers]
        device = next(model.parameters()).device
        self.optimizer = optimizer
        self.log_eta = math.log(eta)
        self.beta = beta
        self.m_init = m_init
        self.counter = [torch.zeros(n, dtype=torch.long, device=device) for n in sizes]
        self.mean_gap = [torch.full((n,), m_init, device=device) for n in sizes]
        self.seen = [torch.zeros(n, dtype=torch.bool, device=device) for n in sizes]
        self._resets = 0

    def pop_resets(self):
        """Return the number of neurons reset since the last call, then clear it."""
        count, self._resets = self._resets, 0
        return count

    @torch.no_grad()
    def __call__(self, model, activations):
        for i, acts in enumerate(activations):
            self._update_statistics(i, acts)
            to_reset = self._reset_mask(i)
            if to_reset.any():
                self._reset_neurons(model, i, to_reset.nonzero(as_tuple=True)[0])

    def _update_statistics(self, i, acts):
        batch = acts.shape[0]
        fired = acts > 0                                          # (B, H)
        any_fired = fired.any(dim=0)
        k = fired.sum(dim=0)                                      # firings in this batch
        last = batch - 1 - torch.argmax(fired.flip(0).int(), dim=0)
        # the k intervals completed in this batch span (silent so far) + last + 1 examples
        seg_mean = (self.counter[i] + last + 1).float() / k.clamp(min=1)
        first = any_fired & ~self.seen[i]
        blended = (1 - self.beta) * self.mean_gap[i] + self.beta * seg_mean
        new_mean = torch.where(first, seg_mean, blended)
        self.mean_gap[i] = torch.where(any_fired, new_mean, self.mean_gap[i])
        self.seen[i] = self.seen[i] | any_fired
        self.counter[i] = torch.where(any_fired, batch - 1 - last, self.counter[i] + batch)

    def _reset_mask(self, i):
        p = (1.0 / self.mean_gap[i]).clamp(max=1 - 1e-6)
        # (1 - p)^a <= eta  <=>  a * log(1 - p) <= log(eta)
        return self.counter[i].float() * torch.log1p(-p) <= self.log_eta

    def _state(self, param):
        state = self.optimizer.state.get(param, {})
        return [state[key] for key in ("exp_avg", "exp_avg_sq") if key in state]

    def _reset_neurons(self, model, i, idx):
        layer = model.hidden_layers[i]
        nxt = model.hidden_layers[i + 1] if i + 1 < len(model.hidden_layers) else model.out
        bound = 1.0 / math.sqrt(layer.in_features)                # nn.Linear default range
        device = layer.weight.device
        layer.weight.data[idx] = torch.empty(len(idx), layer.in_features, device=device).uniform_(-bound, bound)
        layer.bias.data[idx] = torch.empty(len(idx), device=device).uniform_(-bound, bound)
        nxt.weight.data[:, idx] = 0.0
        for tensor in self._state(layer.weight) + self._state(layer.bias):
            tensor[idx] = 0.0
        for tensor in self._state(nxt.weight):
            tensor[:, idx] = 0.0
        self.counter[i][idx] = 0
        self.mean_gap[i][idx] = self.m_init
        self.seen[i][idx] = False
        self._resets += len(idx)