"""Continual Backprop (CBP, Dohare et al. 2024) as a comparison method.

Status: executable; implemented from the published description.
Per unit i of each hidden layer (or Transformer FFN):
- contribution utility u_i <- decay * u_i + (1 - decay) * mean|h_i| * sum_k |w_out[k, i]|
- age_i counts optimizer steps since the unit was (re)initialized.
- each step, a layer earns `replacement_rate * batch_size` replacements (SNR paper, Appendix C:
  "one replacement per layer every 1/r training examples"). Whole replacements are spent on
  the mature units (age > maturity) with the lowest utility. Credit is earned only while some
  unit is mature, so no burst of resets builds up during warm-up.
Reset mechanics are shared with SNR (resets.py). Defaults decay=0.99 and maturity=100 follow
Dohare et al.; replacement_rate is the Table 7 hyperparameter.
"""
import torch

from src.methods.resets import reinit_units


class CBP:
    def __init__(self, model, optimizer, replacement_rate=1e-4, maturity=100, decay=0.99,
                 batch_size=16):
        self.optimizer = optimizer
        self.groups = model.reset_groups()
        self.rate = replacement_rate
        self.maturity = maturity
        self.decay = decay
        self.batch_size = batch_size
        device = next(model.parameters()).device
        sizes = [layer.out_features for layer, _ in self.groups]
        self.utility = [torch.zeros(n, device=device) for n in sizes]
        self.age = [torch.zeros(n, dtype=torch.long, device=device) for n in sizes]
        self.credit = [0.0 for _ in sizes]
        self._resets = 0
        self.reset_listeners = []

    def pop_resets(self):
        count, self._resets = self._resets, 0
        return count

    @torch.no_grad()
    def __call__(self, model, activations):
        for i, acts in enumerate(activations):
            in_layer, out_layer = self.groups[i]
            h = acts.reshape(-1, acts.shape[-1]).abs().mean(dim=0)
            contrib = h * out_layer.weight.abs().sum(dim=0)
            self.utility[i] = self.decay * self.utility[i] + (1 - self.decay) * contrib
            self.age[i] += 1
            eligible = self.age[i] > self.maturity
            if not eligible.any():
                continue
            self.credit[i] += self.rate * self.batch_size
            n = min(int(self.credit[i]), int(eligible.sum()))
            if n == 0:
                continue
            self.credit[i] -= n
            inf = torch.full_like(self.utility[i], float("inf"))
            scores = torch.where(eligible, self.utility[i], inf)
            idx = torch.topk(scores, n, largest=False).indices
            self._resets += reinit_units(in_layer, out_layer, idx, self.optimizer)
            self.utility[i][idx] = 0.0
            self.age[i][idx] = 0
            for listener in self.reset_listeners:
                listener(i, idx)

    def end_task(self, task=None):
        pass

    def state_dict(self):
        return {"utility": self.utility, "age": self.age, "credit": self.credit}

    def load_state_dict(self, state):
        dev = self.age[0].device
        self.utility = [t.to(dev) for t in state["utility"]]
        self.age = [t.to(dev) for t in state["age"]]
        self.credit = list(state["credit"])
