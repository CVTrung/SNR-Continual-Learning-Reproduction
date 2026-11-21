"""Self-Normalized Resets (SNR), Algorithm 1 of the paper, for MLP neurons and Transformer FFN units.

Status: executable; implemented from the paper's text. The authors' code was read for verification
only, never copied.
Rule: reset unit i when P(R_i >= a_i) <= eta, where a_i is the number of consecutive optimizer
steps on which the unit has not fired, and R_i is the length of a silent run of that unit.

Team decisions (docs/reproduction_notes.md records the evidence):
- Time unit = optimizer step. A unit fires on a step if its activation is > 0 for any
  example or token of the mini-batch. The authors' code measures firing the same way, and the
  paper's Figure 3 counts inactivity in steps. Our first version counted single examples
  inside a batch. Then neurons that fire on almost every example had an estimated firing
  probability near 1 and were reset after one or two silent examples, which is the over-reset
  seen in our first eta sweep.
- Only completed silent runs (length >= 1 step) enter the estimate. Consecutive firings are not
  "intervals". The authors' code skips age-0 entries of its histogram in the same way.
- estimator="geometric" (paper footnote 3): track the mean silent-run length mu_i and assume R_i
  is geometric on {1, 2, ...} with p = 1/mu, so P(R >= a) = (1 - p)^(a-1). The mean uses an
  exponential moving average (factor `beta`) in place of the paper's fixed trailing window.
  This keeps one number per unit. The first completed run sets mu directly. Before that
  mu = m_init steps, which also sets how long a unit that is dead from initialization waits.
- estimator="histogram" (paper main text): an empirical histogram of silent-run lengths per
  unit, collected over a trailing window of `window_tasks` tasks. At the end of each window,
  threshold_i becomes the smallest r with empirical P(R > r) <= eta, and the unit is reset
  when a_i > threshold_i. Units with no completed run in the window keep their threshold.
- A reset unit starts fresh. Its age, mean and histogram are cleared, and so is its own
  threshold in histogram mode. The mechanics are in resets.py: incoming weights redrawn,
  bias 0, outgoing weights 0, Adam moments of those entries 0.
Use as the `on_step` hook of src/train.py: snr(model, activations). Call end_task() after each task.
"""
import math

import torch

from src.methods.resets import fired_per_step, reinit_units


class SNR:
    def __init__(self, model, optimizer, eta=0.01, estimator="geometric", beta=0.1,
                 m_init=1000.0, window_tasks=1, tau_max=20_000):
        if estimator not in ("geometric", "histogram"):
            raise ValueError(f"unknown estimator: {estimator}")
        self.optimizer = optimizer
        self.groups = model.reset_groups()
        self.eta = eta
        self.log_eta = math.log(eta)
        self.estimator = estimator
        self.beta = beta
        self.m_init = float(m_init)
        self.window_tasks = window_tasks
        self.tau_max = int(tau_max)
        device = next(model.parameters()).device
        sizes = [layer.out_features for layer, _ in self.groups]
        self.age = [torch.zeros(n, dtype=torch.long, device=device) for n in sizes]
        self.run_mean = [torch.full((n,), self.m_init, device=device) for n in sizes]
        self.seen = [torch.zeros(n, dtype=torch.bool, device=device) for n in sizes]
        self.tau_init = self._geometric_threshold(self.m_init)
        if estimator == "histogram":
            self.hist = [torch.zeros(n, self.tau_max + 1, dtype=torch.long, device=device)
                         for n in sizes]
            self.threshold = [torch.full((n,), self.tau_init, dtype=torch.long, device=device)
                              for n in sizes]
        self._tasks_in_window = 0
        self._resets = 0
        self.reset_listeners = []          # callables f(layer_index, idx), e.g. diagnostics

    def _geometric_threshold(self, mu):
        """Largest silent run a geometric(1/mu) unit may show before it is reset."""
        p = min(1.0, 1.0 / mu)
        if p >= 1.0:
            return 1
        return min(self.tau_max, math.ceil(self.log_eta / math.log1p(-p)))

    def pop_resets(self):
        """Return the number of units reset since the last call, then clear it."""
        count, self._resets = self._resets, 0
        return count

    @torch.no_grad()
    def __call__(self, model, activations):
        for i, acts in enumerate(activations):
            fired = fired_per_step(acts)
            self._record_runs(i, fired)
            self.age[i] = torch.where(fired, torch.zeros_like(self.age[i]), self.age[i] + 1)
            idx = self._reset_mask(i).nonzero(as_tuple=True)[0]
            if len(idx):
                self._reset(i, idx)

    def _record_runs(self, i, fired):
        ended = fired & (self.age[i] > 0)
        if not ended.any():
            return
        runs = self.age[i]
        if self.estimator == "geometric":
            r = runs.float()
            blended = (1 - self.beta) * self.run_mean[i] + self.beta * r
            new = torch.where(self.seen[i], blended, r)
            self.run_mean[i] = torch.where(ended, new, self.run_mean[i])
        else:
            units = ended.nonzero(as_tuple=True)[0]
            bins = runs[units].clamp(max=self.tau_max)
            self.hist[i].index_put_((units, bins), torch.ones_like(bins), accumulate=True)
        self.seen[i] |= ended

    def _reset_mask(self, i):
        a = self.age[i]
        if self.estimator == "histogram":
            return a > self.threshold[i]
        p = (1.0 / self.run_mean[i]).clamp(max=1.0)
        # (1 - p)^(a - 1) <= eta  <=>  (a - 1) * log(1 - p) <= log(eta); p = 1 gives -inf for a >= 2
        log_tail = (a - 1).float() * torch.log1p(-p)
        return (a >= 1) & (torch.nan_to_num(log_tail, nan=0.0) <= self.log_eta)

    def _reset(self, i, idx):
        in_layer, out_layer = self.groups[i]
        self._resets += reinit_units(in_layer, out_layer, idx, self.optimizer)
        self.age[i][idx] = 0
        self.run_mean[i][idx] = self.m_init
        self.seen[i][idx] = False
        if self.estimator == "histogram":
            self.hist[i][idx] = 0
            self.threshold[i][idx] = self.tau_init
        for listener in self.reset_listeners:
            listener(i, idx)

    @torch.no_grad()
    def end_task(self, task=None):
        """Histogram mode: refresh thresholds at the end of each `window_tasks`-task window."""
        if self.estimator != "histogram":
            return
        self._tasks_in_window += 1
        if self._tasks_in_window < self.window_tasks:
            return
        self._tasks_in_window = 0
        for i, hist in enumerate(self.hist):
            counts = hist[:, 1:].float()
            total = counts.sum(dim=1, keepdim=True)
            cdf = counts.cumsum(dim=1) / total.clamp(min=1)
            q = torch.argmax((cdf >= 1 - self.eta - 1e-12).int(), dim=1) + 1
            has_data = total.squeeze(1) > 0
            self.threshold[i] = torch.where(has_data, q, self.threshold[i])
            hist.zero_()

    def state_dict(self):
        out = {"age": self.age, "run_mean": self.run_mean, "seen": self.seen,
               "tasks_in_window": self._tasks_in_window}
        if self.estimator == "histogram":
            out.update(hist=self.hist, threshold=self.threshold)
        return out

    def load_state_dict(self, state):
        self.age, self.run_mean, self.seen = state["age"], state["run_mean"], state["seen"]
        self._tasks_in_window = state["tasks_in_window"]
        if self.estimator == "histogram":
            self.hist, self.threshold = state["hist"], state["threshold"]
