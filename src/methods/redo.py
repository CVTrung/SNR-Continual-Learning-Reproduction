"""ReDO (Sokar et al. 2023) as a comparison method.

Status: executable; implemented from the published description.
Score of unit i in a layer: s_i = mean|h_i| / ((1/H) * sum_j mean|h_j|), where mean|h| is the
average absolute activation over all steps since the last check. Every `every_tasks` tasks,
units with s_i <= tau are reset with the shared mechanics in resets.py. Following the SNR
paper's Appendix C, the reset frequency is measured in tasks; tau and the frequency are the
Table 7 grid. tau = 0 resets only units that were never active during the period.
"""
import torch

from src.methods.resets import reinit_units


class ReDO:
    def __init__(self, model, optimizer, tau=0.0, every_tasks=1):
        self.optimizer = optimizer
        self.groups = model.reset_groups()
        self.tau = tau
        self.every_tasks = every_tasks
        device = next(model.parameters()).device
        self.sum_act = [torch.zeros(layer.out_features, device=device) for layer, _ in self.groups]
        self.steps = 0
        self._resets = 0
        self.reset_listeners = []

    def pop_resets(self):
        count, self._resets = self._resets, 0
        return count

    @torch.no_grad()
    def __call__(self, model, activations):
        for i, acts in enumerate(activations):
            self.sum_act[i] += acts.reshape(-1, acts.shape[-1]).abs().mean(dim=0)
        self.steps += 1

    @torch.no_grad()
    def end_task(self, task):
        if (task + 1) % self.every_tasks != 0 or self.steps == 0:
            return
        for i, total in enumerate(self.sum_act):
            mean = total / self.steps
            score = mean / (mean.mean() + 1e-12)
            idx = (score <= self.tau).nonzero(as_tuple=True)[0]
            in_layer, out_layer = self.groups[i]
            self._resets += reinit_units(in_layer, out_layer, idx, self.optimizer)
            for listener in self.reset_listeners:
                listener(i, idx)
            total.zero_()
        self.steps = 0

    def state_dict(self):
        return {"sum_act": self.sum_act, "steps": self.steps}

    def load_state_dict(self, state):
        dev = self.sum_act[0].device
        self.sum_act, self.steps = [t.to(dev) for t in state["sum_act"]], state["steps"]
