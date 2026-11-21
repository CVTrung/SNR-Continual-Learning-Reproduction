"""Shrink & Perturb (Ash & Adams 2020) as a comparison method.

Status: executable.
After every optimizer step: theta <- (1 - shrink) * theta + sigma * eps, eps ~ N(0, I), for
every parameter, biases included. The SNR paper's Table 7 sweeps (1 - p) = shrink and sigma
separately. The noise comes from torch's global generator, which the runner seeds, so runs are
reproducible. Use as an on_step hook: sp(model, activations).
"""
import torch


class ShrinkPerturb:
    def __init__(self, shrink=1e-5, sigma=1e-4):
        self.shrink = shrink
        self.sigma = sigma

    @torch.no_grad()
    def __call__(self, model, activations):
        for p in model.parameters():
            p.mul_(1.0 - self.shrink)
            if self.sigma:
                p.add_(torch.randn_like(p), alpha=self.sigma)

    def pop_resets(self):
        return 0

    def end_task(self, task=None):
        pass
