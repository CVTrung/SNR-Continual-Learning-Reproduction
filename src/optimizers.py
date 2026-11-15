"""Build the optimizer.

Status: first executable version.
Settings from the paper's Appendix C that still need verification by the team:
SGD lr 0.01 (0.1 for Random Label MNIST); Adam lr 0.001 with eps 1e-7.
PyTorch's default Adam eps is 1e-8, so eps is passed explicitly.
"""
import torch


def build_optimizer(name, params, lr, eps=1e-7):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, eps=eps)
    raise ValueError(f"unknown optimizer: {name}")