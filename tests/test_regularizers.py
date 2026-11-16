"""Checks for the L2 and L2 Init penalties with hand-computed values."""
import torch

from src.methods.regularizers import L2, L2Init
from src.models.mlp import MLP


def _model():
    model = MLP(in_dim=2, hidden=(2,), out_dim=1)   # 4 + 2 + 2 + 1 = 9 parameters
    for p in model.parameters():
        p.data.fill_(1.0)
    return model


def test_l2_value():
    # 0.5 * 0.2 * (9 ones squared) = 0.9
    assert abs(L2(0.2)(_model()).item() - 0.9) < 1e-6


def test_l2init_is_zero_at_init_and_grows_when_moved():
    model = _model()
    penalty = L2Init(model, 0.2)
    assert penalty(model).item() == 0.0
    for p in model.parameters():
        p.data += 1.0                               # every parameter moves by 1
    assert abs(penalty(model).item() - 0.9) < 1e-6