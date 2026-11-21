"""Checks for the CBP, ReDO and Shrink & Perturb baselines (decision rules, not accuracy)."""
import torch

from src.methods.cbp import CBP
from src.methods.redo import ReDO
from src.methods.shrink_perturb import ShrinkPerturb
from src.models.mlp import MLP


def _model():
    torch.manual_seed(0)
    return MLP(in_dim=4, hidden=(5,), out_dim=2)


def test_redo_resets_only_dormant_units_at_period_end():
    model = _model()
    redo = ReDO(model, None, tau=0.1, every_tasks=2)
    acts = torch.ones(8, 5)
    acts[:, 3] = 0.0                       # unit 3 never active
    redo(model, [acts])
    redo.end_task(0)                       # period not finished
    assert redo.pop_resets() == 0
    redo(model, [acts])
    redo.end_task(1)
    assert redo.pop_resets() == 1
    assert torch.all(model.out.weight[:, 3] == 0)


def test_redo_score_is_normalized_mean_activity():
    model = _model()
    redo = ReDO(model, None, tau=0.5, every_tasks=1)
    acts = torch.ones(4, 5)
    acts[:, 0] = 0.6                       # score 0.6 / mean(0.6,1,1,1,1)=0.92 -> 0.652 > 0.5
    acts[:, 1] = 0.2                       # score 0.2 / mean(0.6,0.2,1,1,1)=0.76 -> 0.263 <= 0.5
    redo(model, [acts])
    redo.end_task(0)
    assert redo.pop_resets() == 1
    assert torch.all(model.out.weight[:, 1] == 0)
    assert not torch.all(model.out.weight[:, 0] == 0)


def test_cbp_replaces_lowest_utility_mature_units():
    model = _model()
    cbp = CBP(model, None, replacement_rate=1 / 16, maturity=2, decay=0.0, batch_size=16)
    acts = torch.ones(16, 5)
    acts[:, 2] = 0.0                       # unit 2 has zero contribution utility
    cbp(model, [acts])                     # age 1: not mature yet
    assert cbp.pop_resets() == 0
    cbp(model, [acts])
    cbp(model, [acts])                     # mature; 16 examples * 1/16 = 1 replacement per step
    assert cbp.pop_resets() >= 1
    assert torch.all(model.out.weight[:, 2] == 0)


def test_shrink_perturb_shrinks_without_noise():
    model = _model()
    before = [p.detach().clone() for p in model.parameters()]
    ShrinkPerturb(shrink=0.1, sigma=0.0)(model, None)
    for p, b in zip(model.parameters(), before):
        assert torch.allclose(p, 0.9 * b)
