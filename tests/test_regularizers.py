"""Checks for the L2 and L2 Init penalties (with parameter masks), hand-computed values."""
import torch

from src.methods.regularization import L2, L2Init
from src.models.mlp import MLP
from src.models.transformer import DecoderTransformer


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


def test_attention_only_mask_ignores_ffn():
    torch.manual_seed(0)
    model = DecoderTransformer(vocab_size=10, context=4, n_ffn=8)
    for p in model.parameters():
        p.data.fill_(0.0)
    for p in model.ffn_parameters():
        p.data.fill_(1.0)
    assert L2(1.0, select="attention")(model).item() == 0.0
    for p in model.attention_parameters():
        p.data.fill_(1.0)
    n_attn = sum(p.numel() for p in model.attention_parameters())
    assert abs(L2(1.0, select="attention")(model).item() - 0.5 * n_attn) < 1e-4


def test_legacy_module_reexports():
    from src.methods import regularizers
    assert regularizers.L2 is L2 and regularizers.L2Init is L2Init
