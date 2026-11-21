"""Architecture-agnostic reset mechanics shared by SNR, ReDO and CBP.

Status: executable.
A "unit" is output row `idx` of an incoming nn.Linear (an MLP neuron or a Transformer FFN
unit). Resetting it:
- redraws its incoming weights with the model's LeCun-normal init and sets its bias to 0,
- sets its outgoing weights (column `idx` of the next nn.Linear) to 0, so the network
  function is unchanged at the moment of the reset,
- zeroes the Adam moments (exp_avg, exp_avg_sq) of exactly those entries; SGD has no state.
Decision rules (which units to reset) live in snr.py / redo.py / cbp.py, not here.
"""
import torch

from src.models.mlp import lecun_normal_


def _moments(optimizer, param):
    if optimizer is None:
        return []
    state = optimizer.state.get(param, {})
    return [state[key] for key in ("exp_avg", "exp_avg_sq") if key in state]


@torch.no_grad()
def reinit_units(in_layer, out_layer, idx, optimizer=None):
    """Reset units `idx` (LongTensor) of `in_layer`; `out_layer` consumes them.

    Returns the number of units reset.
    """
    if len(idx) == 0:
        return 0
    fresh = torch.empty(len(idx), in_layer.in_features, device=in_layer.weight.device)
    in_layer.weight[idx] = lecun_normal_(fresh, in_layer.in_features)
    if in_layer.bias is not None:
        in_layer.bias[idx] = 0.0
    out_layer.weight[:, idx] = 0.0
    for tensor in _moments(optimizer, in_layer.weight):
        tensor[idx] = 0.0
    if in_layer.bias is not None:
        for tensor in _moments(optimizer, in_layer.bias):
            tensor[idx] = 0.0
    for tensor in _moments(optimizer, out_layer.weight):
        tensor[:, idx] = 0.0
    return len(idx)


def fired_per_step(acts):
    """(B, ..., H) activations -> (H,) bool: did each unit fire on any example/token."""
    return (acts > 0).reshape(-1, acts.shape[-1]).any(dim=0)
