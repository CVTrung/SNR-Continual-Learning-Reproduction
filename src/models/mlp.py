"""Define the MLP used for Permuted MNIST.

Status: executable.
Inputs: input size, hidden layer sizes, output size, optional Layer Normalization.
Outputs: logits, and optionally the post-ReLU activation of every hidden layer.
Responsibilities: architecture and initialization only. Do not load data or generate tasks.
Decisions (docs/reproduction_notes.md):
- 784-100-100-10 (two hidden layers of 100 ReLU units). The paper says the MLP is identical to
  Kumar et al. (2023b); the authors' released code (verification only) uses two Dense(100)
  layers, so the earlier provisional 3 x 100 MLP was replaced.
- LeCun-normal weights (truncated normal, variance 1/fan_in) and zero biases, which is the
  default of the Flax Dense layer used by the authors. Resets redraw from the same rule.
- Layer Normalization (a model option, Table 1 "Layer Norm.") sits between each hidden
  linear layer and its ReLU.
"""
import math

import torch
from torch import nn


@torch.no_grad()
def lecun_normal_(tensor, fan_in):
    """Fill `tensor` in place from a truncated normal (+-2 std) with variance 1/fan_in.

    The 0.8796 factor is the standard deviation of a unit normal truncated at +-2, so the
    variance after truncation is exactly 1/fan_in. Returns the tensor.
    """
    std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def init_linear_(layer):
    """LeCun-normal weights and zero bias for an nn.Linear layer."""
    lecun_normal_(layer.weight, layer.in_features)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """ReLU MLP whose forward pass can also return hidden activations.

    Parameters: in_dim, hidden (tuple of widths), out_dim, layer_norm (bool).
    """

    def __init__(self, in_dim=784, hidden=(100, 100), out_dim=10, layer_norm=False):
        super().__init__()
        dims = [in_dim, *hidden]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.norms = nn.ModuleList(nn.LayerNorm(b) for b in hidden) if layer_norm else None
        self.out = nn.Linear(dims[-1], out_dim)
        for layer in [*self.hidden_layers, self.out]:
            init_linear_(layer)

    def forward(self, x, return_activations=False):
        """Return logits; with return_activations also the list of (B, H) hidden activations."""
        activations = []
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = torch.relu(x)
            activations.append(x)  # a unit "fires" when its activation is > 0
        logits = self.out(x)
        return (logits, activations) if return_activations else logits

    def reset_groups(self):
        """List of (incoming layer, outgoing layer) per hidden layer, aligned with activations."""
        nxt = [*self.hidden_layers[1:], self.out]
        return list(zip(self.hidden_layers, nxt))
