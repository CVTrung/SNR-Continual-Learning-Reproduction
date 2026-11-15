"""Define the mlp architecture.

Status: design scaffold; no executable implementation.
Inputs: Resolved architecture settings, input/output dimensions, and initialization seed.
Intended outputs: Model plus activations/parameter groups for interventions and diagnostics.
Responsibilities: Verify layer widths/depth, ReLU placement, initialization, output size, and optional Layer Normalization from the paper and cited architecture sources. Do not load datasets or generate tasks.
"""

"""Define the MLP architecture.

Status: first executable version.
Inputs: input size, hidden layer sizes, output size.
Outputs: logits, and optionally the post-ReLU activation of every hidden layer.
Responsibilities: architecture only. Do not load data or generate tasks.
UNVERIFIED decision (log in docs/reproduction_notes.md): 3 hidden layers of 100 units and
PyTorch's default initialization are provisional. Check the exact architecture in
Kumar et al. 2023b, which the paper says it follows.
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=(100, 100, 100), out_dim=10):
        super().__init__()
        dims = [in_dim, *hidden]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.out = nn.Linear(dims[-1], out_dim)

    def forward(self, x, return_activations=False):
        """Return logits; with return_activations also the list of hidden activations."""
        activations = []
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            activations.append(x)  # a neuron "fires" when its activation is > 0
        logits = self.out(x)
        return (logits, activations) if return_activations else logits