"""Shape and activation checks for the MLP."""
import torch

from src.models.mlp import MLP


def test_output_and_activation_shapes():
    model = MLP()
    logits, acts = model(torch.zeros(5, 784), return_activations=True)
    assert logits.shape == (5, 10)
    assert len(acts) == 3 and all(a.shape == (5, 100) for a in acts)


def test_activations_are_relu_outputs():
    torch.manual_seed(0)
    model = MLP()
    _, acts = model(torch.randn(8, 784), return_activations=True)
    assert all((a >= 0).all() for a in acts)


def test_forward_without_flag_returns_logits_only():
    out = MLP()(torch.zeros(2, 784))
    assert out.shape == (2, 10)