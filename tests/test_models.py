"""Shape, activation and parameter-group checks for the MLP and the decoder Transformer."""
import torch

from src.models.mlp import MLP
from src.models.transformer import DecoderTransformer


def test_output_and_activation_shapes():
    model = MLP()
    logits, acts = model(torch.zeros(5, 784), return_activations=True)
    assert logits.shape == (5, 10)
    assert len(acts) == 2 and all(a.shape == (5, 100) for a in acts)


def test_default_mlp_matches_reference_size():
    # 784-100-100-10: 78500 + 10100 + 1010 parameters
    assert sum(p.numel() for p in MLP().parameters()) == 89_610


def test_activations_are_relu_outputs():
    torch.manual_seed(0)
    model = MLP()
    _, acts = model(torch.randn(8, 784), return_activations=True)
    assert all((a >= 0).all() for a in acts)


def test_forward_without_flag_returns_logits_only():
    out = MLP()(torch.zeros(2, 784))
    assert out.shape == (2, 10)


def test_layer_norm_option_adds_norm_layers():
    model = MLP(layer_norm=True)
    assert len(model.norms) == 2
    _, acts = model(torch.randn(4, 784), return_activations=True)
    assert all((a >= 0).all() for a in acts)


def test_reset_groups_pair_incoming_and_outgoing_layers():
    model = MLP()
    groups = model.reset_groups()
    assert groups[0] == (model.hidden_layers[0], model.hidden_layers[1])
    assert groups[1] == (model.hidden_layers[1], model.out)


def test_biases_start_at_zero():
    assert all(torch.all(layer.bias == 0) for layer in MLP().hidden_layers)


def test_transformer_shapes_and_groups():
    torch.manual_seed(0)
    model = DecoderTransformer(vocab_size=50, context=16)
    x = torch.randint(0, 50, (3, 16))
    logits, acts, probs = model(x, return_activations=True, return_attention=True)
    assert logits.shape == (3, 16, 50)
    assert len(acts) == 1 and acts[0].shape == (3, 16, 256)
    assert probs.shape == (3, 2, 16, 16)
    attn = {id(p) for p in model.attention_parameters()}
    ffn = {id(p) for p in model.ffn_parameters()}
    assert attn and ffn and not (attn & ffn)
    assert model.reset_groups() == [(model.block.fc1, model.block.fc2)]


def test_transformer_is_causal():
    torch.manual_seed(0)
    model = DecoderTransformer(vocab_size=20, context=8)
    x = torch.randint(0, 20, (1, 8))
    y = x.clone()
    y[0, 5:] = (y[0, 5:] + 1) % 20         # change only future tokens
    assert torch.allclose(model(x)[0, :5], model(y)[0, :5], atol=1e-6)
