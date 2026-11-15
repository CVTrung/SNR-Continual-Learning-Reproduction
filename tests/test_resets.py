"""Checks for SNR statistics and resets, with hand-computed expected values."""
import torch

from src.methods.snr import SNR
from src.models.mlp import MLP


def _setup(optimizer="sgd", eta=0.01):
    torch.manual_seed(0)
    model = MLP(in_dim=4, hidden=(3,), out_dim=2)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
    snr = SNR(model, opt, eta=eta)
    snr.mean_gap[0] = torch.full((3,), 2.0)        # p = 0.5
    snr.seen[0] = torch.ones(3, dtype=torch.bool)
    return model, opt, snr


def test_silent_neurons_are_reset():
    model, _, snr = _setup()
    old_in = model.hidden_layers[0].weight.detach().clone()
    snr(model, [torch.zeros(8, 3)])
    # 8 silent examples: 0.5**8 = 0.0039 <= 0.01, so all three are reset
    assert snr.pop_resets() == 3
    assert torch.all(model.out.weight == 0)
    assert not torch.equal(model.hidden_layers[0].weight, old_in)
    assert snr.counter[0].tolist() == [0, 0, 0]
    assert torch.allclose(snr.mean_gap[0], torch.full((3,), snr.m_init))


def test_firing_neurons_are_not_reset_and_mean_updates():
    model, _, snr = _setup()
    snr(model, [torch.ones(8, 3)])
    assert snr.pop_resets() == 0
    # span = 0 + 7 + 1 = 8 over 8 intervals -> 1.0 ; EMA: 0.9 * 2 + 0.1 * 1 = 1.9
    assert torch.allclose(snr.mean_gap[0], torch.full((3,), 1.9))
    assert snr.counter[0].tolist() == [0, 0, 0]


def test_first_interval_sets_mean_and_counter():
    model, _, snr = _setup()
    snr.seen[0] = torch.zeros(3, dtype=torch.bool)
    snr.counter[0] = torch.tensor([5, 5, 5])
    snr.mean_gap[0] = torch.full((3,), 10_000.0)
    acts = torch.zeros(4, 3)
    acts[2, :] = 1.0                                # every neuron fires on example 2 only
    snr(model, [acts])
    # one interval spanning 5 + 2 + 1 = 8 examples; 1 silent example after the firing
    assert torch.allclose(snr.mean_gap[0], torch.full((3,), 8.0))
    assert snr.counter[0].tolist() == [1, 1, 1]
    assert snr.pop_resets() == 0


def test_adam_state_is_cleared_for_reset_neurons():
    model, opt, snr = _setup("adam")
    model(torch.randn(8, 4)).sum().backward()
    opt.step()                                      # creates exp_avg / exp_avg_sq
    snr(model, [torch.zeros(8, 3)])                 # all three neurons reset
    for param in (model.hidden_layers[0].weight, model.hidden_layers[0].bias, model.out.weight):
        assert torch.all(opt.state[param]["exp_avg"] == 0)
        assert torch.all(opt.state[param]["exp_avg_sq"] == 0)