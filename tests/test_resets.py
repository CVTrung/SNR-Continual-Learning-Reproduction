"""Checks for SNR statistics, reset decisions and reset mechanics, with hand-computed values.

Time unit is one optimizer step: a unit "fires" on a step when its activation is > 0 for at
least one example (or token) of the mini-batch.
"""
import math

import torch

from src.methods.resets import reinit_units
from src.methods.snr import SNR
from src.models.mlp import MLP
from src.models.transformer import DecoderTransformer


def _setup(optimizer="sgd", eta=0.01, estimator="geometric"):
    torch.manual_seed(0)
    model = MLP(in_dim=4, hidden=(3,), out_dim=2)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt, SNR(model, opt, eta=eta, estimator=estimator)


def _silent(b=4):
    return [torch.zeros(b, 3)]


def _firing(b=4):
    return [torch.ones(b, 3)]


def test_fired_is_any_example_in_batch():
    model, _, snr = _setup()
    acts = torch.zeros(4, 3)
    acts[2, 0] = 1.0                       # neuron 0 fires on one example only
    snr(model, [acts])
    assert snr.age[0].tolist() == [0, 1, 1]


def test_geometric_threshold_from_mean_silent_run():
    model, _, snr = _setup(eta=0.01)
    snr.run_mean[0] = torch.full((3,), 2.0)  # mean silent run 2 steps -> p = 0.5
    snr.seen[0] = torch.ones(3, dtype=torch.bool)
    # P(R >= a) = 0.5 ** (a - 1) <= 0.01  <=>  a >= 1 + log(0.01)/log(0.5) = 7.64 -> a = 8
    for _ in range(7):
        snr(model, _silent())
    assert snr.pop_resets() == 0
    snr(model, _silent())
    assert snr.pop_resets() == 3


def test_always_firing_neuron_survives_one_silent_step():
    """Regression for the old per-example rule that reset dense neurons after 1-2 silences."""
    model, _, snr = _setup(eta=0.08)
    for _ in range(50):
        snr(model, _firing())
    snr(model, _silent())
    assert snr.pop_resets() == 0


def test_completed_run_updates_ema():
    model, _, snr = _setup()
    snr.seen[0] = torch.ones(3, dtype=torch.bool)
    snr.run_mean[0] = torch.full((3,), 10.0)
    for _ in range(4):
        snr(model, _silent())
    snr(model, _firing())                  # completes a run of 4 steps
    # EMA beta = 0.1: 0.9 * 10 + 0.1 * 4 = 9.4
    assert torch.allclose(snr.run_mean[0], torch.full((3,), 9.4))
    assert snr.age[0].tolist() == [0, 0, 0]


def test_first_run_sets_mean_directly():
    model, _, snr = _setup()
    for _ in range(3):
        snr(model, _silent())
    snr(model, _firing())
    assert torch.allclose(snr.run_mean[0], torch.full((3,), 3.0))


def test_dead_from_init_is_reset_after_initial_threshold():
    model, _, snr = _setup(eta=0.08)
    expected = 1 + math.ceil(math.log(0.08) / math.log1p(-1.0 / snr.m_init))
    for _ in range(expected - 1):
        snr(model, _silent())
    assert snr.pop_resets() == 0
    snr(model, _silent())
    assert snr.pop_resets() == 3


def test_dense_unit_waits_for_minimum_silence_whatever_eta():
    """Review finding I1: with mu = 1 the geometric tail is 0 for a >= 2, so without a floor
    every dense unit was reset after 2 silent steps for any eta."""
    for eta in (0.08, 0.00125):
        model, _, snr = _setup(eta=eta)
        snr.run_mean[0] = torch.full((3,), 1.0)
        snr.seen[0] = torch.ones(3, dtype=torch.bool)
        for _ in range(snr.min_silence):
            snr(model, _silent())
        assert snr.pop_resets() == 0
        snr(model, _silent())
        assert snr.pop_resets() == 3


def test_reset_clears_statistics_and_outgoing_weights():
    model, _, snr = _setup()
    snr.min_silence = 1
    snr.run_mean[0] = torch.full((3,), 1.0)
    snr.seen[0] = torch.ones(3, dtype=torch.bool)
    old_in = model.hidden_layers[0].weight.detach().clone()
    snr(model, _silent())
    snr(model, _silent())                  # p = 1 -> any run of 2 steps is impossible
    assert snr.pop_resets() == 3
    assert torch.all(model.out.weight == 0)
    assert torch.all(model.hidden_layers[0].bias == 0)
    assert not torch.equal(model.hidden_layers[0].weight, old_in)
    assert snr.age[0].tolist() == [0, 0, 0]
    assert not snr.seen[0].any()


def test_histogram_estimator_uses_empirical_quantile():
    model, _, snr = _setup(eta=0.25, estimator="histogram")
    snr.min_silence = 1
    # neuron runs recorded: 1, 1, 1, 4  -> P(R > 1) = 0.25 <= eta, so q = 1: reset when a > 1
    for run in (1, 1, 1, 4):
        for _ in range(run):
            snr(model, _silent())
        snr(model, _firing())
    snr.end_task()
    assert snr.threshold[0].tolist() == [1, 1, 1]
    snr(model, _silent())
    assert snr.pop_resets() == 0
    snr(model, _silent())
    assert snr.pop_resets() == 3


def test_adam_state_is_cleared_for_reset_neurons():
    model, opt, snr = _setup("adam")
    snr.min_silence = 1
    model(torch.randn(8, 4)).sum().backward()
    opt.step()                             # creates exp_avg / exp_avg_sq
    snr.run_mean[0] = torch.full((3,), 1.0)
    snr.seen[0] = torch.ones(3, dtype=torch.bool)
    snr(model, _silent())
    snr(model, _silent())
    for param in (model.hidden_layers[0].weight, model.hidden_layers[0].bias, model.out.weight):
        assert torch.all(opt.state[param]["exp_avg"] == 0)
        assert torch.all(opt.state[param]["exp_avg_sq"] == 0)


def test_reinit_units_touches_only_selected_units():
    torch.manual_seed(0)
    model = MLP(in_dim=4, hidden=(3,), out_dim=2)
    w_in = model.hidden_layers[0].weight.detach().clone()
    w_out = model.out.weight.detach().clone()
    reinit_units(model.hidden_layers[0], model.out, torch.tensor([1]), optimizer=None)
    assert torch.equal(model.hidden_layers[0].weight[[0, 2]], w_in[[0, 2]])
    assert not torch.equal(model.hidden_layers[0].weight[1], w_in[1])
    assert torch.all(model.out.weight[:, 1] == 0)
    assert torch.equal(model.out.weight[:, [0, 2]], w_out[:, [0, 2]])


def test_snr_on_transformer_resets_ffn_units_only():
    torch.manual_seed(0)
    model = DecoderTransformer(vocab_size=11, context=8, n_ffn=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    snr = SNR(model, opt, eta=0.01, min_silence=1)
    snr.run_mean[0] = torch.full((6,), 1.0)
    snr.seen[0] = torch.ones(6, dtype=torch.bool)
    attn_before = [p.detach().clone() for p in model.attention_parameters()]
    acts = torch.zeros(2, 8, 6)            # (B, T, H): silent on every token
    snr(model, [acts])
    snr(model, [acts])
    assert snr.pop_resets() == 6
    assert torch.all(model.block.fc2.weight == 0)
    assert all(torch.equal(a, b) for a, b in zip(attn_before, model.attention_parameters()))
