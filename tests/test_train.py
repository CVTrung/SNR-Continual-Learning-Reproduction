"""Checks for the shared training loop, checkpoint resume and the tracking fallback."""
import numpy as np
import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.models.mlp import MLP
from src.models.transformer import DecoderTransformer
from src.tracking import Tracker
from src.train import train_one_task, train_sequence


def test_classification_returns_online_accuracy_and_loss():
    torch.manual_seed(0)
    model = MLP(in_dim=4, hidden=(8,), out_dim=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(32, 4)
    y = (x[:, 0] > 0).long()
    out = train_one_task(model, opt, x, y, batch_size=8, epochs=2,
                         generator=torch.Generator().manual_seed(0))
    assert 0.0 <= out["accuracy"] <= 1.0 and out["loss"] > 0


def test_language_model_task_reduces_final_epoch_loss():
    torch.manual_seed(0)
    model = DecoderTransformer(vocab_size=5, context=8, n_ffn=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    seqs = torch.tensor(np.tile(np.arange(9) % 5, (16, 1)))
    first = train_one_task(model, opt, seqs, None, batch_size=8, epochs=1, task_type="lm")
    later = train_one_task(model, opt, seqs, None, batch_size=8, epochs=20, task_type="lm")
    assert later["loss"] < first["loss"]


def _run(n_tasks, start_task=0, model=None, opt=None, generator=None):
    def make(t):
        rng = np.random.default_rng(t)
        x = rng.standard_normal((16, 4)).astype(np.float32)
        return x, (x[:, 0] > 0).astype(np.int64)
    return train_sequence(model, opt, make, num_tasks=n_tasks, start_task=start_task,
                          batch_size=8, generator=generator)


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    torch.manual_seed(0)
    model_a = MLP(in_dim=4, hidden=(8,), out_dim=2)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-2)
    full = _run(4, model=model_a, opt=opt_a, generator=torch.Generator().manual_seed(1))

    torch.manual_seed(0)
    model_b = MLP(in_dim=4, hidden=(8,), out_dim=2)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(1)
    part = _run(2, model=model_b, opt=opt_b, generator=gen)
    save_checkpoint(tmp_path / "ck.pt", model=model_b, optimizer=opt_b, generator=gen,
                    next_task=2, history={"acc": [r["accuracy"] for r in part]})

    torch.manual_seed(123)                 # resume must not depend on the global seed
    model_c = MLP(in_dim=4, hidden=(8,), out_dim=2)
    opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-2)
    gen_c = torch.Generator()
    state = load_checkpoint(tmp_path / "ck.pt", model=model_c, optimizer=opt_c, generator=gen_c)
    rest = _run(4, start_task=state["next_task"], model=model_c, opt=opt_c, generator=gen_c)
    resumed = state["history"]["acc"] + [r["accuracy"] for r in rest]
    assert resumed == [r["accuracy"] for r in full]
    for p, q in zip(model_a.parameters(), model_c.parameters()):
        assert torch.equal(p, q)


def test_tracker_none_backend_keeps_local_rows():
    tracker = Tracker(backend="none", project="p", name="n", config={"a": 1})
    tracker.log({"task": 0, "acc": 0.5})
    tracker.finish()
    assert tracker.rows == [{"task": 0, "acc": 0.5}]
    assert tracker.url is None
