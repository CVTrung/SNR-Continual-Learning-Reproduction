"""Save and restore the complete sequential-learning state, so long Colab runs can resume.

Status: executable.
A checkpoint holds model and optimizer state; the state of each method (SNR/CBP/ReDO
statistics, the L2 Init reference); the batch-shuffling generator; the global torch and NumPy
RNG states (resets and Shrink & Perturb draw from the global generator); the next task index;
and the per-task history written so far. Task permutations are rebuilt from (seed, task), so
they need no state. tests/test_train.py checks that a resumed run is bit-identical to an
uninterrupted one. Checkpoints go under checkpoints/, which is ignored by Git.
"""
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, generator, next_task, history, methods=None,
                    extra=None):
    """Write everything needed to continue at task `next_task` (atomic replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "next_task": next_task,
        "history": history,
        "methods": {k: m.state_dict() for k, m in (methods or {}).items()
                    if hasattr(m, "state_dict")},
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer, generator, methods=None):
    """Restore states in place; return {"next_task", "history", "extra"}."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    for key, method in (methods or {}).items():
        if key in state["methods"]:
            method.load_state_dict(state["methods"][key])
    return {"next_task": state["next_task"], "history": state["history"],
            "extra": state["extra"]}
