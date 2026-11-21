"""Experiment tracking with a choice of backend: W&B, Trackio, or local only.

Status: executable.
- backend="wandb": one public W&B project (course requirement). It needs `wandb login` or a
  WANDB_API_KEY in the environment. The key is never written to the repository.
- backend="trackio": Trackio has the same init/log/finish API. Set TRACKIO_SPACE_ID (for
  example "<user>/snr-reproduction") to sync to a Hugging Face Space dashboard. This is the
  dashboard the milestone-2 deck cites.
- backend="none": nothing leaves the machine.
In every mode the rows are also kept in memory, and the runner writes them to
results/metrics/*.csv, so local evidence survives a remote failure. If remote init fails, the
tracker warns and falls back to local; it never invents a run link. Phase tags (tune / eval /
sweep / smoke) separate tuning from evaluation runs.
"""
import os
import warnings


class Tracker:
    def __init__(self, backend="none", project="snr-reproduction", name=None, config=None,
                 tags=None, group=None):
        self.backend = backend
        self.rows = []
        self.url = None
        self._run = None
        self._mod = None
        if backend == "none":
            return
        try:
            if backend == "wandb":
                import wandb
                self._run = wandb.init(project=project, name=name, config=config or {},
                                       tags=tags or [], group=group,
                                       entity=os.environ.get("WANDB_ENTITY") or None)
                self.url = getattr(self._run, "url", None)
                self._mod = wandb
            elif backend == "trackio":
                import trackio
                space = os.environ.get("TRACKIO_SPACE_ID") or None
                trackio.init(project=project, name=name, config=config or {}, space_id=space)
                self.url = f"https://huggingface.co/spaces/{space}" if space else None
                self._mod = trackio
            else:
                raise ValueError(f"unknown tracking backend: {backend}")
        except Exception as exc:  # remote tracking must never kill a long run
            warnings.warn(f"tracking backend {backend!r} unavailable ({exc}); logging locally")
            self.backend = "none"
            self._mod = None

    def log(self, row):
        """Record one row (a dict of scalars) locally and, if enabled, remotely."""
        self.rows.append(dict(row))
        if self._mod is not None:
            try:
                self._mod.log(row)
            except Exception as exc:
                warnings.warn(f"remote log failed ({exc}); row kept locally")

    def finish(self):
        if self._mod is not None:
            try:
                self._mod.finish()
            except Exception as exc:
                warnings.warn(f"remote finish failed ({exc})")
