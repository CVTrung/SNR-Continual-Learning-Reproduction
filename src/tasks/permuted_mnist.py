"""Construct the permuted MNIST task sequence.

Status: first executable version (NumPy only).
Inputs: base MNIST arrays x (N, 784) and y (N,), a task index, and a base seed.
Outputs: permuted images, unchanged labels, and the permutation that was used.
Responsibilities: apply ONE pixel permutation to all images of a task (training and
evaluation images alike), keep labels unchanged, and make the permutation reproducible
from (base_seed, task_id). Do not download data or define architectures.
Decision to record in docs/reproduction_notes.md: which 10,000 images form the subset
is not stated in the paper; here a fixed random subset is drawn once and shared by all tasks.
"""
import numpy as np


def select_subset(x, y, n=10_000, seed=0):
    """Pick the same n images for every task (drawn once, without replacement)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], y[idx]


def make_permutation(task_id, base_seed=0, n_pixels=784):
    """Return a reproducible permutation of pixel positions for one task."""
    rng = np.random.default_rng([base_seed, task_id])
    return rng.permutation(n_pixels)


def make_task(x, y, task_id, base_seed=0):
    """Build one task: permute the pixel columns of x, keep y.

    Returns (x_task, y, perm). x_task[i, j] = x[i, perm[j]].
    """
    perm = make_permutation(task_id, base_seed, n_pixels=x.shape[1])
    return x[:, perm], y, perm