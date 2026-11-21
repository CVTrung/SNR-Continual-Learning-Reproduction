"""Construct the Permuted Shakespeare task sequence.

Status: executable (NumPy only).
Each task applies one random permutation of the restricted vocabulary to the whole Tempest
token sequence, so the text structure is kept and the token identities change. The
permutation is reproducible from (base_seed, task_id), using the same pattern as Permuted
MNIST.
Team decision: we permute token ids of the restricted GPT-2 vocabulary, which is the paper's
"random permutation of the vocabulary". The authors' template permutes whitespace-separated
words before tokenizing, and we note that as a difference.
"""
import numpy as np


def make_vocab_permutation(task_id, vocab_size, base_seed=0):
    """Return a reproducible permutation of the vocabulary for one task."""
    rng = np.random.default_rng([base_seed, task_id])
    return rng.permutation(vocab_size)


def make_task(sequences, task_id, vocab_size, base_seed=0):
    """Apply the task's vocabulary permutation. Returns (permuted sequences, perm)."""
    perm = make_vocab_permutation(task_id, vocab_size, base_seed)
    return perm[sequences], perm
