"""Checks for the Tempest token pipeline and the vocabulary-permutation tasks."""
from pathlib import Path

import numpy as np
import pytest

from src.data.shakespeare import extract_tempest, remap_vocabulary, make_sequences
from src.tasks.permuted_shakespeare import make_task, make_vocab_permutation

CORPUS = Path("src/data/shakespeare/all_shakespeare.txt")


def test_extract_tempest_from_synthetic_text():
    text = "junk\nTHE TEMPEST\n\nby William Shakespeare\nACT I\nhello\nTHE END\nmore"
    play = extract_tempest(text)
    assert play.startswith("THE TEMPEST") and play.rstrip().endswith("THE END")
    assert "junk" not in play and "more" not in play


def test_remap_vocabulary_is_dense_and_invertible():
    tokens = np.array([50, 7, 50, 900, 7])
    ids, vocab = remap_vocabulary(tokens)
    assert ids.tolist() == [1, 0, 1, 2, 0]
    assert vocab[ids].tolist() == tokens.tolist()


def test_make_sequences_shape():
    seqs = make_sequences(np.arange(1000), context=128, n_sequences=7)
    assert seqs.shape == (7, 129)
    assert seqs[1, 0] == 129


def test_permutation_reproducible_and_valid():
    a = make_vocab_permutation(3, 40, base_seed=1)
    assert np.array_equal(a, make_vocab_permutation(3, 40, base_seed=1))
    assert not np.array_equal(a, make_vocab_permutation(4, 40, base_seed=1))
    assert sorted(a.tolist()) == list(range(40))


def test_task_applies_one_permutation_consistently():
    seqs = np.array([[0, 1, 2, 1], [2, 2, 0, 1]])
    x_task, perm = make_task(seqs, task_id=0, vocab_size=3, base_seed=0)
    assert np.array_equal(x_task, perm[seqs])


@pytest.mark.skipif(not CORPUS.exists(), reason="local corpus not prepared")
def test_real_tempest_token_budget():
    from src.data.shakespeare import load_tempest
    seqs, vocab = load_tempest(CORPUS)
    assert seqs.shape == (256, 129)        # 32 batches x 8 sequences, 128 targets each
    assert seqs.shape[0] * (seqs.shape[1] - 1) == 32_768   # predicted tokens per epoch
    assert seqs.max() < len(vocab)
