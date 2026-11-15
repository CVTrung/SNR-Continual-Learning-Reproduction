"""Correctness checks for task generation (small synthetic data, fixed seeds)."""
import numpy as np

from src.tasks.permuted_mnist import make_permutation, make_task, select_subset


def _fake_data(n=20):
    # Every entry is unique, so any pixel shuffling is detectable.
    x = np.arange(n * 784, dtype=np.float32).reshape(n, 784)
    y = np.arange(n, dtype=np.int64) % 10
    return x, y


def test_same_seed_same_permutation():
    assert np.array_equal(make_permutation(3, base_seed=1), make_permutation(3, base_seed=1))


def test_different_tasks_differ():
    assert not np.array_equal(make_permutation(0), make_permutation(1))


def test_permutation_is_valid():
    perm = make_permutation(5)
    assert sorted(perm.tolist()) == list(range(784))


def test_labels_unchanged_and_pixels_only_rearranged():
    x, y = _fake_data()
    x_task, y_task, perm = make_task(x, y, task_id=2)
    assert np.array_equal(y_task, y)
    assert np.array_equal(x_task, x[:, perm])
    assert np.array_equal(np.sort(x_task, axis=1), np.sort(x, axis=1))


def test_same_permutation_for_every_image_in_a_task():
    x, y = _fake_data()
    x_task, _, perm = make_task(x, y, task_id=4)
    # Image 0 and image 7 must be shuffled in exactly the same way.
    assert np.array_equal(x_task[0] - x[0][perm], np.zeros(784))
    assert np.array_equal(x_task[7] - x[7][perm], np.zeros(784))


def test_subset_is_reproducible_and_right_size():
    x, y = _fake_data(50)
    xa, ya = select_subset(x, y, n=10, seed=0)
    xb, yb = select_subset(x, y, n=10, seed=0)
    assert xa.shape == (10, 784) and np.array_equal(xa, xb) and np.array_equal(ya, yb)