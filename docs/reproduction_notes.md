# Reproduction notes

Status: no implementation or runs yet.

For each decision record its paper section, unresolved question, evidence inspected, chosen interpretation, alternatives, affected experiments, and expected consequences.

Track architecture, preprocessing, initialization, task generation, activity estimator/window, reset ordering, optimizer-state handling, regularization masks, metric timing, sweep strategy, and dependency versions. Record reused standard components and attribution.

Separate verified facts from assumptions. Label reduced tasks, seeds, or tuning budgets explicitly. Explain meaningful deviations instead of claiming an exact reproduction.

- Preprocessing: MNIST pixels divided by 255, flattened to 784; no further standardization (paper does not state it). Data read from a local mnist.npz exported from HF ylecun/mnist. Source: src/data/mnist.py.
- Task subset: a fixed random 10,000-image subset (seed 0), shared by all tasks; the paper does not say which images. Permutation of task t comes from (base_seed, t). Source: src/tasks/permuted_mnist.py.