# Author-code comparison

Reference: https://github.com/ajozefiak/SelfNormalizedResets, commit `8b84f0c295cf23269fdbfe20e5d03186638c318f` (2025-08-06). It was cloned outside the team repository to `D:/School/ML/references/SelfNormalizedResets`, read only, and never executed or copied. The team code is a PyTorch reimplementation from the paper.

## What was compared (static reading, 2026-11-21)

| Item | Author code | Team code | Effect |
| --- | --- | --- | --- |
| PM MLP | 2 x Dense(100), ReLU, Flax default init | was 3 x 100 with PyTorch default init; now 2 x 100 with LeCun-normal init and zero bias | first-version runs moved to `results/metrics/preliminary_v1/` |
| SNR firing unit | per mini-batch (any example active) | was per example; now per step | fixed the over-reset and the collapse at large eta |
| SNR threshold | per-unit (1 - eta) percentile of a histogram of silent runs (age-0 excluded), refreshed every 16 tasks, x2 expansion, floor of one batch, start at 20,000 examples | geometric (mean of silent runs, EMA) or histogram (1-task window), no expansion rule | different estimator details; experiment C measures how much that matters |
| SNR eta in the provided notebook | `threshold_percentile = 0.92`, i.e. eta = 0.08 | eta from the paper's grid | consistent with the paper's best Adam value |
| PS corpus | first N tokens of `all_shakespeare.txt` (All's Well That Ends Well) | The Tempest, as the paper states | our text follows the paper and differs from the public template |
| PS permutation | whitespace-word permutation, then BPE | permutation of restricted-vocabulary token ids | ours matches the paper's wording ("permutation of the vocabulary") |
| PS context | windows of 129 tokens (128 + 1) | same | identical |
| Shakespeare file | `src/data/all_shakespeare.txt` | byte-identical after CRLF->LF | same data |

## Not compared

No author run was executed: the authors' JAX environment was not built, so no numerical outputs were compared side by side. The provided notebook `notebooks/SNR_Permuted_MNIST.ipynb` keeps its inherited outputs, which are unverified and are not team results. The author repository has no PS model code, and its template leaves the model step empty.
