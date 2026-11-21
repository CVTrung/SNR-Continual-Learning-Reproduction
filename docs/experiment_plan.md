# Experiment plan (core scope)

## Goal

Reproduce the headline Permuted MNIST comparison (A) and the Permuted Shakespeare result (B), and test the paper's claim that SNR is robust to eta (C). See experiment_coverage.md.

## Fair comparisons

- Matched within each benchmark: same task sequence, same seed meaning, same architecture, optimizer settings and number of steps. The interventions differ only in the reset rule, the penalty, or the LayerNorm flag, all through one training loop (`src/train.py`).
- One seed sets the 10k subset, every task permutation, the initialization and the batch order. This follows the paper's definition of a seed.
- Hyperparameters: there is no budget for the paper's tuning sweeps, so each method uses the best value the paper reports (Appendix C, Tables 9-11). No team tuning took place, so every seed (100-102 for PM, 100-101 for PS) is a fresh evaluation seed. Seeds 0-2 were used by earlier exploratory runs and are not reused.
- Metrics: PM uses mean online accuracy over the last 10% of tasks, where online accuracy is the prediction made before each update, over the task's single epoch. PS uses the loss on the final epoch of each task, averaged over all tasks, the first 10% and the last 10% of tasks (the paper's first/last 50 of 500).

## Protocols

| | Paper | Reduced (local CPU, done) | Full (Colab notebook) |
| --- | --- | --- | --- |
| PM tasks / seeds | 2400 / 5 | 600 / 3 | 2400 / 5 |
| PM eta sweep | 2400 / 5 | 600 / 1 (seed 100) | 2400 / 1+ |
| PS tasks x epochs / seeds | 500 x 100 / 9 | 50 x 50 / 2 | 500 x 100 / 2+ |

## Measured resources (Windows 11, 16 logical CPU cores, no GPU)

- PM: 1.2 s per task for a single-threaded process running alone, and about 2.5-4 s per task with 10-16 processes running at once. A 600-task run takes 25-40 minutes.
- PS: 1.2-1.6 s per epoch single-threaded, so the paper protocol (50,000 epochs) would take about 18-22 hours per run on CPU.
- Single-threaded processes (`OMP_NUM_THREADS=1`) give the best throughput on CPU, and `scripts/sweep.py` sets that.

## Evidence

Each run writes `results/metrics/<run>.csv` (per task) and `<run>.json` (resolved config, commit, Python/torch versions, host, seconds per task, summary). `scripts/summarize_results.py` writes the tables and `scripts/plot_results.py` the figures. Remote tracking (`--tracking trackio|wandb`) is optional and adds a dashboard link. Never fabricate runs.
