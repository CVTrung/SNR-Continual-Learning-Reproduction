# Metrics

One CSV (one row per task) and one JSON (resolved config, commit, Python/torch versions, host, seconds per task, last-10% summary, tracking URL) per run, written by `scripts/run_experiments.py`.

File name: `<benchmark>_<condition>_<optimizer>_seed<s>_tasks<N>[_ep<E>].csv`, e.g. `pm_snr_eta0.005_sgd_seed100_tasks600.csv`, `ps_snr_eta0.001+l2star_lam0.0001_adam_seed100_tasks50_ep50.csv` (`l2star` = attention-only L2).

- PM columns: task, resets, online_accuracy, loss, dead_fraction (share of neurons silent for >= 1000 examples).
- PS columns: task, resets, final_loss, inactive_units (FFN units silent for >= 5 steps), q_norm, k_norm, ffn_norm, attn_entropy.

Seeds 100-102 (PM) and 100-101 (PS) are the reduced-protocol evaluation runs of 2026-11-21 (local CPU). `preliminary_v1/` holds the superseded first-version runs.
