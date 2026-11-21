# Experiment coverage (core scope)

The course allows **core scope**: at least the two most important experiments, including the headline comparison, plus one team-designed analysis in place of the omitted experiments. The team switched from full scope to core scope on 2026-11-21. Compute was the reason: no GPU was available locally, and the full matrix of five benchmarks is far beyond the time budget.

Reference: [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf).

| Slot | Paper target | Data / model | Conditions | Config | Protocol run so far | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A (headline) | Table 1 / 6, PM column; Figure 1 | MNIST, 784-100-100-10 MLP | None, SNR, CBP, ReDO, L2, L2 Init, S&P, LayerNorm; SGD and Adam | [main.yaml](../configs/experiments/main.yaml) | Reduced: 600 of 2400 tasks; SGD 2-3 seeds, Adam 1 seed; paper-best hyperparameters | see README mapping |
| B | Table 2, Figures 2-4 | Tempest, 1-layer decoder Transformer | None, L2, L2*, L2 Init, SNR, SNR+L2, SNR+L2*; Adam | [shakespeare.yaml](../configs/experiments/shakespeare.yaml) | Reduced: 30 of 500 tasks, 50 of 100 epochs, 1 of 9 seeds | see README mapping |
| C (team analysis) | Tests the robustness claim of Section 3.2 / Tables 10-11 | PM, MLP | SNR, eta in {0.08 ... 0.00125} and 1e-6; geometric vs histogram estimator; SGD and Adam | [sensitivity.yaml](../configs/experiments/sensitivity.yaml) | Reduced: 300 tasks, 1 seed | see README mapping |
| Full protocol | Same as A-C | same | same | [notebooks/colab.ipynb](../notebooks/colab.ipynb) | 2400 tasks / 5 seeds; 500 x 100 / 9 seeds | PS full runs started on Colab (logging live to W&B); PM not started |

Out of scope: Random-Label MNIST, Random-Label CIFAR and Continual ImageNet (Table 1, other columns); CBP and ReDO on PS (the paper gives no PS grid for them); the scaling studies (Table 3, Appendix A.1); generalization (Section 3.2.2); the remaining Appendix C sweeps. The CIFAR-10 and ImageNet-32 files are still in `src/data/` but no code uses them.

## Completion evidence

The README "Experiment-to-evidence mapping" lists, for each slot, the config, the command, the result files and the conclusion. Each run writes a CSV and a JSON with the resolved config, commit, environment and seconds per task. Reduced protocols are labelled everywhere they appear.
