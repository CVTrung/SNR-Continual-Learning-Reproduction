# Experiment coverage

All rows are planned only. File existence does not demonstrate completed coverage.

Reference: [published ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf). References below use this version's numbering.

Vision conditions: no intervention, SNR, CBP, ReDO, L2, L2 Init, Shrink and Perturb, Layer Normalization.

| Target | Dataset / task / model | Method / optimizer | Config under configs/experiments/ | Protocol / metric / output |
| --- | --- | --- | --- | --- |
| Figure 1 | MNIST / PM / MLP | Baseline/SNR; SGD/Adam | illustration.yaml | Single-seed accuracy/inactivity curves; inactive after 1,000 consecutive examples |
| Tables 1, 6: PM | MNIST / PM / MLP | Vision conditions; SGD/Adam | main.yaml | 2,400 tasks; final-10%-task accuracy mean/std over 5 fresh evaluation seeds |
| Tables 1, 6: RM | MNIST / RM / MLP | Vision conditions; SGD/Adam | main.yaml | 100 tasks; same final-task summary |
| Tables 1, 6: RC | CIFAR-10 / RC / CNN | Vision conditions; SGD/Adam | main.yaml | 50 tasks; same final-task summary |
| Tables 1, 6: CI | ImageNet-32 / CI / CNN | Vision conditions; SGD/Adam | main.yaml | 500 binary tasks; same final-task summary |
| Figure 2 / Table 2 | Shakespeare / PS / decoder | Baseline, SNR, CBP, ReDO, L2, L2 Init, SNR+L2, attention-only L2 and SNR+attention-only L2; Adam | shakespeare.yaml | 500 tasks, 9 seeds; final-epoch loss, cumulative and first/last-50 summaries |
| Figures 3-4 | Shakespeare / PS / decoder | Figure-specific language conditions; Adam | shakespeare.yaml | Inactivity for 5 steps, Q/K/FFN norms, attention entropy; verify sampling/reductions |
| Table 3 | Shakespeare / PS / scaled decoder | L2/SNR+L2; Adam | scaling.yaml | Scales 1/16, 500 tasks, 4 seeds; loss and inactive rate |
| Section 3.2.2 / Figures 5-6 / Table 5 | MNIST/CIFAR / PM-G1, PM-G2, RC-G1 / MLP/CNN | Figure-specific methods; optimizer unresolved | generalization.yaml | Held-out accuracy/noise; RC-G1 16 versus 100 epochs; verify omitted protocol |
| Appendix C / Figures 7-10 / Tables 7-17 | Benchmark datasets/tasks and matching models | Method-specific sweeps; vision SGD/Adam, language Adam | sensitivity.yaml | Verify exact grids, tuning seeds and selection; sensitivity heatmaps/tables |
| Appendix A.1 / Table 4 | Shakespeare / PS / scale-256 decoder | L2/SNR+L2; Adam | scaling_256.yaml | 50 tasks, 100 epochs/task, 5 seeds; final-epoch training loss |
| Team critique | Selected for a named limitation | Controlled comparisons to justify | critique.yaml | Hypothesis-specific metrics and interpretation; no findings yet |

## Completion evidence

For each row record reference values, exact resolved config, entry-point command, completed public W&B runs, environment/commit/data revisions, aggregate uncertainty, deviation, and explanation. Mark reduced, blocked, or incomplete coverage explicitly. Theoretical claims belong in the reading/critique documents; a benchmark does not reproduce a theorem.
