# SNR Continual Learning Reproduction

Group 7's reproduction of **Self-Normalized Resets for Plasticity in Continual Learning** (Farias & Jozefiak, ICLR 2025).

- [Published paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf)
- [Author code](https://github.com/ajozefiak/SelfNormalizedResets), read for verification only, never copied
- [Course project](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)

## Scope: core

The course allows core scope: the two most important experiments, including the headline comparison, plus one team-designed analysis. We chose it because no GPU was available locally (see [docs/experiment_coverage.md](docs/experiment_coverage.md)).

| Slot | What | Paper target |
| --- | --- | --- |
| A, headline | Permuted MNIST, MLP 784-100-100-10, 8 conditions (None, SNR, CBP, ReDO, L2, L2 Init, Shrink & Perturb, LayerNorm), SGD and Adam | Table 1 / 6, Figure 1 |
| B | Permuted Shakespeare, 1-layer decoder Transformer, Adam; None, L2, L2* (attention-only), L2 Init, SNR, SNR+L2, SNR+L2* | Table 2, Figures 2-4 |
| C, team analysis | SNR's sensitivity to eta (paper grid plus 1e-6), geometric vs histogram estimator | tests the robustness claim, Section 3.2 / Tables 10-11 |

The reduced-protocol results below were produced on a laptop CPU. The full paper protocol is prepared in [notebooks/colab.ipynb](notebooks/colab.ipynb).

## Quick start

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
python -m scripts.prepare_data                       # MNIST (HF ylecun/mnist, pinned) + Shakespeare text, sha256-checked
python -m pytest -q                                  # unit tests
python -m scripts.run_experiments --benchmark pm --reset snr --eta 0.005 --tasks 20
python -m scripts.sweep configs/experiments/main.yaml --workers 10
python -m scripts.summarize_results && python -m scripts.plot_results
```

More commands and the tracking backends (`--tracking wandb|trackio|none`) are in [docs/setup.md](docs/setup.md).

## Layout

```text
src/data/      loaders (datasets themselves sit in ignored src/data/<name>/ folders)
src/tasks/     permuted MNIST pixels, permuted Shakespeare vocabulary
src/models/    MLP (optional LayerNorm), decoder Transformer
src/methods/   snr.py (decision rule), resets.py (shared reset mechanics), cbp.py, redo.py,
               shrink_perturb.py, regularization.py (L2 / L2 Init with masks)
src/train.py   one training loop for every method and both benchmarks
scripts/       prepare_data, run_experiments, sweep, summarize_results, plot_results
configs/experiments/   main.yaml (A), shakespeare.yaml (B), sensitivity.yaml (C)
results/       metrics (per-run CSV + JSON), tables, figures
deliverables/milestone2/   deck build script and the milestone-2 PPTX (Group7_SNR_Milestone2.pptx)
```

## Experiment-to-evidence mapping

Public W&B project: [wandb.ai/vutrung2903-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/snr-reproduction](https://wandb.ai/vutrung2903-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/snr-reproduction). Run names match the files in `results/metrics/`. Tags give the phase: `eval`, `sweep`, or `preliminary-v1` for the superseded first version. Runs tagged `uploaded-from-csv` were logged from the saved per-task files after local training, while Colab runs log live. Run links: `results/wandb_runs.csv`.

| Experiment / paper target | Script and config | Runs (W&B) | Result (reduced protocol) and conclusion |
| --- | --- | --- | --- |
| A. Permuted MNIST, Table 1 (8 conditions, SGD/Adam) | `scripts/sweep.py configs/experiments/main.yaml` | runs `pm_*_tasks600` with tag `eval` | 600 of 2400 tasks; SGD 2-3 seeds, Adam 1 seed. SNR 0.831 (SGD; paper 0.851) and 0.874 (Adam; paper 0.889). SNR is level with CBP (0.833) and L2 Init (0.880 with Adam), and well above no intervention (0.763 / 0.746). Plasticity loss and dead units appear clearly. Shrink & Perturb collapses (0.663), probably because of our noise scale; see deviations. [table](results/tables/pm_main.md) |
| B. Permuted Shakespeare, Table 2 / Figs 2-4 | `scripts/sweep.py configs/experiments/shakespeare.yaml` | runs `ps_*_tasks30_ep50` | 30 of 500 tasks, 50 of 100 epochs, 1 seed. Loss on the last tasks: no intervention 2.80, SNR 2.85 (≈ no intervention, as in the paper), L2 1.74, SNR+L2 1.66 (lowest, as in the paper). L2* (attention only) did not help (2.79), unlike the paper. SNR keeps the FFN units alive, and only L2 stops the attention-entropy collapse. [table](results/tables/ps_main.md), [diagnostics](results/tables/ps_diagnostics.md) |
| C. η sensitivity (team analysis) | `scripts/sweep.py configs/experiments/sensitivity.yaml` | runs `pm_snr*_tasks300` with tag `sweep` | 300 tasks, 1 seed. Across the paper's grid SGD stays at 0.820-0.826 and Adam at 0.864-0.875 (best at η = 0.08, as in the paper), so the robustness claim reproduces with the step-level SNR. The first version (per-example firing) collapsed to 0.25 at η = 0.01. [table](results/tables/eta_sweep.md) |
| Full protocol (2400 tasks; 500 × 100 PS) | `notebooks/colab.ipynb` | PS runs `*_tasks500_ep100` (live) | running on Colab, not yet reported |


Every run has `results/metrics/<run>.csv` (per task) and `<run>.json` (resolved config, commit, environment, seconds per task). The deck cites the W&B project and the team's Trackio dashboard. `python -m scripts.sync_wandb --include-v1` uploads newly finished local runs.

## Integrity notes

- SNR and the baselines were reimplemented from the paper. The authors' code was read to verify details: the MLP shape, step-level firing, and the Shakespeare template. See [docs/author_comparison.md](docs/author_comparison.md).
- Every team decision and deviation is in [docs/reproduction_notes.md](docs/reproduction_notes.md). The first teammate version (a 3x100 MLP and per-example SNR) is kept in `results/metrics/preliminary_v1/` as history and is not reported as the result.
- Hyperparameters are the paper's reported best values. We did no tuning, and all reported seeds are evaluation seeds.

## Navigation

[Assignment checklist](docs/assignment_requirements.md) · [Setup](docs/setup.md) · [Data sources](docs/data_sources.md) · [Plan](docs/experiment_plan.md) · [Coverage](docs/experiment_coverage.md) · [Decisions](docs/reproduction_notes.md) · [Author comparison](docs/author_comparison.md) · [File guide](docs/repository_guide.md) · [Agent rules](AGENTS.md)

Team repository: [CVTrung/SNR-Continual-Learning-Reproduction](https://github.com/CVTrung/SNR-Continual-Learning-Reproduction).
