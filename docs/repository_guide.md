# File-by-file repository guide

Status: core scope, executable (updated 2026-11-21). The full-scope stubs (CNN, Random-Label MNIST/CIFAR, Continual ImageNet, scaling and generalization configs, per-component config folders, legacy `src/snr.py`, `configs/baseline.yaml`, `configs/snr.yaml`) were removed when the team switched to core scope. Git history keeps them.

## Pipeline

`src/data/` loads the base datasets. `src/tasks/` builds the task sequence. `src/models/` defines the architectures. `src/methods/` holds the interventions. `src/train.py` is one shared loop for every method and both benchmarks. `scripts/` holds thin entry points. Results go to `results/`.

| File | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| [src/data/mnist.py](../src/data/mnist.py) | Read MNIST Parquet (or legacy .npz) as float32 (N, 784) in [0, 1]. | `src/data/mnist/*.parquet` | x, y arrays |
| [src/data/shakespeare.py](../src/data/shakespeare.py) | Extract The Tempest, tokenize with GPT-2 BPE, restrict the vocabulary, cut 256 windows of 129 tokens. | `src/data/shakespeare/all_shakespeare.txt` | sequences (256, 129), vocab ids |
| [src/tasks/permuted_mnist.py](../src/tasks/permuted_mnist.py) | Fixed 10k subset, one pixel permutation per task from (seed, task). | x, y, task id | permuted x, y |
| [src/tasks/permuted_shakespeare.py](../src/tasks/permuted_shakespeare.py) | One vocabulary permutation per task from (seed, task). | sequences, task id | permuted sequences |
| [src/models/mlp.py](../src/models/mlp.py) | 784-100-100-10 ReLU MLP, LeCun-normal init, optional LayerNorm; `reset_groups()`. | sizes | logits, hidden activations |
| [src/models/transformer.py](../src/models/transformer.py) | 1-layer decoder (2 heads x 16, 256 ReLU FFN); FFN activations, attention probabilities, attention/FFN parameter groups. | token ids | logits, FFN activations, attention |
| [src/methods/snr.py](../src/methods/snr.py) | SNR decision rule (step-level firing, geometric or histogram estimator). | activations each step | resets, statistics |
| [src/methods/resets.py](../src/methods/resets.py) | Shared reset mechanics: new incoming weights, bias 0, outgoing weights 0, Adam moments 0. | layers, unit indices, optimizer | modified parameters and state |
| [src/methods/cbp.py](../src/methods/cbp.py), [redo.py](../src/methods/redo.py), [shrink_perturb.py](../src/methods/shrink_perturb.py) | Comparison methods for Table 1. | activations or parameters | resets or perturbed parameters |
| [src/methods/regularization.py](../src/methods/regularization.py) | L2 / L2 Init penalties with masks `all` and `attention` (L2*). | model | loss term |
| [src/methods/regularizers.py](../src/methods/regularizers.py) | Legacy import path that re-exports regularization.py. | - | - |
| [src/optimizers.py](../src/optimizers.py) | SGD and Adam (eps 1e-7). | parameters, lr | optimizer |
| [src/train.py](../src/train.py) | Shared sequential training for classification and language-model tasks. | model, optimizer, task function, hooks | per-task accuracy and loss |
| [src/evaluate.py](../src/evaluate.py) | Batch accuracy; mean over the last 10% of tasks. | per-task values | summary |
| [src/diagnostics.py](../src/diagnostics.py) | Dead/inactive counters (examples or steps), attention entropy, Q/K/FFN norms. | activations, model | diagnostics |
| [src/checkpoint.py](../src/checkpoint.py) | Save/restore model, optimizer, method state, RNGs, task index, history. | run state | `checkpoints/*.pt` (ignored) |
| [src/tracking.py](../src/tracking.py) | Tracker with backend `wandb`, `trackio` or `none`; rows are also kept locally. | config, rows | remote run (optional) |
| [scripts/prepare_data.py](../scripts/prepare_data.py) | Download MNIST (HF `ylecun/mnist`, pinned revision) and the Shakespeare text (pinned GitHub raw), checked by SHA-256. | network | `src/data/<name>/` |
| [scripts/run_experiments.py](../scripts/run_experiments.py) | Run one experiment (PM or PS) with composable reset rule, penalty and LayerNorm. | CLI / flat YAML | `results/metrics/<run>.csv` + `.json` |
| [scripts/sweep.py](../scripts/sweep.py) | Run an experiment matrix in parallel; skips finished runs. | `configs/experiments/*.yaml` | many runs |
| [scripts/summarize_results.py](../scripts/summarize_results.py) | Aggregate runs into the deck/README tables, next to the paper's values. | `results/metrics/*.json/csv` | `results/tables/*.csv/.md` |
| [scripts/plot_results.py](../scripts/plot_results.py) | Team figures (accuracy, dead units, eta sweep, PS loss/inactivity/norms/entropy). | metrics, tables | `results/figures/*.png` |
| [configs/experiments/main.yaml](../configs/experiments/main.yaml) | Experiment A: PM Table 1, 8 conditions x SGD/Adam. | - | run matrix |
| [configs/experiments/shakespeare.yaml](../configs/experiments/shakespeare.yaml) | Experiment B: PS Table 2 conditions. | - | run matrix |
| [configs/experiments/sensitivity.yaml](../configs/experiments/sensitivity.yaml) | Experiment C: SNR eta sweep, geometric vs histogram estimator. | - | run matrix |
| [configs/experiments/critique.yaml](../configs/experiments/critique.yaml) | Placeholder for the poster's critique experiments (milestone 3). | - | - |
| [notebooks/colab.ipynb](../notebooks/colab.ipynb) | Full-protocol runs on a Colab GPU with Drive checkpoints. | repo branch | metrics on Drive |
| [notebooks/SNR_Permuted_MNIST.ipynb](../notebooks/SNR_Permuted_MNIST.ipynb) | User-supplied harness that runs the authors' JAX code; reference only, not team results. | author repo | inherited outputs |
| [notebooks/author_comparison.ipynb](../notebooks/author_comparison.ipynb) | Notes for author-code verification (see docs/author_comparison.md). | - | - |
| [tests/](../tests) | pytest checks for tasks, models, SNR/resets, baselines, penalties, diagnostics, training, checkpoint resume, tracking. | - | `pytest -q` |
| [deliverables/milestone2/build_deck.py](../deliverables/milestone2/build_deck.py) | Rebuilds the milestone-2 deck from docs/7.pptx plus results/tables and results/figures. | results | `Group7_SNR_Milestone2.pptx` |
| [results/metrics/preliminary_v1/](../results/metrics/preliminary_v1) | First-version teammate runs (3x100 MLP, per-example SNR), superseded. | - | CSVs |
| [docs/](.) | Assignment checklist, setup, data sources, plan, coverage, decisions, author comparison, reading notes. | - | - |

## Responsibility boundaries

Datasets live in the ignored `src/data/<name>/` folders, which is where the user moved them. Only loader modules are tracked. Models do not load data or build tasks. Methods decide which units to reset, and `resets.py` performs the resets. Penalties are loss terms, so they compose with any reset rule in the same loop.
