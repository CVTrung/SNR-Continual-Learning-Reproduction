# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

AGENTS.md (imported above) is the canonical rule set: authorization limits, scientific scope, implementation rules, and reproducibility requirements. Agent.md only points to it. This file adds orientation that is not stated there; if the two ever conflict, AGENTS.md and later explicit user instructions win.

## Current state (2026-11-21, branch `milestone2-core-scope`)

Core scope is implemented and runnable: A = Permuted MNIST Table 1 (8 conditions, SGD/Adam), B = Permuted Shakespeare Table 2 (Transformer), C = SNR eta-sensitivity analysis. Reduced-protocol results come from local CPU runs. The full protocol lives in `notebooks/colab.ipynb` and has not been run yet.

- SNR (`src/methods/snr.py`) counts firing per **optimizer step** (any example/token active) and uses only completed silent runs, with a `geometric` (default) or `histogram` estimator. The old per-example version over-reset and collapsed at large eta. Its runs are in `results/metrics/preliminary_v1/` and must not be reported as results.
- MLP is 784-100-100-10 with LeCun-normal init and zero bias, matching the authors' Flax MLP. Reset mechanics are shared in `src/methods/resets.py`; models expose `reset_groups()`.
- Datasets live in ignored `src/data/<name>/` folders (the user moved them there). `scripts/prepare_data.py` re-downloads and sha256-verifies them.
- `deliverables/milestone2/build_deck.py` rebuilds the English deck from `docs/7.pptx` plus `results/tables/*.csv` and `results/figures/*.png`. Re-run it after new results.
- Tracking: `--tracking wandb|trackio|none`. The deck cites a Trackio dashboard. The course's public W&B project has not been created yet; that is the user's step.

## Commands

Use `.venv/Scripts/python.exe` (Python 3.12; pins in requirements.txt). Run from the repo root, because imports are `src.*` / `scripts.*`.

```bash
python -m pytest -q                                   # ~1 min; pytest.ini sets --basetemp=.pytest_tmp
python -m pytest tests/test_resets.py::test_always_firing_neuron_survives_one_silent_step
python -m scripts.run_experiments --benchmark pm --reset snr --eta 0.005 --tasks 5
python -m scripts.run_experiments --benchmark ps --reset snr --penalty l2 --lam 1e-4 --eta 0.001 --tasks 1 --epochs 5
python -m scripts.sweep configs/experiments/main.yaml --workers 8 [--dry-run]   # skips finished/running runs
python -m scripts.summarize_results && python -m scripts.plot_results
python deliverables/milestone2/build_deck.py
```

On this laptop (i5-1240P, 16 GB RAM), single-threaded processes are fastest. `sweep.py` sets `OMP_NUM_THREADS=1`. More than about 10 concurrent runs mostly adds contention: a PM task takes 1.2 s alone and about 3.5 s under load.

## Validating documentation and scaffold changes

AGENTS.md asks for checks of file inventory, local references, notebook JSON, and scaffold syntax. This command needs only the system Python (3.13 was available on this machine) and installs nothing:

```bash
python -B -c "import ast,glob,json; [ast.parse(open(f,encoding='utf-8').read(),f) for f in glob.glob('src/**/*.py',recursive=True)+glob.glob('scripts/*.py')+glob.glob('tests/*.py')]; [json.load(open(f,encoding='utf-8')) for f in glob.glob('notebooks/*.ipynb')]; print('ok')"
```

Use `git ls-files` to check the inventory. Check relative Markdown links against the tree by hand.

## Architecture

The pipeline is split into layers, and each layer has one job:

`src/data/<name>/` (raw, ignored) -> `src/data/*.py` (loaders) -> `src/tasks/` (pixel permutations, Shakespeare vocabulary permutation) -> `src/models/` (MLP with optional LayerNorm, decoder Transformer) -> `src/methods/` (interventions) -> `src/train.py` (one loop for classification and LM tasks) -> `src/tracking.py`, `src/checkpoint.py`, `src/diagnostics.py`.

- Decision rules and reset mechanics are kept apart. `snr.py`, `redo.py` and `cbp.py` decide *which* units to reset. `resets.reinit_units` *performs* the reset: new incoming weights, bias 0, outgoing weights 0, Adam moments 0. Models expose `reset_groups()`, which is (incoming, outgoing) Linear pairs: MLP hidden layers, or only the Transformer FFN. Attention is never reset. Methods expose `reset_listeners`, so diagnostics can clear counters for reset units.
- Penalties (`regularization.py`, masks `all` / `attention`) are loss terms passed as `penalty=`. Reset rules are `on_step=` hooks plus `end_task()`. They compose in the same loop (for example SNR+L2*).
- Experiments are matrices in `configs/experiments/{main,shakespeare,sensitivity}.yaml` (common options, seeds, runs). Keys are `run_experiments` long options. `critique.yaml` is a placeholder for the poster.
- `scripts/` are thin entry points (prepare data, run a selected experiment, summarize, plot). `scripts/run_experiments.py` must never default to running the full matrix.
- `results/` holds only small exports (metrics, tables, figures). Checkpoints, logs, and `wandb/` are ignored.

## Where decisions live

- docs/reproduction_notes.md: verified decisions, deviations, and open uncertainties. Update it when you resolve something.
- docs/experiment_plan.md: protocol, seeds, and resource plan. docs/experiment_coverage.md: coverage of the paper's figures and tables.
- docs/repository_guide.md: purpose, inputs, and outputs for each file. Keep it in sync when you add or rename files.
- docs/assignment_requirements.md: the course deliverables (reproduction, critique, empirical demonstration).
- README.md "Experiment-to-evidence mapping" table: fill it only with real configs, completed public W&B runs, and supported conclusions.
