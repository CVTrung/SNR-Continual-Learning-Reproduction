# Project instructions for agents and contributors

## Purpose and authorization

This is a student reproduction of Self-Normalized Resets for Plasticity in Continual Learning (ICLR 2025). The target is the **core scope** in docs/experiment_coverage.md (user decision, 2026-11-21): A = Permuted MNIST Table 1, B = Permuted Shakespeare Table 2 / Figures 2-4, C = team eta-sensitivity analysis. It also covers the course-required critique and empirical demonstrations.

The user has authorized implementation, the local .venv, dependency installation and reduced-protocol training for core scope (2026-11-21). Still ask before creating remote resources (W&B/Trackio projects, HF Spaces), publishing, pushing, or deleting datasets. Later explicit user instructions take precedence. The verified environment and data provenance are in docs/setup.md and docs/data_sources.md; the public W&B project is not yet created.

## Read first

Read README.md, docs/setup.md, docs/data_sources.md, docs/assignment_requirements.md, docs/repository_guide.md, docs/experiment_coverage.md, docs/experiment_plan.md, and relevant per-file notes. Record verified decisions and uncertainties in docs/reproduction_notes.md. Prefer current owning sources over past suggestions.

Sources:
- Paper: https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf
- Author code, verification only: https://github.com/ajozefiak/SelfNormalizedResets
- Course: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/
- Reproduction: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction
- Critique: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis
- Empirics: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics

## Scientific scope

Core scope: PM with an MLP (8 conditions, SGD/Adam), PS with a decoder Transformer (Table 2 conditions, including the attention-only L2 ablation), and the SNR eta sensitivity analysis. RM, RC, CI, scaling, generalization and scale-256 are out of scope. Verify protocols and metric timing before claiming fidelity.

Distinguish new-task plasticity, old-task forgetting, and held-out generalization. Preserve the language regularization combinations: SNR alone is insufficient there. The theoretical oracle-reset result is not a general practical-SNR guarantee.

The course asks for at least three strengths, two theoretical limitations, and two linked future directions. Critiques remain hypotheses until supported theoretically and empirically. Tag demonstration runs separately from reproduction.

## Implementation rules when implementation is requested

Reimplement the core method from the paper description. Author code may be read/run for verification and comparison, but never copy method code into team source or present it as team work. Do not vendor the author repository. Standard blocks, loaders, metrics, and tools may be reused with attribution.

Raw datasets live in the ignored src/data/<name>/ folders (the user moved them there); only the loader modules src/data/*.py are tracked. Do not move, delete, overwrite, or re-download existing datasets without explicit authorization. Keep src/tasks/ for task construction, src/models/ for architectures, and src/methods/ for interventions. Models do not download data or generate tasks. Prefer simple train/evaluate functions; avoid speculative base classes, registries, or infrastructure. Put parameter/seed variants in configs. Compose SNR and regularization without duplicating training loops. Layer Normalization is a model option.

Handle MLP neurons, CNN channels, and Transformer FFN units explicitly. Verify activity reduction axes, reset connectivity, biases, optimizer state, and method-statistic resets. Do not reset attention matrices as MLP neurons. Distinguish diagnostic inactivity from reset criteria. Keep notebooks thin. Important functions need docstrings stating purpose, parameters, and returns.

## Environment and deferred work

Use the PyTorch team code, the local .venv (Python 3.12, pins in requirements.txt) for development and reduced runs, and notebooks/colab.ipynb on a Colab GPU for full-protocol runs. Do not invent version pins or claim commands worked without testing them. W&B/Trackio login or project creation and remote publication still need the user. Reference code stays outside the team repo at D:/School/ML/references/SelfNormalizedResets (read for verification only).

Executable experiment matrices are configs/experiments/{main,shakespeare,sensitivity}.yaml, run with scripts/sweep.py. Evaluation seeds are 100-104; seeds 0-2 were used by exploratory runs. Hyperparameters come from the paper's reported best values (no team tuning), as documented in docs/experiment_plan.md. The included notebooks/SNR_Permuted_MNIST.ipynb is an attributed user-supplied harness invoking author JAX code, with unverified inherited outputs. Its original cells/outputs are preserved; exact export origin/reference revision and numerical results are not validated. Do not present it as team implementation/results or execute its relative clone inside the team repository. Do not fabricate team members or assignments.

.env.example may contain only blank W&B variable values and explanatory comments. Real credentials stay in ignored .env or runtime secrets. W&B account, public project, and authentication are not assumed to exist. A .env file requires an explicitly chosen loading mechanism.

## Reproducibility

Prepare data programmatically from verified Hugging Face/Kaggle sources. Record revision, preprocessing, subsets, task-generation seeds, and provenance. Remote uploads/publication require appropriate user authorization.

Record resolved config, seed roles, commit, environment/hardware, and metric timing. Separate tuning from fresh evaluation seeds and keep comparisons matched. Use one public W&B project with phase tags. README maps experiment -> script/notebook/config -> completed public run -> conclusion. Never fabricate runs or results. Separate published references from team measurements and explain deviations/uncertainty/reduced protocols.

Checkpoint all required model, optimizer, method, initialization-reference, task-progress, and random states. Keep secrets, datasets, checkpoints, and bulk logs out of Git. Never embed credentials. Measure actual runtime/memory before full sweeps or paid-compute proposals. Do not launch the whole matrix by default.

## Files, checks, and communication

Work part by part. Explain the main change first, then why and supporting changes in simple language. Prefer simpler code when performance is equivalent. Do not modify unrelated/user-owned files without authorization. Ask before deleting project files. Retain earlier legacy scaffold paths until deletion is explicitly approved. Do not commit or push unless requested.

For documentation changes validate file inventory, local references, notebook JSON, and scaffold syntax without training or installing dependencies. For future authorized implementation use meaningful task/reset/metric/resume checks. Report exactly what was verified and what remains unresolved. File existence is not experimental completion.
