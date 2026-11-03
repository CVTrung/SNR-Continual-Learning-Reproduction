# Project instructions for agents and contributors

## Purpose and authorization

This is a student reproduction of Self-Normalized Resets for Plasticity in Continual Learning (ICLR 2025). The target is the full experimental scope in docs/experiment_coverage.md plus course-required critique and empirical demonstrations.

Current work is structure and documentation only. Do not implement algorithms, run training, install dependencies, create remote resources, or publish results unless the user subsequently requests that work. Later explicit user instructions take precedence. Module/config notes describe intended work, not functioning features. No verified results, framework, hardware budget, hosted dataset choices, or W&B project exist yet.

## Read first

Read README.md, docs/assignment_requirements.md, docs/repository_guide.md, docs/experiment_coverage.md, docs/experiment_plan.md, and relevant per-file notes. Record verified decisions and uncertainties in docs/reproduction_notes.md. Prefer current owning sources over past suggestions.

Sources:
- Paper: https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf
- Author code, verification only: https://github.com/ajozefiak/SelfNormalizedResets
- Course: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/
- Reproduction: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction
- Critique: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis
- Empirics: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics

## Scientific scope

Cover PM/RM with an MLP, RC/CI with a CNN, and PS with a decoder Transformer. Include attention ablations, scale studies, generalization, and hyperparameter sensitivity. Track supplementary scale-256 explicitly. Verify protocols and metric timing before claiming fidelity.

Distinguish new-task plasticity, old-task forgetting, and held-out generalization. Preserve the language regularization combinations: SNR alone is insufficient there. The theoretical oracle-reset result is not a general practical-SNR guarantee.

The course asks for at least three strengths, two theoretical limitations, and two linked future directions. Critiques remain hypotheses until supported theoretically and empirically. Tag demonstration runs separately from reproduction.

## Implementation rules when implementation is requested

Reimplement the core method from the paper description. Author code may be read/run for verification and comparison, but never copy method code into team source or present it as team work. Do not vendor the author repository. Standard blocks, loaders, metrics, and tools may be reused with attribution.

Keep src/data/ for original datasets, src/tasks/ for task construction, src/models/ for architectures, and src/methods/ for interventions. Models do not download data or generate tasks. Prefer simple train/evaluate functions; avoid speculative base classes, registries, or infrastructure. Put parameter/seed variants in configs. Compose SNR and regularization without duplicating training loops. Layer Normalization is a model option.

Handle MLP neurons, CNN channels, and Transformer FFN units explicitly. Verify activity reduction axes, reset connectivity, biases, optimizer state, and method-statistic resets. Do not reset attention matrices as MLP neurons. Distinguish diagnostic inactivity from reset criteria. Keep notebooks thin. Important functions need docstrings stating purpose, parameters, and returns.

## Reproducibility

Prepare data programmatically from verified Hugging Face/Kaggle sources. Record revision, preprocessing, subsets, task-generation seeds, and provenance. Remote uploads/publication require appropriate user authorization.

Record resolved config, seed roles, commit, environment/hardware, and metric timing. Separate tuning from fresh evaluation seeds and keep comparisons matched. Use one public W&B project with phase tags. README maps experiment -> script/notebook/config -> completed public run -> conclusion. Never fabricate runs or results. Separate published references from team measurements and explain deviations/uncertainty/reduced protocols.

Checkpoint all required model, optimizer, method, initialization-reference, task-progress, and random states. Keep secrets, datasets, checkpoints, and bulk logs out of Git. Never embed credentials. Measure actual runtime/memory before full sweeps or paid-compute proposals. Do not launch the whole matrix by default.

## Files, checks, and communication

Work part by part. Explain the main change first, then why and supporting changes in simple language. Prefer simpler code when performance is equivalent. Do not modify unrelated/user-owned files without authorization. Ask before deleting project files. Retain earlier legacy scaffold paths until deletion is explicitly approved. Do not commit or push unless requested.

For documentation changes validate file inventory, local references, notebook JSON, and scaffold syntax without training or installing dependencies. For future authorized implementation use meaningful task/reset/metric/resume checks. Report exactly what was verified and what remains unresolved. File existence is not experimental completion.
