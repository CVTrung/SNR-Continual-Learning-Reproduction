# SNR Continual Learning Reproduction

Team project studying **Self-Normalized Resets for Plasticity in Continual Learning** (ICLR 2025).

- [Published paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf)
- [Author code: verification/comparison only](https://github.com/ajozefiak/SelfNormalizedResets)
- [Course project](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)

## Status

Documentation and design scaffold only. No experimental implementation, executable configurations, installed dependencies, measured results, or verified public W&B runs exist yet. Python files contain explanatory docstrings, YAML files contain comments, and notebooks contain markdown only. Running a script stub performs no experiment. requirements.txt does not specify an operational environment yet.

## Intended scope

| Dataset | Task | Model |
| --- | --- | --- |
| MNIST | Permuted MNIST; Random Label MNIST | MLP |
| CIFAR-10 | Random Label CIFAR | CNN |
| ImageNet-32 | Sequential binary class-pair tasks | CNN |
| Shakespeare | Permuted-vocabulary language modeling | Decoder Transformer |

Cover main comparisons, SGD/Adam where applicable, attention-only regularization ablations, activity/norm/entropy diagnostics, scaling, generalization, and sensitivity. Track supplementary scale-256 explicitly. Full-scope completion and resource feasibility are not established.

Plasticity means learning new tasks effectively after long training. Forgetting concerns old tasks; generalization concerns unseen examples. They are not interchangeable. Language experiments compare regularization/SNR combinations, not SNR alone.

## Navigation

- [Assignment requirements and submission checklist](docs/assignment_requirements.md).
- [Agent instructions](AGENTS.md), also linked from [Agent.md](Agent.md).
- [File-by-file purpose, inputs, and outputs](docs/repository_guide.md).
- [Paper figure/table coverage](docs/experiment_coverage.md).
- [Protocol and resource plan](docs/experiment_plan.md).
- [Reading outline](docs/paper_understanding.md), [decisions/deviations](docs/reproduction_notes.md), and [author verification](docs/author_comparison.md).
- [Critique](docs/critical_analysis.md), [empirical demonstration](docs/empirical_demonstration.md), and [contributions](docs/contributions.md).

## Architecture

```text
configs/       Task, model, method, optimizer, and experiment specifications
src/data/      Original dataset loading and preparation
src/tasks/     Sequential task construction and transformations
src/models/    MLP, CNN, and decoder Transformer architectures
src/methods/   SNR, comparison interventions, and reset operations
scripts/       Preparation, experiment, aggregation, and plotting entry points
notebooks/     Colab entry point and independent author-code verification
results/       Small metric exports, original tables, and figures
docs/          Paper understanding, protocol, coverage, critique, and contributions
tests/         Planned checks for task, reset, and metric correctness
```

src/data/ prepares base datasets; src/tasks/ creates task sequences; src/models/ defines architectures; src/methods/ defines interventions and architecture-specific resets. Training/evaluation use shared functions. Configurations combine task/model/method/optimizer settings into named experiments rather than duplicating source code. Notebooks remain thin entry points.

Earlier src/snr.py and configs/baseline.yaml / configs/snr.yaml are retained as legacy notes pointing to the new locations, not alternate implementations. Nothing was deleted to replace them.

## Reproducibility and integrity

Implement the core method from the paper description. Author method code may be read/run for verification, never copied into team source. Attribute standard reused blocks. Choose a supported framework and pin verified dependencies later; framework choice is unresolved.

Prepare data programmatically from verified Hugging Face/Kaggle sources. Record revisions, preprocessing, subsets, task provenance, complete config, seed roles, commit, environment/hardware, and metric timing. Separate tuning from fresh evaluation. Track completed runs in one public W&B project, with separate phase tags for reproduction and team critique. Keep datasets, checkpoints, secrets, and bulky logs outside Git.

## Experiment-to-evidence mapping

No completed runs exist. Fill rows only with actual commands/configs, completed public run links, and supported conclusions.

| Experiment / paper target | Script or notebook and resolved config | Completed public W&B runs | Result / conclusion |
| --- | --- | --- | --- |

Compare published and team values, report uncertainty/deviations, and explain differences. Run author-code comparisons separately. Additional theoretically and empirically supported weaknesses beyond the poster's selected two belong here with evidence.

## Before implementation

Resolve architecture, preprocessing, practical SNR estimation/resets, baseline variants, sweep grids, metric timing, data sources, framework, and measured compute budget. Working setup/run commands will be documented after implementation and environment verification. Reduced experiments must be labelled explicitly.

## Repository-link checkpoint

Team repository: [CVTrung/SNR-Continual-Learning-Reproduction](https://github.com/CVTrung/SNR-Continual-Learning-Reproduction).

One representative submits this single public repository URL through Moodle. The provided checkpoint wording allows continued work until the Final Poster deadline; it is not a code freeze. Exact dates must be checked on Moodle. This initial commit documents the intended work and does not claim the experimental-source-code requirement is already fulfilled. Nothing has been submitted to Moodle by creating or pushing this repository.
