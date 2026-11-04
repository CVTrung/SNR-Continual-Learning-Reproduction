# Local and Colab environment plan

Purpose: describe the proposed team environment and the checks needed before using it. Inputs: the team code, current hardware, and verified compatibility information. Intended output: a reproducible local/Colab setup after validation. Status: documentation only; no environment creation, installation, login, or training performed in this update.

## Proposed workflow

Use PyTorch for the team implementation. Develop and debug in a local .venv under the repository root; use a Google Colab GPU for longer experiments when resources are measured and available. Use the same team modules rather than maintaining independent notebook implementations.

Python, PyTorch, torchvision (if needed), CUDA compatibility, and tokenizer versions have not been selected or tested. GPU availability and runtime limits on Colab are not guaranteed. requirements.txt currently describes anticipated dependencies in comments; installing it does not prepare a working environment.

## Local preparation, to perform later

1. Inspect the actual local Python installation and hardware. Select a Python/PyTorch combination supported by the intended runtime before creating .venv.
2. Create the ignored .venv, install the verified dependency set, and record versions and hardware. Do not borrow author-code dependencies as the team environment.
3. Use the existing data/ files read-only while their provenance and schema are checked. src/data/ contains loaders, never raw datasets. See data_sources.md.
4. Once implementation is authorized, verify imports, a small data batch, model behavior, and checkpoint/resume before attempting longer runs. None of those checks has been performed here.
5. Record confirmed setup/run commands only after executing them successfully; no operational installation/training commands are claimed in this document.

## Colab preparation, to perform later

Use notebooks/colab.ipynb as a thin entry point. Select the same repository commit and compatible pinned dependencies as local development. Record any unavoidable platform differences rather than claiming identical environments. Prepare datasets programmatically from verified Hugging Face/Kaggle sources; the local data/ directory will not travel through Git.

Record Python, PyTorch, CUDA/runtime, GPU model, repository commit, dependency versions, and dataset revision. Check persistent checkpoint storage and resume behavior before relying on an ephemeral runtime. No Colab session or GPU has been provisioned or tested.

## W&B and secrets

.env.example contains blank WANDB_ENTITY, WANDB_PROJECT, and WANDB_API_KEY entries. Values belong only in the ignored local .env or the runtime's secret facility. Merely adding .env does not load it into Python: a loading mechanism must be explicitly chosen when implementation begins. Do not display or commit API keys.

One public W&B project is required for experiment evidence. Its account/entity, project, visibility, and authentication are not configured or verified in this scaffold. Do not create a project, log in, or publish a run on someone's behalf without authorization. Separate reference-code verification from team runs through clear provenance and later phase tags.

## Deferred decisions

Executable experiment YAML, official tuning/evaluation seed lists, and result-file conventions are intentionally deferred. Seed 42 is only a possible debug seed, not a mandatory protocol. Formal comparisons need multiple seeds and separate tuning/evaluation roles. See experiment_plan.md; this setup plan does not finalize them.
