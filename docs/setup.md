# Setup (verified 2026-11-21)

## Local (Windows, CPU)

Run these from the repository root. Each one was run successfully on Windows 11, with Python 3.12 managed by uv and 16 logical cores, no GPU.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m scripts.prepare_data          # MNIST from HF + Shakespeare text, sha256-checked
.venv/Scripts/python.exe -m pytest -q                     # all tests
.venv/Scripts/python.exe -m scripts.run_experiments --benchmark pm --reset snr --eta 0.005 --tasks 5
.venv/Scripts/python.exe -m scripts.run_experiments --benchmark ps --reset snr --penalty l2 --lam 1e-4 --eta 0.001 --tasks 1 --epochs 5
.venv/Scripts/python.exe -m scripts.sweep configs/experiments/main.yaml --workers 10      # reduced protocol A
.venv/Scripts/python.exe -m scripts.sweep configs/experiments/sensitivity.yaml --workers 10
.venv/Scripts/python.exe -m scripts.sweep configs/experiments/shakespeare.yaml --workers 6
.venv/Scripts/python.exe -m scripts.summarize_results
.venv/Scripts/python.exe -m scripts.plot_results
.venv/Scripts/python.exe deliverables/milestone2/build_deck.py   # needs python-pptx
```

`pytest.ini` points pytest's temporary directory to `.pytest_tmp/` inside the repository, because the Windows user temp folder was not writable in the sandbox. Long runs can resume with `--checkpoint-every N --resume`.

## Colab (GPU)

`notebooks/colab.ipynb` clones the branch, installs the packages Colab lacks, prepares the data, runs the tests, and then launches the full-protocol sweeps. Metrics and checkpoints go to Google Drive, so a disconnected session resumes where it stopped. It has not been executed yet.

## Tracking and secrets

- `--tracking trackio` sends rows to Trackio. Set `TRACKIO_SPACE_ID` (and `HF_TOKEN`) in the environment or in Colab Secrets to sync to a Hugging Face Space dashboard.
- `--tracking wandb` needs `wandb login` or `WANDB_API_KEY`, plus optionally `WANDB_ENTITY`. The course asks for one public W&B project.
- `--tracking none` (the default) keeps everything local.

In every mode the CSV/JSON files in `results/metrics/` are the local record. `.env.example` lists the variable names with blank values, and real values never enter Git.
