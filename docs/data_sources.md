# Dataset sources and provenance (verified 2026-11-21)

## Storage

The user moved the datasets from the root `data/` folder into `src/data/<name>/`. Those folders are ignored by Git (`/src/data/*/` in `.gitignore`). Only the loader modules `src/data/*.py` are tracked. `scripts/prepare_data.py` recreates the core-scope data from pinned sources and checks SHA-256. It keeps a correct existing file and never overwrites one with a different hash.

## Core-scope datasets

| Dataset | Programmatic source (pinned) | Local path | Verification | Preprocessing (team code) |
| --- | --- | --- | --- | --- |
| MNIST | Hugging Face `ylecun/mnist`, revision `77f3279092a1c1579b2250db8eafed0ad422088c` | `src/data/mnist/{train,test}-00000-of-00001.parquet` | sha256 of both local files equals the HF LFS hashes (train `f2c01285...`, test `d49fcf55...`); 60,000 / 10,000 rows; balanced labels | PNG decoded, divided by 255, flattened to 784 (`src/data/mnist.py`); fixed 10k subset per seed (`src/tasks/permuted_mnist.py`) |
| Shakespeare | Text file published with the paper's code: `raw.githubusercontent.com/ajozefiak/SelfNormalizedResets/8b84f0c.../src/data/all_shakespeare.txt` | `src/data/shakespeare/all_shakespeare.txt` | byte-identical to the authors' file after CRLF->LF; sha256 (LF) `4e81a0f2...9a23`; 5,342,509 bytes | The Tempest extracted (title to "THE END", 110,203 characters, 38,818 GPT-2 tokens); first 33,024 tokens = 256 windows of 129; 3,570-token restricted vocabulary (`src/data/shakespeare.py`) |

No Hugging Face or Kaggle copy of the authors' Shakespeare file was found to match it byte for byte. The course asks for programmatic loading, so the pinned GitHub raw URL is the reproducible programmatic source, and the deviation is recorded here. Only the text file is downloaded, never author code.

## Out-of-scope files still on disk

`src/data/cifar10/` (HF `uoft-cs/cifar10` Parquet: 50,000 / 10,000 rows) and `src/data/imagenet32/` (7 train shards of 183,024 rows, plus val with 50,000) are not used by core-scope code. They take about 3.1 GB and can be deleted locally if space is needed. The project rules require the user to make that call; no agent deletes datasets.
