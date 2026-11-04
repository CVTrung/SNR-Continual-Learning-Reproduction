# Author-code comparison

Purpose: plan independent reference verification without copying the paper's implementation. Inputs: the authors' repository/version, isolated environment, and later agreed comparison inputs. Intended outputs: documented output differences and localized causes. Status: no clone, author environment, or reference execution has been verified in this update.

## Reference and location

Repository: https://github.com/ajozefiak/SelfNormalizedResets

Proposed location outside the team repository: D:/School/ML/references/SelfNormalizedResets. This is a proposed path, not confirmation that a clone exists. Do not vendor the author repository inside the team project. The exact reference commit must be inspected and recorded later; none is claimed here.

The team proposes PyTorch. Reference code may use JAX and its own dependencies; keep its environment separate. Do not assume either environment has been created or the authors' released code covers every published experiment.

## Later verification workflow

1. With authorization to run reference work, inspect the chosen author revision and available entry points; record actual commit, license, dependencies, and dataset requirements.
2. Prepare its separate environment and record successful commands only after testing them. Never copy author method code into src/ or submit it as team work.
3. Once a comparison protocol is agreed, align inputs, preprocessing, model settings, and seed roles; record unavoidable differences. Experiment configs and formal seed lists remain deferred.
4. Run reference and team pipelines separately. Record output artifacts, actual environment/hardware, command/config, and completed run links where applicable. No outputs or W&B links exist from this update.
5. Localize the first divergence: data/task sequence, initialization, forward results, activity statistics, reset decisions, optimizer behavior, or metric aggregation. Distinguish paper-versus-code discrepancies from bugs in the team implementation.

## Evidence record to fill

| Reference commit | Reference environment/command | Team commit/environment/command | Compared inputs | Outputs/run links | First divergence and explanation |
| --- | --- | --- | --- | --- | --- |

Standard blocks may be reused with documented attribution under the course rules; the core method must be implemented from the paper description. The reference notebook documents verification, not a second team training implementation.

## Included reference notebook

[notebooks/SNR_Permuted_MNIST.ipynb](../notebooks/SNR_Permuted_MNIST.ipynb) is a supplied harness calling the author JAX package. Static inspection found clone/install cells, reference parameters, and stored output logs. A provenance cell distinguishes it from the team PyTorch implementation. Original code cells and outputs are retained; no cells were run here and saved values are not verified team results. Exact export origin and author commit remain unknown. Run it only in an isolated reference workspace; its MNIST download path is not the team's required hosted-data pipeline.
