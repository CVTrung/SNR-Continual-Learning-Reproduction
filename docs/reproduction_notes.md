# Reproduction notes: decisions, evidence, deviations

Status: core-scope implementation complete (2026-11-21). Reduced-protocol runs were made on local CPU, and the full protocol is prepared for Colab. Below, *paper* means the ICLR 2025 PDF. *Author code* means github.com/ajozefiak/SelfNormalizedResets at commit 8b84f0c, which was read for verification only. None of its code is in this repository.

## Method: SNR (src/methods/snr.py)

| Question | Paper says | Author code does (verification) | Team decision | Why / consequence |
| --- | --- | --- | --- | --- |
| Time unit | Algorithm 1 loops "for each training example" | a unit "fires" on a mini-batch if active on any example, and its age grows by the batch size while silent | **one optimizer step**: fired = active on any example or token of the batch | Our first version counted examples inside a batch, so dense neurons got p close to 1 and were reset after 1-2 silent examples. That caused the collapse at large eta in our first sweep (0.25 at eta 0.01). Step-level counting fixed it: the 100-task check gave SGD 0.803-0.805 for eta 0.08-0.00125. |
| What is an "inter-firing time" | histogram of past inter-firing times; footnote 3: track the mean and assume a geometric distribution | its histogram skips age-0 entries | only **completed silent runs of length >= 1 step** are counted | Back-to-back firings are not silences. Under a geometric law on {1, 2, ...} with mean mu, P(R >= a) = (1 - 1/mu)^(a-1). |
| Estimator | main text: histogram; footnote 3: geometric mean over a trailing window | per-unit histogram percentile, refreshed every 16 tasks, expanded by x2, never below 1 batch | `estimator="geometric"` (default) with an EMA (beta 0.1) in place of the window; `estimator="histogram"` with a 1-task window | The EMA keeps one number per unit. Experiment C compares both estimators. |
| Start-up | not stated | initial threshold 20,000 examples | before the first completed run mu = m_init = 1000 steps | A unit that is dead from initialization is reset after about mu ln(1/eta) steps. |
| Reset operation | re-initialize the neuron | incoming weights redrawn from the init, bias 0, outgoing weights 0; Adam state handled | same, plus zeroing the Adam moments of exactly those entries (`resets.py`) | The network function is unchanged at the moment of the reset. |
| Statistics after a reset | not stated | age reset, histogram kept | age, mean and histogram cleared (a fresh unit) | The new unit's firing pattern is unrelated to the old one. |

## Architectures

- **MLP:** 784-100-100-10, ReLU. The paper says it is identical to Kumar et al. (2023b), and the author code has two Dense(100) layers. Our first version used 3 x 100, so those runs are in `results/metrics/preliminary_v1/` and are not compared with the paper. Init: LeCun-normal (truncated at 2 std), zero bias, the Flax default used by the authors. LayerNorm, when enabled, sits after each hidden linear layer and before its ReLU.
- **Transformer:** 1 decoder layer, 2 heads x 16 (width 32), 256-unit ReLU FFN, context 128. Pre-LayerNorm GPT-style block, learned positional embeddings, untied head. The paper does not give these details. SNR, CBP and ReDO act only on the FFN units, never on attention.

## Data and tasks

- **PM:** pixels / 255, a fixed random 10k subset per seed, one pixel permutation per task from (seed, task). Task 0 is permuted too. Batch 16, 1 epoch, so 625 steps per task.
- **PS:** The Tempest, as the paper states. The authors' public template reads the first tokens of the complete-works file instead, which is *All's Well That Ends Well*. We use 256 windows of 129 tokens, so each epoch is 32 batches of 8 x 128 predictions, matching the paper's 32,768 tokens. The vocabulary is restricted to the 3,570 GPT-2 ids present. Each task permutes those token ids; the authors' template permutes whitespace words before tokenizing.

## Baselines (implemented from the published descriptions)

- **CBP:** contribution utility mean|h| * sum|w_out| with an EMA (decay 0.99), maturity 100 steps, and replacement credit r x 16 examples per step per layer (the paper: "one replacement every 1/r examples").
- **ReDO:** score = mean|h| / layer mean over the period, reset if score <= tau, checked every r tasks. The paper measures the frequency in tasks.
- **S&P:** after every step, theta <- (1 - shrink) theta + sigma N(0, 1), biases included.
- **L2 / L2 Init:** explicit loss terms 0.5 lambda ||theta (- theta0)||^2 over all parameters, biases included (team decision). The L2* mask covers the Q/K/V/output projections only.

## Protocol deviations (labelled wherever results appear)

- **Reduced protocol** because only a CPU was available. PM: 600 of 2400 tasks, 3 of 5 seeds. PS: 50 of 500 tasks, 50 of 100 epochs, 2 of 9 seeds. Eta sweep: 600 tasks, 1 seed.
- **No team tuning.** Each method uses the paper's reported best hyperparameter (Tables 9-11), so the paper's tuning protocol is not reproduced. The comparison is conditional on those settings transferring to our implementation and to the shorter runs.
- **Metric window.** "Last 10% of tasks" is the last 60 tasks at 600 tasks and the last 240 at 2400, so reduced runs are not directly comparable with Table 1 numbers. The no-intervention baseline declines over time, so the gap would widen at 2400 tasks.
- **Tracking.** Runs were logged locally (CSV + JSON with config, commit and environment). The deck cites the team's Trackio dashboard. The course's public W&B project still has to be created by the team, using `--tracking wandb`.

## Reused standard components

PyTorch modules (nn.Linear, nn.LayerNorm, nn.Embedding), tiktoken's GPT-2 encoding, huggingface_hub for downloads, and matplotlib. No method code was copied from the authors.
