# Experiment plan

Status: full-scope target; documentation scaffold only. No experimental implementation or runs. Feasibility is unmeasured.

## Goal

Cover all main published experiments: four vision benchmarks, the language benchmark/attention ablations, diagnostics, scale studies, generalization, and sensitivity. Track supplementary scale-256 explicitly. See experiment_coverage.md for artifact-level mapping.

Development can proceed part by part (PM/MLP, remaining vision/CNN, language/Transformer, extended analyses) without redefining the full-scope target. Smoke runs establish functionality, not complete reproduction.

## Fair comparisons

Match task sequences, architectures, training budgets, and seed roles across comparable methods. Separate tuning from fresh evaluation seeds. Explicitly define optimizer options, intervention order, metric timing, selection criterion, and aggregation. Retain raw per-task/per-seed records and summarize only compatible completed runs.

## Decisions before implementation

Verify MLP/CNN dimensions, preprocessing, subsets, tokenization/corpus, initialization, practical SNR estimation/window/warm-up, reset connectivity, optimizer-state handling, comparator variants, regularization masks, diagnostics, and sweep interpretation. PyTorch is proposed for team code, with local .venv development and compatible Colab execution; exact versions are unverified. Keep any author JAX environment separate. See setup.md and data_sources.md.

Executable experiment settings and result conventions are deferred at the user's request. No official seed list is selected; 42 may be used for debugging only. Multi-seed evaluation and separate tuning/evaluation roles still apply.

## Resources

Measure wall time, memory, logging overhead, and checkpoint/resume behavior on actual hardware before estimating full runs. No end-to-end compute budget is established. Never launch the full matrix implicitly. Large Transformer sweeps and scale-256 remain unmeasured commitments.

## Evidence

Prepare verified Hugging Face/Kaggle datasets programmatically. Record full config, seeds, commit, environment/hardware, dataset revision, and metric timing with public W&B links. Compare against published numbers and run independent author-code verification without copying core code.

Old-task retention is a separate forgetting analysis, not a substitute for plasticity. Poster critique experiments need separate hypotheses and phase tags.
