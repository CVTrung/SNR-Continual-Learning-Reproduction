# Preliminary v1 runs (superseded)

The teammates' first PyTorch version (commits c83c9b6..ab362e9): a 784-100-100-100-10 MLP with PyTorch default init, and SNR counting firing per example inside a batch, with an EMA of inter-firing gaps. Seeds 0-2, mostly 2400 tasks; no environment or commit recorded per run.

They are kept as development history. The 300-task eta runs here showed the collapse at large eta that led to the step-level fix (docs/reproduction_notes.md). Do not compare them with the paper as the team result.
