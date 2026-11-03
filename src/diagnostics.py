"""Measure activity, weight norms, and attention entropy.

Status: design scaffold; no executable implementation.
Inputs: Activations, attention probabilities, selected weights, counters, definitions.
Intended outputs: Architecture-aware diagnostic summaries.
Responsibilities: Record inactivity windows, reduction axes, masking, entropy convention, and norm definitions. Diagnostic thresholds need not match SNR reset thresholds.
"""
