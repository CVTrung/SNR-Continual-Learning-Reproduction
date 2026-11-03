"""Compute L2/L2 Init penalties with explicit parameter masks.

Status: design scaffold; no executable implementation.
Inputs: Current parameters, immutable initialization reference, coefficient, mask.
Intended outputs: Penalty terms and parameter-group metadata.
Responsibilities: Support all-eligible-weight and attention-only variants and composition with SNR. Do not silently substitute optimizer weight decay for the paper penalty.
"""
