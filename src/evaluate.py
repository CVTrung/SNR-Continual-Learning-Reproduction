"""Measure performance under the explicit experiment protocol.

Status: design scaffold; no executable implementation.
Inputs: Predictions/targets or model/evaluation batches, metadata, metric settings.
Intended outputs: Named metrics with sample counts and evaluation timing.
Responsibilities: Distinguish online training accuracy, final-epoch loss, held-out evaluation, and optional old-task retention.
"""
