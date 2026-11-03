"""Construct the random label cifar task sequence.

Status: design scaffold; no executable implementation.
Inputs: Prepared base data, task settings, and task-generation seed.
Intended outputs: Reproducible tasks and metadata sufficient to reconstruct their order and transforms.
Responsibilities: Construct reproducible random-label tasks. Keep the main benchmark distinct from held-out/noise-schedule generalization variants. Do not download base datasets or define architectures.
"""
