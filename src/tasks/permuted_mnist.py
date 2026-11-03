"""Construct the permuted mnist task sequence.

Status: design scaffold; no executable implementation.
Inputs: Prepared base data, task settings, and task-generation seed.
Intended outputs: Reproducible tasks and metadata sufficient to reconstruct their order and transforms.
Responsibilities: Apply one pixel permutation consistently to all images within each task, including evaluation images. Preserve labels and record subset/permutation identifiers. Do not download base datasets or define architectures.
"""
