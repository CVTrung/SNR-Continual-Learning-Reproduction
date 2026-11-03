"""Construct the permuted shakespeare task sequence.

Status: design scaffold; no executable implementation.
Inputs: Prepared base data, task settings, and task-generation seed.
Intended outputs: Reproducible tasks and metadata sufficient to reconstruct their order and transforms.
Responsibilities: Remap vocabulary consistently per task and align next-token targets. Verify context, padding, causal batching, and scale-dependent corpus selection. Do not download base datasets or define architectures.
"""
