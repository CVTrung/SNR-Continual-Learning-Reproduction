"""Track reproducible runs in one public W&B project.

Status: design scaffold; no executable implementation.
Inputs: Resolved config, seeds, commit, environment/data/hardware provenance, metrics, phase.
Intended outputs: Public run links and local metadata/metric exports.
Responsibilities: Separate tuning/evaluation/author-comparison/critique runs. Never embed secrets or invent run links; retain local evidence if remote logging fails.
"""
