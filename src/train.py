"""Coordinate sequential-task training through shared functions.

Status: design scaffold; no executable implementation.
Inputs: Model, tasks, optimizer, composed interventions, config, optional checkpoint.
Intended outputs: Per-task training records and resumable state.
Responsibilities: Specify forward/backward/update/intervention order. Preserve weights across tasks. Keep task/epoch/batch counters explicit; avoid duplicate trainers for each method.
"""
