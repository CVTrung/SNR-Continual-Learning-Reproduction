"""Launch an explicitly selected experiment matrix.

Status: design scaffold; no executable implementation.
Inputs: Named experiment config and task/model/method/optimizer settings.
Intended outputs: Individually tracked runs with resolved configs.
Responsibilities: Validate inputs and select seeds explicitly. Never default to the entire matrix or paid compute.
"""
