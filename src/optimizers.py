"""Construct SGD/Adam with explicit verified settings.

Status: design scaffold; no executable implementation.
Inputs: Parameters, optimizer name, learning rate, other options.
Intended outputs: Optimizer and serializable settings/state.
Responsibilities: Framework is undecided. Verify momentum, betas, epsilon, and regularization behavior; never assume defaults reproduce the paper.
"""
