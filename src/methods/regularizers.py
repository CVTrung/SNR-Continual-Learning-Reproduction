"""L2 and L2 Init penalties, added to the loss.

Status: first executable version.
L2:      (strength / 2) * sum ||theta||^2
L2 Init: (strength / 2) * sum ||theta - theta_0||^2, theta_0 = parameters at construction.
TEAM DECISIONS (log in docs/reproduction_notes.md): the 1/2 factor follows the paper's
Section 4; the penalty covers all parameters including biases. L2Init stores theta_0 once
and is not meant to be combined with neuron resets.
"""


class L2:
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, model):
        return 0.5 * self.strength * sum((p ** 2).sum() for p in model.parameters())


class L2Init:
    def __init__(self, model, strength):
        self.strength = strength
        self.reference = [p.detach().clone() for p in model.parameters()]

    def __call__(self, model):
        return 0.5 * self.strength * sum(
            ((p - p0) ** 2).sum() for p, p0 in zip(model.parameters(), self.reference)
        )