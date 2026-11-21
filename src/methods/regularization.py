"""L2 and L2 Init penalties with explicit parameter masks, added to the training loss.

Status: executable.
L2:      (strength / 2) * sum ||theta||^2
L2 Init: (strength / 2) * sum ||theta - theta_0||^2, theta_0 = parameters at construction.
Masks (`select`):
- "all": every parameter, biases included. The paper does not say whether biases are
  penalized. We include them and record this as a team decision.
- "attention": Transformer attention weights and biases only (Q, K, V, output projection).
  This is the L2* / SNR+L2* ablation of Table 2.
These are explicit loss terms, not optimizer weight decay, so they compose with Adam exactly
as the paper writes them. They also compose with SNR, because the training loop takes a
penalty and an on_step hook independently. L2 Init keeps theta_0 from initialization; it is
not updated when a unit is reset.
"""


def select_parameters(model, select="all"):
    """Return the parameter list a penalty applies to."""
    if select == "all":
        return list(model.parameters())
    if select == "attention":
        return list(model.attention_parameters())
    raise ValueError(f"unknown parameter mask: {select}")


class L2:
    def __init__(self, strength, select="all"):
        self.strength = strength
        self.select = select

    def __call__(self, model):
        params = select_parameters(model, self.select)
        return 0.5 * self.strength * sum((p ** 2).sum() for p in params)


class L2Init:
    def __init__(self, model, strength, select="all"):
        self.strength = strength
        self.select = select
        self.reference = [p.detach().clone() for p in select_parameters(model, select)]

    def __call__(self, model):
        params = select_parameters(model, self.select)
        return 0.5 * self.strength * sum(
            ((p - p0) ** 2).sum() for p, p0 in zip(params, self.reference)
        )

    def state_dict(self):
        return {"reference": self.reference}

    def load_state_dict(self, state):
        self.reference = state["reference"]
