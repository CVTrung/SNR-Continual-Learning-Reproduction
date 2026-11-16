"""Sequential-task training.

Status: first executable version.
Weights are kept across tasks; only the data changes. `on_step` is called after every
optimizer step with (model, hidden_activations); `penalty(model)` (optional) is added to the
loss, so no per-method training loops are needed.
"""
import torch
import torch.nn.functional as F


def train_one_task(model, optimizer, x, y, batch_size=16, epochs=1,
                   generator=None, on_step=None, penalty=None):
    """Train on one task; return the online accuracy of the last epoch."""
    model.train()
    n = len(x)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(x.device)
        correct = torch.zeros((), device=x.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            logits, activations = model(xb, return_activations=True)
            loss = F.cross_entropy(logits, yb)
            if penalty is not None:
                loss = loss + penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == yb).sum()  # prediction before this update
            if on_step is not None:
                on_step(model, activations)
    return correct.item() / n


def train_sequence(model, optimizer, make_task_fn, num_tasks, batch_size=16,
                   epochs=1, seed=0, device="cpu", on_step=None, on_task_end=None,
                   penalty=None):
    """Train through tasks 0..num_tasks-1; return the list of per-task online accuracies."""
    generator = torch.Generator().manual_seed(seed)
    accuracies = []
    for t in range(num_tasks):
        x_np, y_np = make_task_fn(t)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        acc = train_one_task(model, optimizer, x, y, batch_size, epochs, generator,
                             on_step, penalty)
        accuracies.append(acc)
        if on_task_end is not None:
            on_task_end(t, acc)
    return accuracies