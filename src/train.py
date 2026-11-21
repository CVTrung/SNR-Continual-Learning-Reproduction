"""Sequential-task training shared by every method and both benchmarks.

Status: executable.
Weights persist across tasks and only the data changes. Per optimizer step the order is
forward, then loss + optional penalty(model), then backward, then optimizer.step(), then
on_step(model, activations). This matches Algorithm 1: forward pass, optimize, then resets.
Two task types share the loop:
- "classification" (Permuted MNIST): x (N, D), y (N,).
- "lm" (Permuted Shakespeare): x (N, T+1) token windows, y=None. The inputs are x[:, :-1]
  and the next-token targets are x[:, 1:].
Per task it returns {"accuracy", "loss"} from the LAST epoch. Both are measured on each batch
BEFORE the update on that batch: online accuracy for PM, and the final-epoch loss for PS.
"""
import torch
import torch.nn.functional as F


def train_one_task(model, optimizer, x, y, batch_size=16, epochs=1, generator=None,
                   on_step=None, penalty=None, task_type="classification"):
    """Train on one task and return {"accuracy": float or None, "loss": float} (last epoch)."""
    model.train()
    n = len(x)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(x.device)
        correct = torch.zeros((), device=x.device)
        loss_sum = torch.zeros((), device=x.device)
        count = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if task_type == "lm":
                xb, yb = x[idx, :-1], x[idx, 1:]
            else:
                xb, yb = x[idx], y[idx]
            logits, activations = model(xb, return_activations=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            objective = loss if penalty is None else loss + penalty(model)
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(idx)
            if task_type != "lm":
                correct += (logits.argmax(dim=1) == yb).sum()  # prediction before this update
            count += len(idx)
            if on_step is not None:
                on_step(model, activations)
    accuracy = None if task_type == "lm" else correct.item() / count
    return {"accuracy": accuracy, "loss": loss_sum.item() / count}


def train_sequence(model, optimizer, make_task_fn, num_tasks, start_task=0, batch_size=16,
                   epochs=1, seed=0, generator=None, device="cpu", on_step=None,
                   on_task_end=None, penalty=None, task_type="classification"):
    """Train through tasks start_task..num_tasks-1; return the list of per-task result dicts.

    `generator` (a torch.Generator, restored from a checkpoint when resuming) shuffles the
    batches. When it is None, a fresh one is seeded with `seed`. make_task_fn(t) returns NumPy
    (x, y), with y None for "lm". on_task_end(t, result) runs after every task.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(seed)
    results = []
    for t in range(start_task, num_tasks):
        x_np, y_np = make_task_fn(t)
        x = torch.from_numpy(x_np).to(device)
        y = None if y_np is None else torch.from_numpy(y_np).to(device)
        result = train_one_task(model, optimizer, x, y, batch_size, epochs, generator,
                                on_step, penalty, task_type)
        results.append(result)
        if on_task_end is not None:
            on_task_end(t, result)
    return results
