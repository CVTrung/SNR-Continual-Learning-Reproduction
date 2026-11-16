"""Runner: Permuted MNIST with an optional intervention (none, snr, l2, l2init).

Run from the repo root:  python -m scripts.run_experiments --method l2init --lam 1e-3 --tasks 300
Writes results/metrics/pm_<method><tag>_<optimizer>_seed<seed>_tasks<N>.csv with one row per
task: online accuracy, fraction of hidden neurons inactive for >= 1000 examples, and the
number of neurons reset during the task (0 unless method is snr).
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from src.data.mnist import load_mnist
from src.diagnostics import InactivityTracker
from src.evaluate import mean_last_fraction
from src.methods.regularizers import L2, L2Init
from src.methods.snr import SNR
from src.models.mlp import MLP
from src.optimizers import build_optimizer
from src.tasks.permuted_mnist import make_task, select_subset
from src.train import train_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="none", choices=["none", "snr", "l2", "l2init"])
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eta", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data/mnist/mnist.npz")
    p.add_argument("--out", default="results/metrics")
    args = p.parse_args()

    lr = args.lr if args.lr is not None else (0.01 if args.optimizer == "sgd" else 0.001)
    torch.manual_seed(args.seed)

    x, y = load_mnist(args.data)
    x, y = select_subset(x, y, n=10_000, seed=args.seed)

    model = MLP()
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr)
    tracker = InactivityTracker([layer.out_features for layer in model.hidden_layers])
    snr = SNR(model, optimizer, eta=args.eta, beta=args.beta) if args.method == "snr" else None
    penalty = None
    if args.method == "l2":
        penalty = L2(args.lam)
    elif args.method == "l2init":
        penalty = L2Init(model, args.lam)
    dead, resets = [], []
    start = time.time()

    def on_step(m, acts):
        tracker.update(acts)
        if snr is not None:
            snr(m, acts)

    def report(t, acc):
        dead.append(tracker.dead_fraction())
        resets.append(snr.pop_resets() if snr is not None else 0)
        if t % 10 == 0 or t == args.tasks - 1:
            print(f"task {t:4d}  online acc {acc:.4f}  dead {dead[-1]:.3f}  "
                  f"resets {resets[-1]:5d}  elapsed {time.time() - start:7.1f}s")

    accs = train_sequence(
        model, optimizer,
        make_task_fn=lambda t: make_task(x, y, t, base_seed=args.seed)[:2],
        num_tasks=args.tasks, seed=args.seed, on_step=on_step, on_task_end=report,
        penalty=penalty,
    )

    seconds = time.time() - start
    print(f"mean online acc over last 10% of tasks: {mean_last_fraction(accs):.4f}")
    print(f"{seconds / args.tasks:.2f} s/task  ->  about {seconds / args.tasks * 2400 / 60:.0f} min for 2400 tasks")

    if args.method == "snr":
        tag = f"_eta{args.eta:g}"
    elif args.method in ("l2", "l2init"):
        tag = f"_lam{args.lam:g}"
    else:
        tag = ""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pm_{args.method}{tag}_{args.optimizer}_seed{args.seed}_tasks{args.tasks}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "online_accuracy", "dead_fraction", "resets"])
        writer.writerows(zip(range(len(accs)), accs, dead, resets))
    print("saved", path)


if __name__ == "__main__":
    main()