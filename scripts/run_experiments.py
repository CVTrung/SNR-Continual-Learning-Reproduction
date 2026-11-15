"""Minimal runner: baseline Permuted MNIST (no SNR yet).

Run from the repo root:  python -m scripts.run_experiments --tasks 20
Writes results/metrics/pm_<optimizer>_seed<seed>_tasks<N>.csv (small file, one row per task).
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from src.data.mnist import load_mnist
from src.evaluate import mean_last_fraction
from src.models.mlp import MLP
from src.optimizers import build_optimizer
from src.tasks.permuted_mnist import make_task, select_subset
from src.train import train_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=None)
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
    start = time.time()

    def report(t, acc):
        if t % 10 == 0 or t == args.tasks - 1:
            print(f"task {t:4d}  online acc {acc:.4f}  elapsed {time.time() - start:7.1f}s")

    accs = train_sequence(
        model, optimizer,
        make_task_fn=lambda t: make_task(x, y, t, base_seed=args.seed)[:2],
        num_tasks=args.tasks, seed=args.seed, on_task_end=report,
    )

    seconds = time.time() - start
    print(f"mean online acc over last 10% of tasks: {mean_last_fraction(accs):.4f}")
    print(f"{seconds / args.tasks:.2f} s/task  ->  about {seconds / args.tasks * 2400 / 60:.0f} min for 2400 tasks")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pm_{args.optimizer}_seed{args.seed}_tasks{args.tasks}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "online_accuracy"])
        writer.writerows(enumerate(accs))
    print("saved", path)


if __name__ == "__main__":
    main()