"""Run ONE continual-learning experiment: Permuted MNIST or Permuted Shakespeare.

Run from the repo root, for example:
  python -m scripts.run_experiments --benchmark pm --reset snr --eta 0.01 --tasks 50
  python -m scripts.run_experiments --benchmark ps --reset snr --penalty l2 --lam 1e-4 \
      --tasks 20 --epochs 10
  python -m scripts.run_experiments --config configs/experiments/main.yaml --reset cbp
A YAML --config supplies defaults (its keys are the long option names with "_"); command-line
options override it. Interventions compose: one --reset rule (none|snr|cbp|redo|snp) and one
--penalty (none|l2|l2init) with a parameter mask (all|attention). Layer Normalization is
--layer-norm (a model option).
Output: results/metrics/<benchmark>_<condition>_<optimizer>_seed<s>_tasks<N>[_ep<E>].csv,
one row per task. PM rows hold online_accuracy, loss, dead_fraction (>= 1000 silent examples)
and resets. PS rows hold final_loss, inactive_units (>= 5 silent steps), resets, Q/K/FFN
norms and attention entropy. The script never launches a sweep; scripts/sweep.py loops it.
"""
import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.diagnostics import InactivityTracker, attention_entropy, transformer_norms
from src.evaluate import mean_last_fraction
from src.methods.cbp import CBP
from src.methods.redo import ReDO
from src.methods.regularization import L2, L2Init
from src.methods.shrink_perturb import ShrinkPerturb
from src.methods.snr import SNR
from src.optimizers import build_optimizer
from src.tracking import Tracker
from src.train import train_sequence

DEFAULT_DATA = {
    "pm": "src/data/mnist/train-00000-of-00001.parquet",
    "ps": "src/data/shakespeare/all_shakespeare.txt",
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None, help="YAML file of default option values")
    p.add_argument("--benchmark", default="pm", choices=["pm", "ps"])
    p.add_argument("--reset", default="none", choices=["none", "snr", "cbp", "redo", "snp"])
    p.add_argument("--penalty", default="none", choices=["none", "l2", "l2init"])
    p.add_argument("--penalty-mask", default="all", choices=["all", "attention"])
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--optimizer", default=None, choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    # SNR
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--estimator", default="geometric", choices=["geometric", "histogram"])
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--m-init", type=float, default=1000.0)
    p.add_argument("--window-tasks", type=int, default=1)
    p.add_argument("--min-silence", type=int, default=6, help="steps; no SNR reset before")
    # penalties
    p.add_argument("--lam", type=float, default=1e-3)
    # baselines
    p.add_argument("--cbp-rate", type=float, default=1e-4)
    p.add_argument("--cbp-maturity", type=int, default=100)
    p.add_argument("--redo-tau", type=float, default=0.0)
    p.add_argument("--redo-every", type=int, default=1)
    p.add_argument("--snp-shrink", type=float, default=1e-5)
    p.add_argument("--snp-sigma", type=float, default=1e-4)
    # infrastructure
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="results/metrics")
    p.add_argument("--tag", default="", help="extra suffix for the output file name")
    p.add_argument("--tracking", default="none", choices=["none", "wandb", "trackio"])
    p.add_argument("--project", default="snr-reproduction")
    p.add_argument("--phase", default="smoke", choices=["smoke", "tune", "eval", "sweep"])
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0, help="tasks; 0 disables")
    p.add_argument("--resume", action="store_true")
    return p


def parse_args(argv=None):
    parser = build_parser()
    first, _ = parser.parse_known_args(argv)
    if first.config:
        import yaml
        with open(first.config, encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = set(values) - known
        if unknown:
            raise SystemExit(f"unknown keys in {first.config}: {sorted(unknown)}")
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    bench = args.benchmark
    args.optimizer = args.optimizer or ("adam" if bench == "ps" else "sgd")
    args.lr = args.lr if args.lr is not None else (0.01 if args.optimizer == "sgd" else 1e-3)
    args.epochs = args.epochs if args.epochs is not None else (100 if bench == "ps" else 1)
    args.batch_size = args.batch_size or (8 if bench == "ps" else 16)
    args.data = args.data or DEFAULT_DATA[bench]
    return args


def condition_name(args):
    """Readable, unique-enough name of the intervention and its hyperparameters."""
    parts = []
    if args.reset == "snr":
        est = "" if args.estimator == "geometric" else "hist"
        parts.append(f"snr{est}_eta{args.eta:g}")
    elif args.reset == "cbp":
        parts.append(f"cbp_r{args.cbp_rate:g}")
    elif args.reset == "redo":
        parts.append(f"redo_tau{args.redo_tau:g}_f{args.redo_every}")
    elif args.reset == "snp":
        parts.append(f"snp_s{args.snp_shrink:g}_sig{args.snp_sigma:g}")
    if args.penalty != "none":
        star = "star" if args.penalty_mask == "attention" else ""
        parts.append(f"{args.penalty}{star}_lam{args.lam:g}")
    if args.layer_norm:
        parts.append("ln")
    return "+".join(parts) if parts else "none"


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_benchmark(args):
    """Return (model, make_task_fn, layer_sizes, task_type, probe batch for diagnostics)."""
    if args.benchmark == "pm":
        from src.data.mnist import load_mnist
        from src.models.mlp import MLP
        from src.tasks.permuted_mnist import make_task, select_subset

        x, y = load_mnist(args.data)
        x, y = select_subset(x, y, n=10_000, seed=args.seed)
        model = MLP(layer_norm=args.layer_norm)
        sizes = [layer.out_features for layer in model.hidden_layers]
        return model, (lambda t: make_task(x, y, t, base_seed=args.seed)[:2]), sizes, \
            "classification", None

    from src.data.shakespeare import load_tempest
    from src.models.transformer import DecoderTransformer
    from src.tasks.permuted_shakespeare import make_task

    seqs, vocab = load_tempest(args.data, batch_size=args.batch_size)
    model = DecoderTransformer(vocab_size=len(vocab))
    make = lambda t: (make_task(seqs, t, len(vocab), base_seed=args.seed)[0], None)  # noqa: E731
    return model, make, [model.block.fc1.out_features], "lm", make


def build_interventions(args, model, optimizer):
    reset = None
    if args.reset == "snr":
        reset = SNR(model, optimizer, eta=args.eta, estimator=args.estimator, beta=args.beta,
                    m_init=args.m_init, window_tasks=args.window_tasks,
                    min_silence=args.min_silence)
    elif args.reset == "cbp":
        reset = CBP(model, optimizer, replacement_rate=args.cbp_rate,
                    maturity=args.cbp_maturity, batch_size=args.batch_size)
    elif args.reset == "redo":
        reset = ReDO(model, optimizer, tau=args.redo_tau, every_tasks=args.redo_every)
    elif args.reset == "snp":
        reset = ShrinkPerturb(shrink=args.snp_shrink, sigma=args.snp_sigma)
    penalty = None
    if args.penalty == "l2":
        penalty = L2(args.lam, select=args.penalty_mask)
    elif args.penalty == "l2init":
        penalty = L2Init(model, args.lam, select=args.penalty_mask)
    return reset, penalty


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, make_task_fn, sizes, task_type, probe = build_benchmark(args)
    model.to(args.device)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    reset, penalty = build_interventions(args, model, optimizer)
    unit = "step" if task_type == "lm" else "example"
    tracker_dead = InactivityTracker(sizes, device=args.device, unit=unit)
    if hasattr(reset, "reset_listeners"):
        reset.reset_listeners.append(tracker_dead.reset_units)

    name = f"{args.benchmark}_{condition_name(args)}_{args.optimizer}_seed{args.seed}_tasks{args.tasks}"
    if args.benchmark == "ps" or args.epochs != 1:
        name += f"_ep{args.epochs}"
    name += args.tag
    config = {k: v for k, v in vars(args).items()}
    config.update(condition=condition_name(args), commit=_git_commit(),
                  torch=torch.__version__, python=platform.python_version(),
                  host=platform.platform(), n_params=sum(p.numel() for p in model.parameters()))
    tracker = Tracker(args.tracking, project=args.project, name=name, config=config,
                      tags=[args.phase, args.benchmark, args.reset, args.penalty],
                      group=f"{args.benchmark}-{condition_name(args)}")

    generator = torch.Generator().manual_seed(args.seed)
    rows, start_task = [], 0
    ckpt = Path(args.checkpoint_dir) / f"{name}.pt"
    methods = {"reset": reset, "penalty": penalty}
    if args.resume and ckpt.exists():
        state = load_checkpoint(ckpt, model, optimizer, generator,
                                {k: m for k, m in methods.items() if m is not None})
        rows, start_task = state["history"], state["next_task"]
        print(f"resumed {ckpt} at task {start_task}")

    start = time.time()

    def on_step(m, acts):
        if reset is not None:
            reset(m, acts)
        tracker_dead.update(acts)

    def on_task_end(t, result):
        if hasattr(reset, "end_task"):
            reset.end_task(t)
        resets = reset.pop_resets() if hasattr(reset, "pop_resets") else 0
        row = {"task": t, "resets": resets}
        if task_type == "lm":
            row["final_loss"] = result["loss"]
            row["inactive_units"] = tracker_dead.dead_count(threshold=5)
            row.update(transformer_norms(model))
            with torch.no_grad():
                xb = torch.from_numpy(probe(t)[0][:args.batch_size, :-1]).to(args.device)
                _, _, probs = model(xb, return_activations=True, return_attention=True)
            row["attn_entropy"] = attention_entropy(probs)
        else:
            row["online_accuracy"] = result["accuracy"]
            row["loss"] = result["loss"]
            row["dead_fraction"] = tracker_dead.dead_fraction(threshold=1000)
        rows.append(row)
        tracker.log(row)
        if t % 10 == 0 or t == args.tasks - 1:
            metric = (f"loss {row['final_loss']:.4f}" if task_type == "lm"
                      else f"acc {row['online_accuracy']:.4f}")
            print(f"task {t:4d}  {metric}  resets {resets:5d}  elapsed {time.time() - start:7.1f}s",
                  flush=True)
        if args.checkpoint_every and (t + 1) % args.checkpoint_every == 0 and t + 1 < args.tasks:
            save_checkpoint(ckpt, model, optimizer, generator, t + 1, rows,
                            {k: m for k, m in methods.items() if m is not None})

    train_sequence(model, optimizer, make_task_fn, num_tasks=args.tasks, start_task=start_task,
                   batch_size=args.batch_size, epochs=args.epochs, generator=generator,
                   device=args.device, on_step=on_step, on_task_end=on_task_end,
                   penalty=penalty, task_type=task_type)

    seconds = time.time() - start
    done = max(1, args.tasks - start_task)
    key = "final_loss" if task_type == "lm" else "online_accuracy"
    summary = mean_last_fraction([r[key] for r in rows])
    print(f"mean {key} over last 10% of tasks: {summary:.4f}   ({seconds / done:.2f} s/task)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(config, seconds_per_task=seconds / done, summary_last10=summary,
                tracking_url=tracker.url)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    tracker.finish()
    if ckpt.exists():
        ckpt.unlink()          # run finished; the CSV is the record
    print("saved", path)
    return path


if __name__ == "__main__":
    main()
