"""Run the experiment matrix described by a configs/experiments/*.yaml file, in parallel.

Usage (from the repo root):
  python -m scripts.sweep configs/experiments/main.yaml --workers 12
  python -m scripts.sweep configs/experiments/main.yaml --dry-run      # list commands only
YAML format:
  common: {benchmark: pm, tasks: 600, phase: eval}   # options shared by every run
  seeds: [100, 101, 102]                             # each run is repeated per seed
  runs:                                              # one entry per condition
    - {optimizer: sgd, reset: snr, eta: 0.005}
Keys are run_experiments long options with "_" (layer_norm: true becomes --layer-norm).
Each worker is one single-threaded process (OMP_NUM_THREADS=1), which is the fastest
configuration for these small models on CPU. A run whose CSV already exists is skipped, so an
interrupted sweep can be restarted. Logs go to results/logs/ (ignored by Git).
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from scripts.run_experiments import condition_name, parse_args


def to_cli(options):
    """Convert a dict of options into run_experiments command-line arguments."""
    argv = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def expand(path):
    """Return the list of argv lists described by an experiment YAML file."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    common = spec.get("common", {})
    seeds = spec.get("seeds", [0])
    # seed-major order: every condition gets its first seed before any condition gets a second
    return [to_cli({**common, **run, "seed": seed}) for seed in seeds for run in spec["runs"]]


def output_path(argv):
    args = parse_args(argv)
    name = f"{args.benchmark}_{condition_name(args)}_{args.optimizer}_seed{args.seed}_tasks{args.tasks}"
    if args.benchmark == "ps" or args.epochs != 1:
        name += f"_ep{args.epochs}"
    return Path(args.out) / f"{name + args.tag}.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs, seen = [], set()
    for argv in (argv for cfg in args.configs for argv in expand(cfg)):
        key = output_path(argv)
        if key not in seen:           # a run shared by two configs is run once
            seen.add(key)
            jobs.append(argv)
    def running(argv):            # another process wrote this run's log in the last 3 minutes
        log = Path("results/logs") / (output_path(argv).stem + ".log")
        return log.exists() and time.time() - log.stat().st_mtime < 180
    todo = [argv for argv in jobs if not output_path(argv).exists() and not running(argv)]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        for argv in todo:
            print("python -m scripts.run_experiments " + " ".join(argv))
        return

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(argv):
        if output_path(argv).exists() or running(argv):   # re-check: another sweep may have done it
            print("skip  " + output_path(argv).stem, flush=True)
            return 0
        log = log_dir / (output_path(argv).stem + ".log")
        with open(log, "w") as f:
            code = subprocess.call([sys.executable, "-m", "scripts.run_experiments", *argv],
                                   stdout=f, stderr=subprocess.STDOUT, env=env)
        print(("ok    " if code == 0 else f"FAIL {code} ") + log.name, flush=True)
        return code

    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(run, todo))
    failed = sum(c != 0 for c in codes)
    print(f"done: {len(codes) - failed} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
