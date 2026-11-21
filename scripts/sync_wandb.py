"""Upload finished local runs (results/metrics/*.csv + .json) to one W&B project.

Run from the repo root after `wandb login`:
  python -m scripts.sync_wandb --project snr-reproduction [--entity <team>] [--dry-run]
Each local run becomes one W&B run with the same name. Its config comes from the run's JSON
(resolved options, commit, environment, seconds per task), and its rows are the per-task
metrics, logged with step = task. Tags are the run's phase, its benchmark, and
"uploaded-from-csv". That last tag marks runs logged after training from the saved files,
not live from the training process. A run already uploaded is skipped, so the script can be
re-run. Links go to results/wandb_runs.csv for the README and the deck. The repository
holds no credentials; `wandb login` stores the key in the user's profile.
"""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

METRICS = Path("results/metrics")
INDEX = Path("results/wandb_runs.csv")


def local_runs():
    for meta_path in sorted(METRICS.glob("*.json")):
        csv_path = meta_path.with_suffix(".csv")
        if csv_path.exists():
            yield meta_path.stem, json.loads(meta_path.read_text()), csv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="snr-reproduction")
    p.add_argument("--entity", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    done = {}
    if INDEX.exists():
        with open(INDEX, newline="") as f:
            done = {r["run"]: r["url"] for r in csv.DictReader(f)}
    todo = [(n, m, c) for n, m, c in local_runs() if n not in done]
    print(f"{len(done)} already uploaded, {len(todo)} to upload")
    if args.dry_run:
        for name, meta, _ in todo:
            print(name, meta.get("phase"), meta.get("benchmark"))
        return

    import wandb

    for name, meta, csv_path in todo:
        df = pd.read_csv(csv_path)
        run = wandb.init(project=args.project, entity=args.entity, name=name, config=meta,
                         tags=[meta.get("phase", "unknown"), meta.get("benchmark", "?"),
                               "uploaded-from-csv"],
                         group=f"{meta.get('benchmark')}-{meta.get('condition')}", reinit=True)
        for row in df.to_dict("records"):
            run.log({k: v for k, v in row.items() if k != "task"}, step=int(row["task"]))
        run.summary["summary_last10"] = meta.get("summary_last10")
        done[name] = run.url
        run.finish()
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "url"])
            w.writerows(sorted(done.items()))
        print("uploaded", name, done[name], flush=True)


if __name__ == "__main__":
    main()
