"""Summarize run CSVs/JSONs in results/metrics into the deck/README tables in results/tables.

Run from the repo root:  python -m scripts.summarize_results
Reads every results/metrics/*.json written by scripts/run_experiments.py (preliminary_v1/ is
ignored: those runs used the superseded first implementation). Writes:
- results/tables/pm_main.csv   optimizer,method,hparam,n_seeds,n_tasks,mean,std,paper
- results/tables/eta_sweep.csv optimizer,eta,n_seeds,n_tasks,mean,std,resets_per_task,paper,estimator
- results/tables/ps_main.csv   condition,n_seeds,n_tasks,epochs,all_mean,all_std,first_mean,
                               first_std,last_mean,last_std,paper_all,paper_first,paper_last
Plus a Markdown copy of each table. Metric: PM = mean online accuracy over the last 10% of
tasks; PS = mean final-epoch loss over all tasks, the first 10% and the last 10% of tasks
(the paper's first/last 50 of 500). std is the sample std over seeds (0 for one seed).
Paper values are copied from the paper's Tables 1, 2, 10 and 11 for comparison only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = Path("results/metrics")
TABLES = Path("results/tables")

PAPER_PM = {  # Table 1, Permuted MNIST column
    "sgd": {"None": .71, "SNR": .85, "CBP": .84, "ReDO": .83, "L2": .82, "L2 Init": .83,
            "S&P": .83, "LayerNorm": .69},
    "adam": {"None": .64, "SNR": .88, "CBP": .88, "ReDO": .85, "L2": .88, "L2 Init": .88,
             "S&P": .88, "LayerNorm": .66},
}
PAPER_ETA = {  # Tables 10/11 (Figure 7), SNR row, eta grid 0.08 ... 0.00125
    "sgd": dict(zip([0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125],
                    [.849, .849, .848, .847, .850, .848, .847])),
    "adam": dict(zip([0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125],
                     [.887, .885, .880, .876, .872, .867, .872])),
}
PAPER_PS = {  # Table 2: all / first 50 / last 50 tasks
    "None": (2.7397, 1.8164, 3.0147), "L2": (0.2762, 0.1560, 0.3101),
    "L2*": (0.1506, 0.1874, 0.1092), "L2 Init": (1.5052, 1.2931, 1.5262),
    "SNR": (2.6872, 1.7338, 3.0242), "SNR+L2": (0.2177, 0.1370, 0.2551),
    "SNR+L2*": (0.1402, 0.1549, 0.0909), "CBP": (2.4922, 1.3732, 2.9410),
    "ReDO": (2.3258, 1.3689, 2.8119),
}
METHOD_ORDER = ["None", "SNR", "CBP", "ReDO", "L2", "L2 Init", "S&P", "LayerNorm"]
PS_ORDER = ["None", "L2", "L2*", "L2 Init", "SNR", "SNR+L2", "SNR+L2*", "CBP", "ReDO"]


def load_runs():
    runs = []
    for meta_path in sorted(METRICS.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        csv_path = meta_path.with_suffix(".csv")
        if csv_path.exists():
            runs.append((meta, pd.read_csv(csv_path)))
    return runs


def pm_method(meta):
    reset, penalty = meta["reset"], meta["penalty"]
    if meta.get("layer_norm") and reset == "none" and penalty == "none":
        return "LayerNorm", "-"
    if reset == "snr" and penalty == "none":
        return "SNR", f"eta={meta['eta']:g}"
    if reset == "cbp":
        return "CBP", f"r={meta['cbp_rate']:g}"
    if reset == "redo":
        return "ReDO", f"tau={meta['redo_tau']:g}, every {meta['redo_every']} tasks"
    if reset == "snp":
        return "S&P", f"shrink={meta['snp_shrink']:g}, sigma={meta['snp_sigma']:g}"
    if penalty == "l2":
        return "L2", f"lambda={meta['lam']:g}"
    if penalty == "l2init":
        return "L2 Init", f"lambda={meta['lam']:g}"
    return "None", "-"


def ps_condition(meta):
    star = "*" if meta["penalty_mask"] == "attention" else ""
    pen = {"none": "", "l2": "L2" + star, "l2init": "L2 Init"}[meta["penalty"]]
    reset = {"none": "", "snr": "SNR", "cbp": "CBP", "redo": "ReDO", "snp": "S&P"}[meta["reset"]]
    name = "+".join(x for x in (reset, pen) if x)
    return name or "None"


def last10(values):
    n = max(1, int(round(len(values) * 0.1)))
    return float(np.mean(values[-n:])), float(np.mean(values[:n])), float(np.mean(values))


def agg(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def write(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n")


def main():
    runs = load_runs()
    rows = {}
    for meta, df in runs:
        if meta["benchmark"] != "pm" or meta.get("phase") != "eval":
            continue
        method, hparam = pm_method(meta)
        key = (meta["optimizer"], method, hparam, meta["tasks"])
        rows.setdefault(key, []).append(last10(df["online_accuracy"].to_numpy())[0])
    pm = [dict(optimizer=o, method=m, hparam=h, n_seeds=len(v), n_tasks=t, mean=agg(v)[0],
               std=agg(v)[1], paper=PAPER_PM[o][m]) for (o, m, h, t), v in rows.items()]
    if pm:
        pm = pd.DataFrame(pm)
        pm["order"] = pm["method"].map(METHOD_ORDER.index)
        write(pm.sort_values(["optimizer", "order"], ascending=[False, True]).drop(columns="order"),
              "pm_main")

    rows = {}
    for meta, df in runs:
        if meta["benchmark"] != "pm" or meta["reset"] != "snr" or meta["penalty"] != "none":
            continue
        if meta["seed"] != 100:       # the sweep uses seed 100; main runs add seeds 101-102
            continue
        key = (meta["optimizer"], meta.get("estimator", "geometric"), meta["eta"], meta["tasks"])
        rows.setdefault(key, []).append((last10(df["online_accuracy"].to_numpy())[0],
                                         float(df["resets"].mean())))
    eta = [dict(optimizer=o, eta=e, n_seeds=len(v), n_tasks=t, mean=agg([a for a, _ in v])[0],
                std=agg([a for a, _ in v])[1], resets_per_task=float(np.mean([r for _, r in v])),
                paper=PAPER_ETA[o].get(e, np.nan) if est == "geometric" else np.nan,
                estimator=est)
           for (o, est, e, t), v in rows.items()]
    if eta:
        write(pd.DataFrame(eta).sort_values(["optimizer", "estimator", "eta"],
                                            ascending=[False, True, False]), "eta_sweep")

    rows = {}
    for meta, df in runs:
        if meta["benchmark"] != "ps":
            continue
        key = (ps_condition(meta), meta["tasks"], meta["epochs"])
        rows.setdefault(key, []).append(last10(df["final_loss"].to_numpy()))
    ps = []
    for (cond, t, ep), v in rows.items():
        last, first, allv = zip(*v)
        p = PAPER_PS.get(cond, (np.nan,) * 3)
        ps.append(dict(condition=cond, n_seeds=len(v), n_tasks=t, epochs=ep,
                       all_mean=agg(allv)[0], all_std=agg(allv)[1],
                       first_mean=agg(first)[0], first_std=agg(first)[1],
                       last_mean=agg(last)[0], last_std=agg(last)[1],
                       paper_all=p[0], paper_first=p[1], paper_last=p[2]))
    if ps:
        ps = pd.DataFrame(ps)
        ps["order"] = ps["condition"].map(lambda c: PS_ORDER.index(c) if c in PS_ORDER else 99)
        write(ps.sort_values("order").drop(columns="order"), "ps_main")
    print("wrote", sorted(p.name for p in TABLES.glob("*.csv")))


if __name__ == "__main__":
    main()
