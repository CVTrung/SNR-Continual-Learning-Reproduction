"""Draw the team's own result figures from results/metrics into results/figures (PNG, 200 dpi).

Run from the repo root (after scripts.summarize_results):  python -m scripts.plot_results
Figures, all made by the team from team runs (never redrawn from the paper):
- pm_accuracy_sgd.png / pm_accuracy_adam.png: online accuracy per task, mean over seeds,
  20-task rolling mean, one line per condition (Table 1 / Figure 1 analogue)
- pm_dead_fraction.png: share of neurons silent for >= 1000 examples (Figure 1 analogue)
- eta_sweep.png: last-10% accuracy against eta, team vs paper Tables 10/11
- ps_cumulative_loss.png: cumulative average of the final-epoch loss (Figure 2 analogue)
- ps_inactive.png: FFN units inactive for >= 5 steps at each task end (Figure 3 analogue)
- ps_norms_entropy.png: Q, K, FFN weight norms and attention entropy (Figure 4 analogue)
Colours: fixed categorical order per condition (validated default palette of the dataviz
guide), so a condition keeps its colour in every figure.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.summarize_results import PAPER_ETA, pm_method, ps_condition  # noqa: E402

METRICS = Path("results/metrics")
FIGURES = Path("results/figures")
TABLES = Path("results/tables")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PM_COLORS = dict(zip(["None", "SNR", "CBP", "ReDO", "L2", "L2 Init", "S&P", "LayerNorm"], PALETTE))
PS_COLORS = dict(zip(["None", "SNR", "L2", "L2*", "L2 Init", "SNR+L2", "SNR+L2*", "CBP"], PALETTE))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("saved", FIGURES / name)


def runs(benchmark, phase=None):
    out = []
    for meta_path in sorted(METRICS.glob(f"{benchmark}_*.json")):
        meta = json.loads(meta_path.read_text())
        if phase and meta.get("phase") != phase:
            continue
        out.append((meta, pd.read_csv(meta_path.with_suffix(".csv"))))
    return out


def mean_curve(frames, column):
    n = min(len(f) for f in frames)
    return np.mean([f[column].to_numpy()[:n] for f in frames], axis=0)


def plot_pm():
    groups = {}
    for meta, df in runs("pm", "eval"):
        groups.setdefault((meta["optimizer"], pm_method(meta)[0]), []).append(df)
    if not groups:
        return
    for opt in ("sgd", "adam"):
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        for method, color in PM_COLORS.items():
            frames = groups.get((opt, method))
            if not frames:
                continue
            curve = pd.Series(mean_curve(frames, "online_accuracy")).rolling(20, min_periods=1).mean()
            ax.plot(curve, color=color, linewidth=2 if method in ("SNR", "None") else 1.4,
                    label=f"{method} ({len(frames)} seed{'s' if len(frames) > 1 else ''})")
        style(ax, f"Permuted MNIST, {opt.upper()}: online accuracy per task",
              "task", "online accuracy (20-task rolling mean)")
        ax.legend(fontsize=7.5, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
        save(fig, f"pm_accuracy_{opt}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, opt in zip(axes, ("sgd", "adam")):
        for method, color in PM_COLORS.items():
            frames = groups.get((opt, method))
            if frames:
                ax.plot(mean_curve(frames, "dead_fraction") * 100, color=color,
                        linewidth=2 if method in ("SNR", "None") else 1.2, label=method)
        style(ax, f"{opt.upper()}", "task", "dead neurons (%)" if opt == "sgd" else "")
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle("Neurons silent for >= 1000 consecutive examples", x=0.01, ha="left",
                 fontsize=11, color=INK)
    save(fig, "pm_dead_fraction.png")


def plot_eta():
    path = TABLES / "eta_sweep.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, opt in zip(axes, ("sgd", "adam")):
        geo = df[(df.optimizer == opt) & (df.estimator == "geometric")].sort_values("eta")
        hist = df[(df.optimizer == opt) & (df.estimator == "histogram")].sort_values("eta")
        paper = sorted(PAPER_ETA[opt].items())
        ax.plot([e for e, _ in paper], [v for _, v in paper], color=MUTED, linestyle="--",
                linewidth=1.5, marker="s", markersize=5, label="paper (2400 tasks, 5 seeds)")
        ax.plot(geo.eta, geo["mean"], color=PALETTE[0], linewidth=2, marker="o", markersize=6,
                label="team, geometric estimator")
        if len(hist):
            ax.plot(hist.eta, hist["mean"], color=PALETTE[1], linewidth=2, marker="^",
                    markersize=6, label="team, histogram estimator")
        ax.set_xscale("log")
        ticks = sorted(set(geo.eta) | set(e for e, _ in paper))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks], rotation=45, ha="right")
        ax.minorticks_off()
        style(ax, opt.upper(), "eta (log scale)", "last-10% accuracy" if opt == "sgd" else "")
    lo = min(df["mean"].min(), 0.8) - 0.02
    for ax in axes:
        ax.set_ylim(lo, 0.9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    tasks = int(df.n_tasks.iloc[0])
    fig.suptitle(f"SNR sensitivity to eta, Permuted MNIST ({tasks} tasks, seed 100)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, "eta_sweep.png")


def plot_ps():
    groups = {}
    for meta, df in runs("ps", "eval"):
        groups.setdefault(ps_condition(meta), []).append(df)
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for cond, color in PS_COLORS.items():
        if cond in groups:
            loss = mean_curve(groups[cond], "final_loss")
            ax.plot(np.cumsum(loss) / np.arange(1, len(loss) + 1), color=color, linewidth=2,
                    label=f"{cond} ({len(groups[cond])} seed{'s' if len(groups[cond]) > 1 else ''})")
    style(ax, "Permuted Shakespeare: cumulative average of final-epoch loss", "task",
          "cumulative average loss")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    save(fig, "ps_cumulative_loss.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for cond, color in PS_COLORS.items():
        if cond in groups:
            ax.plot(mean_curve(groups[cond], "inactive_units"), color=color, linewidth=2,
                    label=cond)
    style(ax, "FFN units inactive for >= 5 consecutive steps (of 256)", "task", "inactive units")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    save(fig, "ps_inactive.png")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, [("q_norm", "Query weight norm"), ("k_norm", "Key weight norm"),
                                       ("ffn_norm", "FFN weight norm"),
                                       ("attn_entropy", "Attention entropy (nats)")]):
        for cond, color in PS_COLORS.items():
            if cond in groups:
                ax.plot(mean_curve(groups[cond], col), color=color, linewidth=1.8, label=cond)
        style(ax, title, "task", "")
    axes[-1].legend(fontsize=7, frameon=False)
    save(fig, "ps_norms_entropy.png")


def main():
    plot_pm()
    plot_eta()
    plot_ps()


if __name__ == "__main__":
    main()
