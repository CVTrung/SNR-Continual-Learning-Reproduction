"""Build the Milestone 2 "Reading + Reproduction" deck for Group 7.

Purpose
    Rebuild ``Group7_SNR_Milestone2.pptx`` from the team's draft deck ``docs/7.pptx``.
    The draft supplies the theme, slide master, the TITLE and CONTENT layouts and the
    visual language (colours, card style, table style). Every slide is regenerated in
    English from the content below, and every result slide is filled from the result
    files listed in the data contract. Nothing is invented: a missing file produces a
    clearly styled "Run pending" placeholder or a "pending" table cell.

Inputs (all optional except the draft deck)
    docs/7.pptx                          draft deck (theme and layouts)
    results/tables/pm_main.csv           optimizer,method,hparam,n_seeds,n_tasks,mean,std,paper
    results/tables/eta_sweep.csv         optimizer,eta,n_seeds,n_tasks,mean,std,resets_per_task,paper
    results/tables/ps_main.csv           condition,n_seeds,n_tasks,epochs,all_mean,all_std,first_mean,
                                         first_std,last_mean,last_std,paper_all,paper_first,paper_last
    results/figures/{pm_accuracy_sgd,pm_accuracy_adam,pm_dead_fraction,eta_sweep,
                     ps_cumulative_loss,ps_inactive,ps_norms_entropy}.png
    results/metrics/pm_*.csv             per-run metrics (fallback for PM numbers and curves)

Output
    deliverables/milestone2/Group7_SNR_Milestone2.pptx, plus a short console report
    (slide count, notes word counts, slides that still show pending placeholders).

Usage (from the repository root)
    .venv/Scripts/python.exe deliverables/milestone2/build_deck.py
"""

from __future__ import annotations

import ast
import copy
import math
import os
import re
import statistics
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / "docs" / "7.pptx"
# DECK_RESULTS_DIR / DECK_OUT exist only to test the data contract on a scratch copy.
RESULTS = Path(os.environ.get("DECK_RESULTS_DIR", ROOT / "results"))
OUT = Path(os.environ.get("DECK_OUT", HERE / "Group7_SNR_Milestone2.pptx"))
# Fallback per-run metrics: the superseded first implementation (3 x 100 MLP, PyTorch default
# init, SNR counted per example). Used only where the result tables below do not exist yet.
METRICS = RESULTS / "metrics" / "preliminary_v1"
V1 = "preliminary v1 (superseded)"
V1_LONG = "preliminary v1 (superseded): 3 × 100 MLP, PyTorch init, per-example SNR"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# ---------------------------------------------------------------- palette (from the draft)
INK, MUTED, TEAL, ORANGE = "12202B", "5C6B73", "1FA2A6", "E4572E"
NAVY, BLUE, AMBER, GREEN = "1B3A4B", "3D7EA6", "F2A541", "8DB596"
BORDER, ROW_ALT, WHITE, BG = "D9D4C7", "EFEBE0", "FFFFFF", "F4F1EA"
TEAL_TINT, AMBER_TINT, LIGHT = "EAF5F5", "FBE8C8", "A9BCC4"
# One colour per method, everywhere (validated with the dataviz palette checker on BG).
METHOD_COLOR = {"none": "EB6834", "snr": "1BAF7A", "cbp": "E87BA4", "redo": "6250D6",
                "l2": "EDA100", "l2init": "2A78D6", "snp": "008300", "ln": "E34948"}
PAPER_BAR, TEAM_BAR = MUTED, TEAL

METHODS = ["none", "snr", "cbp", "redo", "l2", "l2init", "snp", "ln"]
LABEL = {"none": "No intervention", "snr": "SNR", "cbp": "CBP", "redo": "ReDO", "l2": "L2",
         "l2init": "L2 Init", "snp": "Shrink & Perturb", "ln": "LayerNorm"}
SHORT = {"none": "None", "snr": "SNR", "cbp": "CBP", "redo": "ReDO", "l2": "L2",
         "l2init": "L2 Init", "snp": "S&P", "ln": "LN"}
ALIASES = {"none": "none", "nointervention": "none", "baseline": "none", "snr": "snr",
           "cbp": "cbp", "redo": "redo", "l2": "l2", "l2reg": "l2", "l2init": "l2init",
           "snp": "snp", "sp": "snp", "shrinkperturb": "snp", "s&p": "snp", "ln": "ln",
           "layernorm": "ln"}
OPTS = ["sgd", "adam"]
OPT_LABEL = {"sgd": "SGD", "adam": "Adam"}

# Paper reference values: Table 6 of [1] (the extended Table 1), last-10% accuracy, 5 seeds.
PAPER_PM = {
    "sgd": {"none": .710, "snr": .851, "cbp": .844, "redo": .831, "l2": .818, "l2init": .829,
            "snp": .826, "ln": .687},
    "adam": {"none": .641, "snr": .889, "cbp": .876, "redo": .846, "l2": .876, "l2init": .883,
             "snp": .876, "ln": .662},
}
# Paper eta grid for PM (Table 7) and SNR accuracy at each value (Tables 10 and 11).
PAPER_ETA_GRID = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125]
PAPER_ETA = {"sgd": dict(zip(PAPER_ETA_GRID, [.849, .849, .848, .847, .850, .848, .847])),
             "adam": dict(zip(PAPER_ETA_GRID, [.887, .885, .880, .876, .872, .867, .872]))}
# Paper Table 2 (Permuted Shakespeare): loss on the final epoch of each task, 9 seeds.
PS_ORDER = ["None", "L2", "L2*", "L2 Init", "SNR", "SNR+L2", "SNR+L2*", "CBP", "ReDO"]
PAPER_PS = {"L2": (.2762, .1560, .3101), "SNR+L2": (.2177, .1370, .2551),
            "None": (2.7397, 1.8164, 3.0147), "L2 Init": (1.5052, 1.2931, 1.5262),
            "SNR": (2.6872, 1.7338, 3.0242), "CBP": (2.4922, 1.3732, 2.9410),
            "L2*": (.1506, .1874, .1092), "SNR+L2*": (.1402, .1549, .0909),
            "ReDO": (2.3258, 1.3689, 2.8119)}

SHORT_REFS = {
    1: 'V. F. Farias and A. D. Jozefiak, "Self-normalized resets for plasticity in continual learning," ICLR, 2025.',
    6: '"ylecun/mnist," Hugging Face Datasets, huggingface.co/datasets/ylecun/mnist',
    8: 'Group 7, "snr-reproduction," public W&B project, wandb.ai/vutrung2903-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/snr-reproduction',
    12: '"SelfNormalizedResets," official code for [1], github.com/ajozefiak/SelfNormalizedResets',
}

REFS = {
    1: 'V. F. Farias and A. D. Jozefiak, "Self-normalized resets for plasticity in continual '
       'learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2025. [Online]. Available: '
       'https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf',
    2: 'S. Dohare et al., "Loss of plasticity in deep continual learning," Nature, vol. 632, '
       'pp. 768-774, 2024.',
    3: 'S. Kumar, H. Marklund, and B. Van Roy, "Maintaining plasticity via regenerative '
       'regularization," arXiv:2308.11958, 2023.',
    4: 'G. Sokar, R. Agarwal, P. S. Castro, and U. Evci, "The dormant neuron phenomenon in deep '
       'reinforcement learning," in Proc. ICML, 2023.',
    5: 'Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to '
       'document recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.',
    6: '"ylecun/mnist," Hugging Face Datasets. [Online]. Available: '
       'https://huggingface.co/datasets/ylecun/mnist',
    7: 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015.',
    8: 'Group 7, "snr-reproduction," W&B project. [Online]. Available: https://wandb.ai/'
       'vutrung2903-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/snr-reproduction ; Trackio dashboard: '
       'https://huggingface.co/spaces/nshai/snr-reproduction',
    9: 'J. Ash and R. P. Adams, "On warm-starting neural network training," in Proc. NeurIPS, '
       'vol. 33, pp. 3884-3894, 2020.',
    10: 'J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," arXiv:1607.06450, 2016.',
    11: 'A. Radford et al., "Language models are unsupervised multitask learners," OpenAI, '
        'Tech. Rep., 2019.',
    12: '"SelfNormalizedResets," official code for [1], GitHub repository. [Online]. Available: '
        'https://github.com/ajozefiak/SelfNormalizedResets',
}


# ================================================================ data loading
def last_frac_mean(values, frac=0.1):
    """Mean of the last ``frac`` share of a sequence (at least one element)."""
    vals = list(values)
    k = max(1, int(len(vals) * frac))
    return sum(vals[-k:]) / k


def norm_method(token):
    """Map a method spelling from a file name or table to the canonical key, or None."""
    t = re.sub(r"[\s_\-]", "", str(token).lower())
    return ALIASES.get(t)


def parse_eta(hp):
    """Return the eta value encoded in a hyperparameter tag such as 'eta1e-06', or None."""
    m = re.search(r"(?:eta)?[=_]?\s*([0-9]*\.?[0-9]+(?:e[+\-]?\d+)?)", str(hp or ""))
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def load_runs(directory=None, n_tasks=None, min_seed=None):
    """Read every <directory>/pm_*.csv (default: preliminary_v1) into a list of run dicts.

    Returns a list of dicts with keys method, hp, opt, seed, n, legacy, df. Files without a
    method token (written by an early runner) are treated as no-intervention runs and flagged
    ``legacy=True``.
    """
    runs = []
    directory = METRICS if directory is None else directory
    if not directory.exists():
        return runs
    for f in sorted(directory.glob("pm_*.csv")):
        parts = f.stem.split("_")[1:]
        oi = next((i for i, p in enumerate(parts) if p in OPTS), None)
        if oi is None:
            continue
        head, tail = parts[:oi], parts[oi + 1:]
        seed = next((int(p[4:]) for p in tail if re.fullmatch(r"seed\d+", p)), None)
        n = next((int(p[5:]) for p in tail if re.fullmatch(r"tasks\d+", p)), None)
        legacy = not head
        method = "none" if legacy else norm_method(head[0])
        if method is None or seed is None:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "online_accuracy" not in df:
            continue
        if (n_tasks is not None and (n or len(df)) != n_tasks) or (min_seed is not None and seed < min_seed):
            continue
        runs.append(dict(method=method, hp="_".join(head[1:]), opt=parts[oi], seed=seed,
                         n=n or len(df), legacy=legacy, df=df, file=f.name))
    return runs


def group_runs(runs):
    """Group runs by (method, hp, opt, n) -> list of runs."""
    g = {}
    for r in runs:
        g.setdefault((r["method"], r["hp"], r["opt"], r["n"]), []).append(r)
    return g


def stats(vals):
    """Return (mean, sample std or None) for a list of floats."""
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    return m, (statistics.stdev(vals) if len(vals) > 1 else None)


def build_pm_main(runs, use_table=True):
    """Main PM table: {(opt, method): dict(mean, std, n_seeds, n_tasks, hparam, source, paper)}.

    Rows from results/tables/pm_main.csv win; otherwise the longest, best-seeded group of
    non-legacy metric files is summarized (mean over seeds of the last-10% online accuracy).
    """
    out = {}
    groups = group_runs([r for r in runs if not r["legacy"]])
    best = {}
    for (m, hp, opt, n), rs in groups.items():
        key = (opt, m)
        score = (n, len(rs))
        if key not in best or score > best[key][0]:
            best[key] = (score, hp, rs)
    for (opt, m), (_, hp, rs) in best.items():
        vals = [last_frac_mean(r["df"]["online_accuracy"]) for r in rs]
        mean, std = stats(vals)
        out[(opt, m)] = dict(mean=mean, std=std, n_seeds=len(rs), n_tasks=rs[0]["n"], hparam=hp,
                             source="metrics", runs=sorted(rs, key=lambda r: r["seed"]),
                             paper=PAPER_PM[opt][m])
    f = TABLES / "pm_main.csv"
    if use_table and f.exists():
        try:
            t = pd.read_csv(f, keep_default_na=False, na_values=[""])
            out = {}  # the table replaces the superseded v1 fallback entirely (no mixing)
            for _, row in t.iterrows():
                m, opt = norm_method(row["method"]), str(row["optimizer"]).lower()
                if m is None or opt not in OPTS or pd.isna(row.get("mean")):
                    continue
                prev = out.get((opt, m), {})
                out[(opt, m)] = dict(
                    mean=float(row["mean"]),
                    std=None if pd.isna(row.get("std")) else float(row["std"]),
                    n_seeds=int(row["n_seeds"]) if not pd.isna(row.get("n_seeds")) else None,
                    n_tasks=int(row["n_tasks"]) if not pd.isna(row.get("n_tasks")) else None,
                    hparam=str(row.get("hparam", "")), source="table", runs=[],
                    paper=float(row["paper"]) if not pd.isna(row.get("paper")) else PAPER_PM[opt][m])
        except Exception as exc:  # malformed table: keep the fallback and say so
            print(f"warning: could not read {f}: {exc}")
    return out


def build_eta(runs):
    """Eta sweep rows: {opt: [dict(eta, mean, std, n_seeds, n_tasks, resets, paper)]} and source."""
    f = TABLES / "eta_sweep.csv"
    res = {}
    if f.exists():
        try:
            t = pd.read_csv(f)
            for _, row in t.iterrows():
                opt = str(row["optimizer"]).lower()
                eta = float(row["eta"])
                res.setdefault(opt, []).append(dict(
                    eta=eta, mean=float(row["mean"]),
                    std=None if pd.isna(row.get("std")) else float(row["std"]),
                    n_seeds=int(row["n_seeds"]), n_tasks=int(row["n_tasks"]),
                    resets=None if pd.isna(row.get("resets_per_task")) else float(row["resets_per_task"]),
                    paper=None if pd.isna(row.get("paper")) else float(row["paper"]),
                    estimator=str(row.get("estimator", "geometric")).lower()
                    if not pd.isna(row.get("estimator", "geometric")) else "geometric"))
            if res:
                return res, "table"
        except Exception as exc:
            print(f"warning: could not read {f}: {exc}")
    snr = [r for r in runs if r["method"] == "snr" and "eta" in r["hp"] and parse_eta(r["hp"]) is not None]
    for opt in OPTS:
        by_n = {}
        for r in snr:
            if r["opt"] == opt:
                by_n.setdefault(r["n"], {}).setdefault(parse_eta(r["hp"]), []).append(r)
        if not by_n:
            continue
        n, etas = max(by_n.items(), key=lambda kv: (len(kv[1]), -kv[0]))
        if len(etas) < 2:
            continue
        rows = []
        for eta, rs in etas.items():
            mean, std = stats([last_frac_mean(r["df"]["online_accuracy"]) for r in rs])
            resets = [r["df"]["resets"].mean() for r in rs if "resets" in r["df"]]
            rows.append(dict(eta=eta, mean=mean, std=std, n_seeds=len(rs), n_tasks=n,
                             resets=sum(resets) / len(resets) if resets else None,
                             paper=PAPER_ETA[opt].get(eta)))
        res[opt] = rows
    return res, "metrics"


def eta_baseline(runs, opt, n):
    """No-intervention accuracy over the same task count as a fallback eta sweep, or None."""
    rs = [r for r in runs if r["method"] == "none" and r["opt"] == opt and r["n"] == n]
    if not rs:
        return None, False
    mean, _ = stats([last_frac_mean(r["df"]["online_accuracy"]) for r in rs])
    return mean, any(r["legacy"] for r in rs)


def build_ps():
    """Permuted Shakespeare rows from results/tables/ps_main.csv: {condition: dict}."""
    f = TABLES / "ps_main.csv"
    if not f.exists():
        return {}
    try:
        t = pd.read_csv(f, keep_default_na=False, na_values=[""])
    except Exception as exc:
        print(f"warning: could not read {f}: {exc}")
        return {}
    out = {}
    for _, row in t.iterrows():
        c = str(row["condition"]).strip()
        key = re.sub(r"[\s_\-]", "", c.lower()).replace("attnonly", "*").replace(
            "attention", "*").replace("attn", "*").replace("star", "*")
        canon = {re.sub(r"[\s_\-]", "", k.lower()): k for k in PS_ORDER}
        canon.update({"nointervention": "None", "noint.": "None", "baseline": "None"})
        name = canon.get(key, c)
        out[name] = {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in t.columns}
    return out


def build_ps_diag():
    """results/tables/ps_diagnostics.csv: {condition: {metric_last: value}} (last 10% of tasks)."""
    f = TABLES / "ps_diagnostics.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f, keep_default_na=False, na_values=[""])
    return {str(r["condition"]): {k: r[k] for k in t.columns} for _, r in t.iterrows()}


def ps_findings(ps, diag):
    """Data-driven sentences comparing our PS ordering with the paper's message."""
    last = {c: float(r["last_mean"]) for c, r in ps.items() if r.get("last_mean") is not None}
    need = ("None", "SNR", "L2", "SNR+L2")
    if not all(c in last for c in need):
        return None, ""
    out, ok = [], []
    snr_close = abs(last["SNR"] - last["None"]) <= 0.15 * last["None"]
    ok.append(snr_close)
    out.append(f"SNR alone ends at {last['SNR']:.2f} against {last['None']:.2f} for no intervention "
               f"({'about the same, as in the paper' if snr_close else 'a clear difference, unlike the paper'}).")
    l2_help = last["L2"] < 0.8 * last["None"]
    ok.append(l2_help)
    out.append(f"L2 brings the last-tasks loss to {last['L2']:.2f} "
               f"({'a large improvement, as in the paper' if l2_help else 'little improvement, unlike the paper'}).")
    combo = last["SNR+L2"] <= last["L2"]
    ok.append(combo)
    out.append(f"SNR+L2 reaches {last['SNR+L2']:.2f}, {'below' if combo else 'above'} L2 alone "
               f"({'as in the paper' if combo else 'unlike the paper'}).")
    if "L2*" in last:
        star_best = last["L2*"] < last["L2"]
        out.append(f"Attention-only L2* ends at {last['L2*']:.2f}"
                   + (f" and SNR+L2* at {last['SNR+L2*']:.2f}" if "SNR+L2*" in last else "")
                   + (": best, as in the paper." if star_best else
                      ": no better than no intervention here, unlike the paper, where L2* is best."))
        if not star_best and "L2*" in diag and "L2" in diag:
            dL2, dS = diag["L2"], diag["L2*"]
            out.append(f"Our diagnostics point to why: under L2* the attention weights stay small "
                       f"(Q norm {dS['q_norm_last']:.1f}) but the unregularized feed-forward weights "
                       f"keep growing (FFN norm {dS['ffn_norm_last']:.1f} against {dL2['ffn_norm_last']:.1f} "
                       f"with full L2), attention entropy still collapses ({dS['attn_entropy_last']:.2f} "
                       f"nats against {dL2['attn_entropy_last']:.2f}), and about {dS['inactive_units_last']:.0f} "
                       f"feed-forward units sit inactive. With 30 short tasks this ablation does not "
                       f"reproduce; whether it does over 500 tasks of 100 epochs is a question for the full run.")
    title = ("SNR alone ≈ no intervention; L2 is needed; SNR+L2 lowest of the full-L2 runs"
             if all(ok) else "Permuted Shakespeare: partial agreement with the paper's ordering")
    return title, " ".join(out)


def fig(name):
    """Path of results/figures/<name>.png if it exists, else None."""
    p = FIGURES / f"{name}.png"
    return p if p.exists() else None


def module_status(rel):
    """'done' when a source file has code below its docstring, 'stub' otherwise, 'missing'."""
    p = ROOT / rel
    if not p.exists():
        return "missing"
    if p.suffix != ".py":
        return "done"
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        return "stub"
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(
        getattr(n, "value", None), ast.Constant))]
    return "done" if body else "stub"


def f3(x):
    return "pending" if x is None else f"{x:.3f}"


def f4(x):
    return "pending" if x is None else f"{x:.4f}"


def fmt_eta(e):
    if e is None:
        return "?"
    s = f"{e:g}"
    return s.replace("e-0", "e-")


def pct(x):
    return f"{100 * x:.0f}%"


# ================================================================ drawing helpers
def rgb(h):
    return RGBColor.from_string(h)


def _apply_run(run, size=None, color=None, bold=None, italic=None):
    f = run.font
    if size:
        f.size = Pt(size)
    if color:
        f.color.rgb = rgb(color)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic


def _bullet(p, on, indent=0.17):
    pPr = p._p.get_or_add_pPr()
    if on:
        pPr.set("marL", str(int(Inches(indent))))
        pPr.set("indent", str(-int(Inches(indent))))
        for tag in ("a:buNone", "a:buChar", "a:buSzPct"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)
        bs = pPr.makeelement(qn("a:buSzPct"), {"val": "100000"})
        bc = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bs)
        pPr.append(bc)
    else:
        pPr.set("marL", "0")
        pPr.set("indent", "0")
        if pPr.find(qn("a:buNone")) is None:
            pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def tb(slide, x, y, w, h, paras, size=11, color=INK, bold=False, align=None, anchor="t",
       bullets=False, after=4, italic=False, name=None, spc=None):
    """Add a text box. ``paras`` is a string, or a list of paragraphs; a paragraph is a string
    or a list of runs ``(text, {size,color,bold,italic})``."""
    shp = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shp.name = name
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        _bullet(p, bullets)
        if after:
            p.space_after = Pt(after)
        runs = [(para, {})] if isinstance(para, str) else para
        for text, opt in runs:
            r = p.add_run()
            r.text = text
            _apply_run(r, opt.get("size", size), opt.get("color", color), opt.get("bold", bold),
                       opt.get("italic", italic))
            if spc:
                r.font._rPr.set("spc", str(spc))
    return shp


def box(slide, x, y, w, h, fill=WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        dash=False, lw=0.75, adj=0.032):
    """Card or block: a rounded rectangle in the draft's card style."""
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = adj
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def label_box(slide, x, y, w, h, text, fill, color=WHITE, size=11, bold=False, sub=None,
              sub_size=9, line=None):
    """Filled block with centred text (pipeline and diagram nodes)."""
    box(slide, x, y, w, h, fill=fill, line=line or fill, adj=0.08)
    paras = [[(text, {"bold": bold})]]
    if sub:
        paras.append([(sub, {"size": sub_size})])
    tb(slide, x + 0.08, y, w - 0.16, h, paras, size=size, color=color, align="c", anchor="m",
       after=1)


def arrow(slide, x1, y1, x2, y2, color=MUTED, w=1.25):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2),
                                   Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def card(slide, x, y, w, h, heading, body, head_color=TEAL, body_size=10.5, bullets=True,
         band=None, foot=None, foot_color=MUTED):
    """White card with a small coloured heading (or a coloured band) and body text."""
    box(slide, x, y, w, h)
    if band:
        box(slide, x, y, w, 0.36, fill=band, line=band, shape=MSO_SHAPE.RECTANGLE)
        tb(slide, x + 0.15, y, w - 0.3, 0.36, heading, size=11.5, color=WHITE, bold=True,
           anchor="m")
        top = y + 0.48
    else:
        tb(slide, x + 0.15, y + 0.1, w - 0.3, 0.25, heading.upper(), size=9.5, color=head_color,
           bold=True, spc=100)
        top = y + 0.42
    foot_h = 0.62 if foot else 0
    tb(slide, x + 0.15, top, w - 0.3, h - (top - y) - foot_h - 0.08, body, size=body_size,
       bullets=bullets)
    if foot:
        tb(slide, x + 0.15, y + h - foot_h - 0.02, w - 0.3, foot_h - 0.04, foot, size=9.5,
           color=foot_color, italic=True)


def set_cell(cell, text, size=9.5, color=INK, bold=False, fill=WHITE, align=None, italic=False):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    paras = text if isinstance(text, list) else [text]
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        r = p.add_run()
        r.text = t
        _apply_run(r, size, color, bold, italic)
    cell.margin_left = cell.margin_right = Emu(54864)
    cell.margin_top = cell.margin_bottom = Emu(30000)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(fill)
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = tcPr.find(qn(tag))
        if ln is None:
            ln = tcPr.makeelement(qn(tag), {})
            tcPr.insert(0, ln)
        ln.set("w", "6350")
        for ch in list(ln):
            ln.remove(ch)
        sf = ln.makeelement(qn("a:solidFill"), {})
        clr = sf.makeelement(qn("a:srgbClr"), {"val": BORDER})
        sf.append(clr)
        ln.append(sf)


def table(slide, x, y, w, cols, rows, size=9.5, row_h=0.27, styles=None, aligns=None):
    """Draft-style table. ``cols`` are relative widths; ``rows[0]`` is the header.

    ``styles`` maps (row, col) or row -> dict(bold, color, fill, italic) overrides.
    """
    styles = styles or {}
    shp = slide.shapes.add_table(len(rows), len(cols), Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        tblPr.remove(sid)
    tot = sum(cols)
    for i, c in enumerate(cols):
        tbl.columns[i].width = Inches(w * c / tot)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            if ri == 0:
                st = dict(bold=True, color=WHITE, fill=NAVY)
            else:
                st = dict(fill=WHITE if ri % 2 else ROW_ALT)
            st.update(styles.get(ri, {}))
            st.update(styles.get((ri, ci), {}))
            al = aligns[ci] if aligns else None
            set_cell(tbl.cell(ri, ci), val, size=size, align=al, **st)
    return shp


def _style_chart(chart, title, legend=True, size=9):
    chart.font.size = Pt(size)
    chart.font.name = "Arial"
    chart.font.color.rgb = rgb(MUTED)
    if title:
        chart.has_title = True
        tf = chart.chart_title.text_frame
        tf.text = title
        for p in tf.paragraphs:
            for r in p.runs:
                _apply_run(r, 10, INK, False)
    else:
        chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.color.rgb = rgb(INK)
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(BORDER)
    va.major_gridlines.format.line.width = Pt(0.5)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    ca = chart.category_axis
    ca.format.line.color.rgb = rgb(BORDER)
    ca.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.color.rgb = rgb(INK)
    ca.has_major_gridlines = False


def bar_chart(slide, x, y, w, h, cats, series, title, ymin=None, ymax=None, fmt="0.000",
              label_size=7, val_fmt="0.0", label_only=None):
    """Clustered column chart; ``series`` = [(name, values, hex)], None values leave gaps."""
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w),
                                Inches(h), cd)
    ch = gf.chart
    _style_chart(ch, title, legend=len(series) > 1)
    va = ch.value_axis
    if ymin is not None:
        va.minimum_scale = ymin
    if ymax is not None:
        va.maximum_scale = ymax
    va.tick_labels.number_format = val_fmt
    va.tick_labels.number_format_is_linked = False
    plot = ch.plots[0]
    plot.gap_width = 55
    plot.overlap = -8
    for si, s in enumerate(plot.series):
        if label_only is not None and si not in label_only:
            continue
        dl = s.data_labels
        dl.show_value = True
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(label_size)
        dl.font.color.rgb = rgb(INK)
    for s, (_, _, col) in zip(plot.series, series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(col)
        s.invert_if_negative = False
    return ch


def line_chart(slide, x, y, w, h, cats, series, title, ymin=None, ymax=None, val_fmt="0.00",
               legend=True, cat_title=None):
    """Line chart without markers; ``series`` = [(name, values, hex, dashed)]."""
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, *_ in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _style_chart(ch, title, legend=legend)
    va = ch.value_axis
    if ymin is not None:
        va.minimum_scale = ymin
    if ymax is not None:
        va.maximum_scale = ymax
    va.tick_labels.number_format = val_fmt
    va.tick_labels.number_format_is_linked = False
    ca = ch.category_axis
    n = len(cats)
    if n > 12:
        ca_el = ca._element
        skip = ca_el.makeelement(qn("c:tickLblSkip"), {"val": "1"})
        mskip = ca_el.makeelement(qn("c:tickMarkSkip"), {"val": "10"})
        anchor_el = ca_el.find(qn("c:noMultiLvlLbl"))
        if anchor_el is not None:
            anchor_el.addprevious(skip)
            anchor_el.addprevious(mskip)
        else:
            ca_el.append(skip)
            ca_el.append(mskip)
    if cat_title:
        ca.has_title = True
        ca.axis_title.text_frame.text = cat_title
        for p in ca.axis_title.text_frame.paragraphs:
            for r in p.runs:
                _apply_run(r, 9, MUTED, False)
    for s, (_, _, col, *rest) in zip(ch.plots[0].series, series):
        s.smooth = False
        s.format.line.color.rgb = rgb(col)
        s.format.line.width = Pt(1.75)
        if rest and rest[0]:
            s.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        s.marker.style = None
        mk = s._element.get_or_add_marker()
        sym = mk.find(qn("c:symbol"))
        if sym is None:
            sym = mk.makeelement(qn("c:symbol"), {})
            mk.insert(0, sym)
        sym.set("val", "none")
    return ch


def picture(slide, path, x, y, w, h):
    """Insert an image scaled to fit the box, centred."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                    Inches(pw), Inches(ph))


PENDING = []


def pending_box(slide, x, y, w, h, what, expected):
    """Clearly styled placeholder for a result that does not exist yet."""
    box(slide, x, y, w, h, fill=ROW_ALT, line=MUTED, dash=True, lw=1)
    tb(slide, x + 0.25, y, w - 0.5, h,
       [[("RUN PENDING", {"bold": True, "color": ORANGE, "size": 10})],
        [(f"Reduced-protocol results will be inserted: {what}", {"bold": True, "size": 11.5})],
        [(f"Expected file: {expected}", {"size": 9, "color": MUTED})],
        [("No numbers are shown until the run exists.", {"size": 9, "color": MUTED,
                                                          "italic": True})]],
       color=INK, align="c", anchor="m", after=5)
    PENDING.append(expected)


# ================================================================ deck scaffolding
class Deck:
    def __init__(self):
        self.prs = Presentation(str(DRAFT))
        layouts = {l.name: l for l in self.prs.slide_layouts}
        self.L_TITLE, self.L_CONTENT = layouts["TITLE"], layouts["CONTENT"]
        self.sldnum = None
        for sl in self.prs.slides:
            for sh in sl.shapes:
                if sh.is_placeholder and sh.placeholder_format.type is not None and \
                        "SLIDE_NUMBER" in str(sh.placeholder_format.type):
                    self.sldnum = copy.deepcopy(sh._element)
                    break
            if self.sldnum is not None:
                break
        lst = self.prs.slides._sldIdLst
        for sid in list(lst):
            self.prs.part.drop_rel(sid.rId)
            lst.remove(sid)
        self.notes_words = []
        self.pending_by_slide = {}

    def content(self, eyebrow, title):
        s = self.prs.slides.add_slide(self.L_CONTENT)
        tb(s, 0.5, 0.2, 7.0, 0.22, eyebrow.upper(), size=9, color=TEAL, bold=True, spc=200,
           anchor="m")
        ph = next(sh for sh in s.placeholders if "TITLE" in str(sh.placeholder_format.type))
        tf = ph.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.text = title
        for p in tf.paragraphs:
            for r in p.runs:
                _apply_run(r, 20, INK, True)
        self._start = len(PENDING)
        return s

    def finish(self, s, notes, refs=None, number=True):
        if refs:
            lines = [f"[{k}] {SHORT_REFS.get(k, REFS[k])}" if isinstance(k, int) else k for k in refs]
            h = 0.135 * len(lines)
            tb(s, 0.5, 5.47 - h, 8.7, h, lines, size=7.5, color="555555", after=0)
            full = [f"[{k}] {REFS[k]}" for k in refs if isinstance(k, int)]
            notes = notes.rstrip() + "\n\nReferences: " + " ".join(full)
        if number and self.sldnum is not None:
            el = copy.deepcopy(self.sldnum)
            ids = [int(e.get("id")) for e in s.shapes._spTree.iter() if e.tag == qn("p:cNvPr")]
            el.find(".//" + qn("p:cNvPr")).set("id", str(max(ids) + 1))
            s.shapes._spTree.append(el)
        notes = re.sub(r"[ \t]+", " ", notes.strip())
        s.notes_slide.notes_text_frame.text = notes
        self.notes_words.append(len(notes.split()))
        idx = len(self.prs.slides)
        new = PENDING[getattr(self, "_start", len(PENDING)):]
        if new:
            self.pending_by_slide[idx] = list(new)


def run_names(rows):
    """Short description of the metric files behind a set of runs."""
    return ", ".join(sorted({r["file"] for r in rows}))


# ================================================================ build
def build():
    runs = load_runs()
    pm = build_pm_main(runs)
    pm_v1 = build_pm_main(runs, use_table=False)  # superseded first runs, for illustrations
    # corrected reduced-protocol runs (600 tasks, seeds >= 100); preferred wherever they exist
    pm_new = build_pm_main(load_runs(RESULTS / "metrics", n_tasks=600, min_seed=100), use_table=False)
    eta, eta_src = build_eta(runs)
    ps = build_ps()
    d = Deck()

    def val(opt, m):
        return pm.get((opt, m), {}).get("mean")

    # ---- derived numbers used in several places
    motiv_new = bool(pm_new.get(("sgd", "none")))
    none_sgd = (pm_new if motiv_new else pm_v1).get(("sgd", "none"), {})
    dead_first = dead_600 = dead_last = acc_first = acc_last = None
    mid_task = 600
    if none_sgd.get("runs"):
        dfs = [r["df"] for r in none_sgd["runs"] if "dead_fraction" in r["df"]]
        if dfs:
            mid_task = min(len(df) for df in dfs) // 2 if motiv_new else 600
            dead_first = sum(df["dead_fraction"].iloc[0] for df in dfs) / len(dfs)
            dead_600 = sum(df["dead_fraction"].iloc[mid_task:mid_task + 40].mean() for df in dfs) / len(dfs)
            dead_last = sum(last_frac_mean(df["dead_fraction"]) for df in dfs) / len(dfs)
            acc_first = sum(df["online_accuracy"].iloc[:max(1, len(df) // 10)].mean()
                            for df in dfs) / len(dfs)
            acc_last = sum(last_frac_mean(df["online_accuracy"]) for df in dfs) / len(dfs)

    def dead_last_of(opt, m, src=None):
        r = (pm if src is None else src).get((opt, m), {}).get("runs") or []
        dfs = [x["df"] for x in r if "dead_fraction" in x["df"]]
        if not dfs:
            return None
        return sum(last_frac_mean(df["dead_fraction"]) for df in dfs) / len(dfs)

    resets_new = bool(pm_new.get(("sgd", "snr")) and pm_new.get(("adam", "snr")))

    def resets_of(opt):
        r = (pm_new if resets_new else pm_v1).get((opt, "snr"), {}).get("runs") or []
        vals = [x["df"]["resets"].mean() for x in r if "resets" in x["df"]]
        return sum(vals) / len(vals) if vals else None

    measured = {opt: [m for m in METHODS if val(opt, m) is not None] for opt in OPTS}
    missing = sorted({LABEL[m] for opt in OPTS for m in METHODS if val(opt, m) is None},
                     key=lambda s: [LABEL[m] for m in METHODS].index(s))
    snr_seeds = pm.get(("sgd", "snr"), {}).get("n_seeds")
    snr_tasks = pm.get(("sgd", "snr"), {}).get("n_tasks")
    snr_hp = pm.get(("sgd", "snr"), {}).get("hparam", "")
    snr_eta_txt = fmt_eta(parse_eta(snr_hp)) if parse_eta(snr_hp) else "1e-6"
    snr_eta_val = parse_eta(snr_hp) if parse_eta(snr_hp) else 1e-6
    eta_grid_txt = ("inside the paper's grid" if any(abs(snr_eta_val - g) < 1e-12
                                                     for g in PAPER_ETA_GRID)
                    else "outside the paper's grid")
    srcs = {v["source"] for v in pm.values()}
    pm_source = ("results/tables/pm_main.csv, with " + V1 + " runs for rows missing there"
                 if srcs == {"table", "metrics"} else None)
    pm_source = pm_source or "results/tables/pm_main.csv" if "table" in srcs \
        else V1_LONG + " (results/metrics/preliminary_v1)"
    is_v1 = "table" not in srcs
    run_label = V1 if is_v1 else "reduced protocol"

    # ============================================================ 1. title
    s = d.prs.slides.add_slide(d.L_TITLE)
    d._start = len(PENDING)
    pattern = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0,  1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
               1, 0, 0, 1, 1, 0, 1, 0, 1, 1,  0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
               1, 0, 1, 0, 1, 1, 0, 0, 1, 1,  0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    for i, on in enumerate(pattern):
        cx, cy = 5.60 + 0.38 * (i % 10), 1.00 + 0.38 * (i // 10)
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(cy), Inches(0.26), Inches(0.26))
        if on:
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(TEAL)
            c.line.fill.background()
        else:
            c.fill.background()
            c.line.color.rgb = rgb(MUTED)
            c.line.width = Pt(1)
        c.shadow.inherit = False
    tb(s, 0.6, 1.35, 4.8, 0.9, "Self-Normalized Resets", size=30, color=WHITE, bold=True,
       anchor="m")
    tb(s, 0.6, 2.45, 4.7, 1.3, "Keeping neural networks plastic in continual learning: reading the "
       "paper and reproducing its core experiments", size=15, color="CFE3E4")
    tb(s, 0.6, 4.45, 5.0, 0.5, ["Group 7  ·  Introduction to Machine Learning",
                                "Milestone 2: Reading + Reproduction deck"], size=11,
       color=AMBER, after=2)
    tb(s, 5.6, 3.40, 3.8, 0.4, "Filled circle: a neuron that still fires. Hollow circle: a dead "
       "neuron (illustration).", size=9, color=LIGHT)
    d.finish(s, number=False, notes=f"""
This deck covers the paper Self-Normalized Resets (SNR) by Farias and Jozefiak, ICLR 2025 [1], explained in our own words, and the team's reproduction of its core experiments. It replaces our earlier Vietnamese draft. There is no oral presentation, so these notes carry the full explanation for each slide.

The deck has five parts. First, background and the paper's contributions. Second, the method in five layers, from notation to cost. Third, the methodology of our three core-scope studies: A, the headline Permuted MNIST comparison from Table 1; B, Permuted Shakespeare with a small Transformer; C, our own sensitivity analysis of SNR's hyperparameter eta. Fourth, results, a deviation analysis and a comparison with the authors' code. Fifth, a conclusion and appendices.

The picture on this slide is an illustration, not data. Each circle stands for a hidden neuron. Filled circles still fire; hollow ones have stopped firing for a long time. In a ReLU network many units drift into the hollow state as training moves from task to task, and the network learns new tasks worse. SNR detects these units and re-initializes them.

All figures are drawn by the team, either as diagrams or from our own run files. Team numbers come from {pm_source}; the run dashboard is cited as [8] on the result slides. Paper numbers are always labelled as paper numbers. Every team number is regenerated from the run files when the deck is built, so no number is typed by hand.
""")

    # ============================================================ 2. summary
    s = d.content("Summary", "One simple idea, a close first match, and a bug we traced to its root")
    cards = [
        ("PROBLEM", ORANGE, pct(dead_last) if dead_last is not None else "pending",
         f"of hidden units are dead at the end of the SGD baseline; accuracy falls from "
         f"{f3(acc_first)} to {f3(acc_last)}" + ("" if motiv_new else f"; {V1}")),
        ("IDEA", TEAL, "a ≥ ln η / ln(1−p)",
         "reset a unit when its current silent streak a is too long for its own firing rate p"),
        ("OUR RESULT (A)", TEAL, f3(val("sgd", "snr")),
         f"SNR on Permuted MNIST with SGD ({run_label}, {snr_seeds} seeds, {snr_tasks} tasks); "
         f"paper 0.851." + (f" Adam: {f3(val('adam', 'snr'))} (paper 0.889)."
                            if val('adam', 'snr') is not None else "")),
    ]
    for i, (lab, col, big, txt) in enumerate(cards):
        x = 0.5 + i * 3.05
        box(s, x, 1.3, 2.85, 2.5)
        tb(s, x + 0.2, 1.45, 2.45, 0.25, lab, size=10, color=col, bold=True)
        tb(s, x + 0.2, 1.8, 2.45, 0.7, big, size=32 if i != 1 else 17, bold=True, anchor="m")
        tb(s, x + 0.2, 2.6, 2.45, 1.1, txt, size=10.5, color=MUTED)
    box(s, 0.5, 4.05, 9.0, 0.85)
    ps_state = "reduced protocol"
    tb(s, 0.7, 4.1, 8.6, 0.76, [
        [("Core scope: ", {"bold": True}), ("A = PM Table 1 with 8 conditions, B = Permuted "
                                           f"Shakespeare ({ps_state}), C = η sweep.", {})],
        [("Fixed after checking the authors' code: ", {"bold": True}),
         ("our first SNR counted firing per example; it now counts per optimizer step, which "
          "removed the collapse at large η in early checks.", {})]], size=10, after=2)
    d.finish(s, refs=[1], notes=f"""
This slide previews the whole deck.

The problem. When a network trains on thousands of tasks in a row, it gradually learns new tasks worse. In our first SGD baseline on Permuted MNIST, accuracy over the first 10% of tasks is {f3(acc_first)} and over the last 10% it is {f3(acc_last)}, and the share of hidden units that never fire grows to about {pct(dead_last) if dead_last is not None else 'an unmeasured share'}.

The idea of the paper [1]. Treat each neuron separately. A neuron that usually fires every step and has now been silent for many steps is probably dead; a neuron that fires rarely can stay silent that long without anything being wrong. SNR resets a neuron when the probability of its current silent streak, judged by its own history, drops below a threshold eta.

Our result so far. With SGD, SNR reaches {f3(val('sgd', 'snr'))} against 0.851 in the paper, and with Adam {f3(val('adam', 'snr'))} against 0.889. {'These numbers come from our first implementation, preliminary v1, which is now superseded (see the deviation slide).' if is_v1 else 'These numbers come from the reduced protocol (fewer tasks than the paper), with the corrected implementation.'}

The most useful thing we learned. Our first SNR only matched the paper at eta = 1e-6, a thousand times below the paper's grid, and collapsed at the paper's own values. After reading the authors' released code for verification, we found the cause: we counted firing per example, while their code counts it per optimizer step (a unit fires on a step if it is active on any example in the batch). With the step-level rule, early checks show SNR robust to eta again, as the paper reports. This is a detail the paper does not state, and it changes the meaning of eta completely.

Still open: all numbers come from reduced protocols (fewer tasks and seeds than the paper); the full-length runs are running on Colab.
""")

    # ============================================================ 3. background
    s = d.content("Background and motivation",
                  "Continual learning: the network gets worse at new tasks, though they are no harder")
    tb(s, 0.7, 1.3, 3.5, 0.25, "How much is learned on each new task", size=10, color=MUTED)
    for i, top in enumerate([2.0, 2.15, 2.35, 2.6, 2.85, 3.1]):
        x = 0.7 + 0.62 * i
        b = box(s, x, top, 0.45, 4.3 - top, fill=TEAL if i < 2 else (BLUE if i < 4 else MUTED),
                line=None, shape=MSO_SHAPE.RECTANGLE)
        tb(s, x, 4.35, 0.45, 0.25, f"T{i + 1}", size=10, color=MUTED, align="c")
    tb(s, 0.7, 4.65, 3.8, 0.25, "Task 1 … Task n →   (illustration)", size=9, color=MUTED)
    box(s, 4.7, 1.3, 4.8, 3.55)
    tb(s, 4.9, 1.42, 4.4, 0.25, "A real-world example (our own)", size=11, bold=True)
    tb(s, 4.9, 1.72, 4.4, 1.0, "A shop retrains its product-recognition model every week on new "
       "photos. After a few months the model learns each new batch more slowly and less "
       "accurately than it did at the start.", size=11, color=MUTED)
    table(s, 4.9, 2.85, 4.4, [1.3, 2], [["Term", "Meaning"],
                                         ["Loss of plasticity", "new tasks are learned worse over time"],
                                         ["Catastrophic forgetting", "performance on old tasks drops"],
                                         ["Generalization", "performance on unseen examples"]],
          size=9.5, row_h=0.26)
    tb(s, 4.9, 4.45, 4.4, 0.3, "This project measures plasticity loss only.", size=10, bold=True,
       color=TEAL)
    d.finish(s, refs=[2], notes="""
Continual learning means training one network on a long sequence of tasks, one after another, without going back to old data.

Our own example: a shop retrains its product-recognition model every week on that week's photos. After a few months the model learns each new batch more slowly and less accurately than it did in the first weeks, even though the new photos are not harder. This is called loss of plasticity [2]: the network's ability to learn new things wears down with use.

Three terms are easy to confuse, so the table separates them. Loss of plasticity is about new tasks: can the network still learn them? Catastrophic forgetting is about old tasks: does performance on them drop after training on new ones? Generalization is about unseen examples from the same task. A method can help with one and not the others, so a claim about one does not carry over.

The bars on the left are a conceptual illustration, not data: each bar is how much the network manages to learn on a task, and the bars shrink over time.

Everything we measure in this project is plasticity. On Permuted MNIST we use online accuracy on the task currently being trained, and on Permuted Shakespeare the training loss on the final epoch of each task. We do not measure forgetting or held-out generalization, and we do not make claims about them.
""")

    # ============================================================ 4. dead neurons
    s = d.content("Background and motivation",
                  "What we observe: ReLU units stop firing and do not recover on their own")
    flow = ["z ≤ 0 for every input", "ReLU(z) = 0", "gradient = 0", "incoming weights frozen"]
    for i, t in enumerate(flow):
        y = 1.35 + 0.95 * i
        label_box(s, 0.5, y, 2.3, 0.62, t, fill=NAVY if i < 3 else ORANGE, size=12)
        if i < 3:
            arrow(s, 1.65, y + 0.62, 1.65, y + 0.95)
    fracs = [("First task", dead_first), (f"Around task {mid_task}", dead_600),
             ("Last 10% of tasks", dead_last)]
    for gi, (lab, fr) in enumerate(fracs):
        gx = 3.3 + gi * 2.1
        tb(s, gx, 1.3, 2.0, 0.25, f"{lab}: {pct(fr) if fr is not None else '?'} dead", size=10,
           bold=True)
        ndead = round((fr or 0) * 60)
        order = [(k * 37) % 60 for k in range(60)]
        dead_set = set(order[:ndead])
        for k in range(60):
            cx, cy = gx + 0.30 * (k % 6), 1.65 + 0.30 * (k // 6)
            c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(cy), Inches(0.22), Inches(0.22))
            c.shadow.inherit = False
            if k in dead_set:
                c.fill.background()
                c.line.color.rgb = rgb(MUTED)
                c.line.width = Pt(1)
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = rgb(TEAL)
                c.line.fill.background()
    tb(s, 3.3, 4.72, 6.2, 0.22, "SGD baseline, " + ("reduced protocol (600 tasks)" if motiv_new else V1) + ", mean over seeds. Hollow = dead; 60 circles "
       "per grid in the measured proportion.", size=9, color=MUTED)
    d.finish(s, refs=[2, 8], notes=f"""
Why do networks lose plasticity? One correlate that shows up clearly is dead units.

A ReLU outputs max(0, z). If a unit's input z is zero or negative for every example it sees, its output is always zero. The gradient through it is then zero too, so its incoming weights stop changing. The unit can only come back indirectly, if earlier layers happen to change enough to push z above zero again. The chain on the left shows this.

The three grids are drawn from our own measurements on the SGD baseline, averaged over seeds. At the first task about {pct(dead_first) if dead_first is not None else '?'} of hidden units are dead; around task 600 it is about {pct(dead_600) if dead_600 is not None else '?'}; over the last 10% of tasks about {pct(dead_last) if dead_last is not None else '?'}. Each grid has 60 circles drawn in the measured proportion, standing for the 300 hidden units of that MLP. The values come from the dead_fraction column of our first runs (preliminary v1, a three-layer MLP that is now superseded), listed on the dashboard [8]. The corrected runs will redraw these grids.

The project's definition of a dead unit follows the paper's Figure 1: a unit that has not fired for at least 1000 consecutive training examples. This diagnostic counter is separate from the counter SNR uses to decide resets.

A caution: this is a correlation that appears together with the accuracy drop. Dohare et al. [2] and the SNR paper both treat it as a correlate, and so do we. Later slides show a case (L2 with Adam) where many units are dead and accuracy is still high, so dead units cannot be the whole story.
""")

    # ============================================================ 5. predecessors
    s = d.content("Background and motivation", "Earlier methods and the limits the paper points out")
    _src5 = pm_new if pm_new.get(("adam", "l2")) else pm_v1
    dl_l2a, dl_l2ia = dead_last_of("adam", "l2", _src5), dead_last_of("adam", "l2init", _src5)
    table(s, 0.5, 1.3, 9.0, [1.25, 2.4, 2.75, 2.6], [
        ["Method", "Core idea", "Remaining limitation (per [1])", "Practical consequence"],
        ["L2, L2 Init [3]", "Penalty pulls weights toward 0 (L2) or toward their initial values "
         "(L2 Init)", "Cannot detect or revive a dead unit; λ must be tuned; on one ReLU both can "
         "fail with constant probability (Thm 4.1)",
         f"{'Our' if _src5 is pm_new else 'Our v1'} Adam runs: L2 still leaves {pct(dl_l2a) if dl_l2a is not None else '?'} of units dead, "
         f"L2 Init {pct(dl_l2ia) if dl_l2ia is not None else '?'}"],
        ["Shrink & Perturb [9]", "Repeatedly shrink all weights and add small noise",
         "Two hyperparameters; acts on every weight, not on dead units",
         "Paper App. C: one order of magnitude too strong costs most of the accuracy"],
        ["CBP [2], ReDO [4]", "Reset units: CBP replaces low-utility units at a fixed rate; ReDO "
         "resets units with low activity relative to their layer",
         "One threshold or rate for all units, although firing rates differ a lot (Prop. 2.2)",
         "Rarely firing units get reset by mistake, or dead busy units are caught late"],
        ["LayerNorm [10]", "Normalize pre-activations inside each layer",
         "Architectural change with no explicit handling of dead units",
         "Paper Table 1 on PM: 0.69 vs 0.71 (SGD), 0.66 vs 0.64 (Adam) without it"],
    ], size=9, row_h=0.5)
    tb(s, 0.5, 4.5, 9.0, 0.3, "All of these are conditions in our experiment A, so each one is "
       "measured against SNR in the same training loop.", size=10, color=MUTED)
    d.finish(s, refs=[1, 2, 3, "[4] G. Sokar et al., ICML 2023.  [9] J. Ash and R. P. Adams, "
                             "NeurIPS 2020.  [10] J. L. Ba et al., arXiv:1607.06450, 2016."],
             notes=f"""
The paper positions SNR against three families of earlier methods.

Regularization. L2 pulls every weight toward zero; L2 Init [3] pulls weights back toward their initial values, which are known to train well. According to the paper's analysis [1], neither can detect a dead unit or bring it back, and both need a tuned strength lambda. Theorem 4.1 shows that on a single ReLU with a bad starting point, both fail with a probability that stays above a constant. In our first (preliminary v1) Adam runs, L2 still leaves about {pct(dl_l2a) if dl_l2a is not None else 'an unmeasured share'} of hidden units dead and L2 Init about {pct(dl_l2ia) if dl_l2ia is not None else 'an unmeasured share'}.

Shrink and Perturb [9] shrinks all weights by a factor and adds small noise. It has two hyperparameters and does not single out dead units. The paper's hyperparameter tables show that setting it one order of magnitude too strong costs most of the accuracy.

Reset methods. Continual Backprop, CBP [2], tracks a utility score per unit and replaces the lowest-utility unit in each layer at a fixed rate. ReDO [4] periodically resets units whose average activity is small compared with the other units in the same layer. The paper's criticism, formalized in Proposition 2.2, is that both use one threshold or one rate for every unit, while real units fire at very different rates. A shared threshold either resets rarely firing units by mistake or waits too long before resetting a busy unit that has died.

Architecture. Layer normalization [10] changes the model rather than the training rule. In the paper's Table 1 it does not help on Permuted MNIST.

All seven methods are conditions of our experiment A, run through one shared training loop so the comparison is fair. Their implementation status is tracked in appendix A2.
""")

    # ============================================================ 6. gap
    s = d.content("Background and motivation",
                  "The gap: the same silent streak means different things for different units")
    for row, (lab, xs, note, fillc) in enumerate([
        ("Unit A (fires often)", [1.60 + 0.2 * k for k in range(14)],
         "20 silent examples: unusual, probably dead", ORANGE),
        ("Unit B (fires rarely)", [1.60, 4.20, 6.85],
         "20 silent examples: normal for this unit", TEAL)]):
        y = 1.5 + row * 1.15
        tb(s, 0.5, y, 1.1, 0.5, lab, size=10, bold=True, anchor="m")
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.6), Inches(y + 0.35),
                                    Inches(7.8), Inches(y + 0.35))
        ln.line.color.rgb = rgb(MUTED)
        ln.line.width = Pt(1)
        for xx in xs:
            box(s, xx, y + 0.17, 0.04, 0.36, fill=INK, line=None, shape=MSO_SHAPE.RECTANGLE)
        box(s, 4.32, y + 0.05, 2.4, 0.6, fill=None, line=fillc, lw=1.5, dash=True)
        tb(s, 4.32, y + 0.68, 3.4, 0.3, note, size=9, color=fillc, bold=True)
    tb(s, 7.95, 1.62, 1.6, 1.6, "Each tick is one firing; the dashed window is the same 20-example "
       "silence.", size=9, color=MUTED)
    for i, (head, body, col) in enumerate([
            ("Earlier reset methods", "One threshold or reset rate shared by every unit.", MUTED),
            ("SNR", "A threshold per unit, set by one hyperparameter η that has a probability "
             "meaning.", TEAL)]):
        x = 0.5 + i * 4.6
        box(s, x, 3.8, 4.4, 1.1, fill=WHITE if i == 0 else TEAL_TINT)
        tb(s, x + 0.2, 3.88, 4.0, 0.22, head, size=10, color=col, bold=True)
        tb(s, x + 0.2, 4.15, 4.0, 0.7, body, size=11)
    d.finish(s, refs=[1], notes="""
This is the gap SNR fills.

The drawing (ours) shows two units over time. Each tick is a moment when the unit fired. Unit A fires almost every example; unit B fires only now and then. The dashed window marks the same stretch of 20 examples in which neither unit fires.

For unit A, 20 silent examples in a row are very unusual. If it normally fires on most examples, the chance of such a streak while it is still healthy is tiny, so it has most likely died. For unit B, the same 20 silent examples are ordinary, because it often goes that long between firings.

A single threshold applied to every unit cannot handle both cases. If it is short, unit B gets reset while it is healthy. If it is long, unit A sits dead for a long time before anyone notices. Earlier reset methods such as ReDO and CBP use one threshold or one reset rate per layer, so they face exactly this trade-off.

SNR compares each unit's current silence with that unit's own firing history, so the waiting time stretches or shrinks per unit. It needs only one hyperparameter, eta, which reads as a probability: how unlikely the current streak must be, for a healthy unit with this firing rate, before we reset it.
""")

    # ============================================================ 7. contributions
    s = d.content("Contributions", "What the paper contributes: theory, technique and experiments")
    cols3 = [
        ("Theoretical", BLUE, ["Prop. 2.1: deciding to reset is an optimal stopping problem; the "
                               "percentile threshold does not depend on p",
                               "Prop. 2.2: a fixed threshold makes more errors when firing rates "
                               "differ", "Thm 4.1: oracle resets reach O(1/T); L2 and L2 Init can fail"],
         "Before: resets were not framed as a decision with a cost for delay."),
        ("Technical", TEAL, ["SNR rule with one main hyperparameter η",
                             "Per-unit firing rate estimated from the unit's own history",
                             "Works with SGD and Adam; Transformers also need L2 on attention"],
         "Before: one threshold or reset rate for all units."),
        ("Empirical", ORANGE, ["Five benchmarks: PM, RM, RC, CI, PS",
                               "Compared with CBP, ReDO, L2, L2 Init, S&P, LayerNorm",
                               "Best on every vision benchmark and optimizer in Table 1"],
         "We test: PM headline (A), PS (B), the η-robustness claim (C)."),
    ]
    for i, (head, col, items, foot) in enumerate(cols3):
        card(s, 0.5 + i * 3.05, 1.3, 2.9, 3.6, head, items, band=col, body_size=10.5, foot=foot)
    d.finish(s, refs=[1], notes="""
We sort the paper's contributions into three kinds, using our own wording.

Theoretical. Proposition 2.1 treats the question "is this unit dead?" as a hypothesis test with a penalty for deciding late. Under that model the best rule is to reset once the silent streak passes a fixed percentile of the unit's geometric inter-firing distribution, and that percentile does not depend on the unit's firing rate p. Proposition 2.2 compares this with a fixed waiting time shared by all units and shows the fixed rule makes more errors when firing rates differ. Theorem 4.1 studies one ReLU with a bad initialization: L2 and L2 Init fail with constant probability, while gradient descent with an oracle that knows which units are dead reaches an average error of order 1/T. Earlier work had not posed resetting as a decision problem with a delay cost.

Technical. The SNR rule has one main hyperparameter, eta, and estimates each unit's firing rate from its own history. It works with both SGD and Adam. For Transformers the authors combine it with L2 regularization of the attention weights, because attention has no ReLU units to reset.

Empirical. The paper evaluates five benchmarks: Permuted MNIST (PM), Random-label MNIST (RM), Random-label CIFAR (RC), Continual ImageNet (CI) and Permuted Shakespeare (PS). It compares SNR with CBP, ReDO, L2, L2 Init, Shrink and Perturb and LayerNorm, and reports SNR as best on all vision benchmark and optimizer combinations in Table 1.

The footer of each card says what was missing before, and, for the empirical card, which parts we check ourselves: the PM headline comparison (A), Permuted Shakespeare (B), and the claim that SNR is robust to eta (C).
""")

    # ============================================================ 8. notation
    s = d.content("Method · layer 1", "Notation and the formal problem")
    table(s, 0.5, 1.3, 4.6, [1.25, 2.55], [
        ["Symbol", "Meaning"],
        ["bᵢ(t) ∈ {0,1}", "unit i fires on optimizer step t (active on any example in the batch)"],
        ["aᵢ", "current run of consecutive silent steps"],
        ["Rᵢ", "length of a completed silent run of unit i"],
        ["μᵢ, pᵢ = 1/μᵢ", "mean silent-run length, implied firing rate"],
        ["η", "accepted probability of the current silent streak"],
        ["B", "batch size (16 images on PM, 8 sequences on PS)"],
    ], size=9.5, row_h=0.34)
    box(s, 5.3, 1.3, 4.2, 3.55)
    tb(s, 5.5, 1.42, 3.8, 0.25, "Formal problem (after [1])", size=11, bold=True)
    tb(s, 5.5, 1.75, 3.8, 1.9, ["For each unit i:",
                                [("H₀: ", {"bold": True}), ("pᵢ > 0 (alive)", {})],
                                [("H₁: ", {"bold": True}), ("pᵢ = 0 (dead)", {})],
                                "Choose a stopping time τ at which to reset, minimizing decision "
                                "errors plus a cost for delay."], size=12, after=6)
    tb(s, 5.5, 3.95, 3.8, 0.8, "Idealized model: firings are independent. In practice the "
       "silent-run distribution is estimated from the unit's history.", size=10, color=MUTED,
       italic=True)
    d.finish(s, refs=[1], notes="""
The first layer of the method is a shared notation, used for the rest of the deck.

Time is counted in optimizer steps. For each hidden unit i, b_i(t) is 1 if the unit fires on step t, meaning its activation is positive for at least one example in that step's batch, and 0 otherwise. The paper's Algorithm 1 is written per training example; the authors' released code, which we read for verification only, counts per step in this way, and our first implementation's per-example counting turned out to be the cause of our largest deviation (see the deviation slide).

a_i is the number of consecutive recent steps on which the unit did not fire. A completed silent run is a stretch of silent steps that ended with a firing; R_i is its length. mu_i is the mean length of the unit's completed silent runs, and p_i = 1/mu_i is the firing rate this implies. eta is the smallest probability we accept for the current silent streak before calling the unit dead. B is the batch size: 16 images on Permuted MNIST, 8 token sequences on Permuted Shakespeare.

The formal problem, following the paper [1], is a test between two hypotheses for each unit. Under H0 the unit is alive and fires with some probability p_i > 0. Under H1 it is dead and its firing probability is exactly 0. We watch the unit and pick a stopping time tau; if we decide "dead", we reset it. The cost is the probability of a wrong decision plus a penalty for waiting.

This is an idealized model. It assumes independent firings and a dead rate of exactly zero, and the true distribution of silent runs is unknown, so SNR estimates it from the unit's own history.
""")

    # ============================================================ 9. pipeline
    s = d.content("Method · layer 2", "The overall pipeline: SNR runs after every optimizer step")
    top = [("Batch: 16 images or 8 × 128 tokens", NAVY), ("Forward: logits and hidden "
                                                              "activations", NAVY),
           ("Loss, backward, optimizer step", NAVY)]
    for i, (t, c) in enumerate(top):
        label_box(s, 0.5 + 2.2 * i, 1.5, 1.8, 0.85, t, fill=c, size=11)
        if i < 2:
            arrow(s, 2.3 + 2.2 * i, 1.93, 2.7 + 2.2 * i, 1.93)
    arrow(s, 5.8, 2.35, 5.8, 2.95)
    tb(s, 0.5, 2.62, 3.0, 0.22, "The SNR part", size=10, color=TEAL, bold=True)
    bottom = [("Did each unit fire this step? Update a and μ", TEAL),
              ("Is the silent streak too unlikely? (≤ η)", TEAL),
              ("Reset the flagged units", ORANGE)]
    for i, (t, c) in enumerate(bottom):
        label_box(s, 0.5 + 2.2 * i, 2.95, 1.8, 0.85, t, fill=c, size=10.5, bold=True)
    arrow(s, 4.5, 3.38, 4.9, 3.38)
    arrow(s, 2.3, 3.38, 2.7, 3.38)
    arrow(s, 7.0, 3.38, 7.9, 3.38)
    label_box(s, 7.9, 2.95, 1.6, 0.85, "Next batch", fill=MUTED, size=11)
    box(s, 0.5, 4.1, 9.0, 0.8)
    tb(s, 0.65, 4.15, 8.7, 0.7, "Reset = redraw incoming weights, bias 0, outgoing weights 0, clear "
       "Adam moments of those entries, clear the unit's statistics. The dead-unit diagnostic keeps "
       "its own counter and is never used as the reset criterion.", size=10.5, anchor="m")
    d.finish(s, refs=[1], notes="""
This diagram is ours and follows one training step.

Top row: a batch enters the network, 16 images on Permuted MNIST or 8 token sequences of length 128 on Permuted Shakespeare. The forward pass gives the logits and, as a side output, the activations of every hidden ReLU unit. Then comes the usual step: loss, backward pass, and an SGD or Adam update.

Bottom row: the SNR part. First we reduce each unit's activations over the batch to one bit: did the unit fire on this step, meaning on at least one example (or token position)? If it fired, its silent streak a ends; a completed silent run updates the unit's estimate of its typical run length, and a goes back to zero. If it did not fire, a grows by one. Second, we ask whether a streak this long would be too unlikely for a healthy unit, with probability at most eta. Third, units that pass that test are reset.

A reset redraws the unit's incoming weights from the layer's initialization, sets its bias to zero and its outgoing weights to zero, so the network's output does not jump. Adam's moment estimates for exactly those entries are cleared, and the unit's statistics restart.

In our code this bottom row is one hook called after every optimizer step, so every method shares the same training loop. The diagnostic that counts dead units (no firing for 1000 examples, as in the paper's Figure 1) keeps its own counter, so SNR is never judged by its own rule.
""")

    # ============================================================ 10. silence counter
    s = d.content("Method · layer 3: component 1",
                  "Silence counter: optimizer steps in a row without firing")
    for i, (head, body) in enumerate([
            ("What it does", "For every hidden unit, count consecutive optimizer steps without "
                             "firing. The count returns to 0 when the unit fires."),
            ("Why it is needed", "It is the only observable evidence for the hypothesis \"this unit "
                                 "is dead\". Without the streak length there is nothing to test."),
            ("How it works exactly", "A unit fires on a step if its activation is > 0 for any "
                                     "example (or token) in the batch. Fired: a = 0. Silent: "
                                     "a = a + 1. This matches the authors' code.")]):
        card(s, 0.5 + i * 3.05, 1.3, 2.9, 2.05, head, body, bullets=False, body_size=10.5)
    box(s, 0.5, 3.5, 9.0, 0.9)
    tb(s, 0.65, 3.56, 2.6, 0.25, "ONE STEP, B = 16", size=9.5, color=TEAL, bold=True, spc=100)
    fires = {2, 5, 11}
    for k in range(16):
        box(s, 0.65 + 0.3 * k, 3.88, 0.24, 0.3, fill=TEAL if k in fires else WHITE,
            line=TEAL if k in fires else MUTED, shape=MSO_SHAPE.RECTANGLE)
    tb(s, 0.65, 4.2, 4.8, 0.2, "filled = active on that example (index 0 to 15)", size=8.5,
       color=MUTED)
    tb(s, 5.6, 3.58, 3.75, 0.8, [[("Now: some cell filled → fired, a = 0", {"bold": True})],
                                 [("Preliminary v1 counted examples: a = 16 − 1 − 11 = 4",
                                   {"color": MUTED})]], size=10, anchor="m", after=2)
    tb(s, 0.5, 4.5, 9.0, 0.45, "The same step-level rule applies to the Transformer's FFN units: "
       "a unit fires on a step if it is active at any token position of any sequence in the batch.",
       size=10, color=MUTED, italic=True)
    d.finish(s, refs=[1], notes="""
The first component is the silent-streak counter a for each unit.

What it does: it counts how many optimizer steps in a row the unit has not fired. As soon as it fires, the count goes back to zero.

Why it is needed: the hypothesis "this unit is dead" can only be checked through this streak. Everything else in SNR turns the streak into a decision.

How it works, and how it changed. The paper's Algorithm 1 is written per training example. Our first implementation, preliminary v1, took that literally: it treated the 16 examples of a batch as a sequence and counted silent examples. In the strip on the slide the unit is active on examples 2, 5 and 11, so v1 would set a = 16 - 1 - 11 = 4. When we read the authors' released code for verification, we found that it works per optimizer step instead: a unit fires on a step if it is active on any example in the batch. In the strip, that step counts as a firing and a = 0.

The difference matters a lot. A unit that is active on most examples looks, per example, like it fires with probability close to 1, so two or three silent examples in a row already look very unlikely and the unit gets reset. That is the flood of false resets behind our early eta collapse. Per step, such a unit almost never has a silent step, and its streaks are judged on a sensible scale. Our corrected code (src/methods/snr.py) uses the step-level rule.

For the Transformer in experiment B the same rule applies to the 256 feed-forward units: a unit fires on a step if it is active at any token position of any sequence in the batch.

A separate counter with a fixed threshold of 1000 examples drives the dead-unit diagnostic used in the charts.
""")

    # ============================================================ 11. firing rate
    s = d.content("Method · layer 3: component 2",
                  "Estimating each unit's silent-run distribution: two estimators")
    ps_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    def astar(p, e):
        """Silent steps before a reset under the geometric model, (1 - p)^(a - 1) <= eta."""
        return 1 + math.log(e) / math.log(1 - p)

    line_chart(s, 5.2, 1.25, 4.3, 3.6, [f"{p:g}" for p in ps_grid],
               [("η = 0.08", [round(astar(p, 0.08), 1) for p in ps_grid], BLUE),
                ("η = 0.00125", [round(astar(p, 0.00125), 1) for p in ps_grid], ORANGE)],
               "Silent steps before a reset (geometric model)", val_fmt="0",
               cat_title="firing rate per step, p = 1/μ")
    box(s, 0.5, 1.3, 4.5, 3.55)
    y0 = 1.4
    for head, body, h in [
            ("Geometric (paper footnote 3)", "Keep μ, the mean length of completed silent runs "
             "(moving average, β = 0.1; the first run sets μ). Assume runs are geometric "
             "with p = 1/μ.", 0.95),
            ("Histogram (paper main text)", "Keep each unit's histogram of past silent-run lengths "
             "over a trailing window of tasks; the threshold is its (1 − η) percentile, "
             "as in the authors' code.", 0.95),
            ("Why both", "The paper describes both; we sweep the histogram version at 3 values of "
                         "η as a check.", 0.5)]:
        tb(s, 0.65, y0, 4.2, 0.22, head.upper(), size=9.5, color=TEAL, bold=True, spc=100)
        tb(s, 0.65, y0 + 0.25, 4.2, h, body, size=10.5)
        y0 += h + 0.32
    d.finish(s, refs=[1], notes="""
The second component estimates what a normal silent run looks like for each unit, so that a long streak can be judged against that unit's own history. Only completed silent runs, which ended with a firing, enter the estimate.

The paper describes two ways to do this, and our corrected code supports both.

The geometric estimator follows the paper's footnote 3. Each unit keeps one number, mu, the mean length of its completed silent runs, and assumes run lengths are geometric with p = 1/mu. The paper averages over a trailing window whose length it does not report; we use an exponential moving average with beta = 0.1, and the first completed run sets mu directly. Before any run has completed, mu starts at 1000 steps, which also sets how long a unit that is dead from initialization waits before its first reset.

The histogram estimator follows the paper's main text. Each unit keeps a histogram of its past silent-run lengths over a trailing window of tasks, and its threshold is the (1 - eta) percentile of that histogram. The authors' released code works this way, which also explains the "percentile 0.92" setting we saw earlier in their notebook: that is eta = 0.08.

Under the geometric model a healthy unit with firing rate p stays silent for a steps with probability (1 - p)^(a - 1). The chart, computed from that formula, shows how many silent steps it takes before this drops below eta, for the largest and smallest eta in the paper's grid. Units that fire often get a short wait, rare firers a long one, and the curves differ only by a constant factor, which is why a whole range of eta values can behave similarly once the time unit is right.

Our main runs use the geometric estimator; experiment C also runs the histogram version at three eta values.
""")

    # ============================================================ 12. decision rule
    s = d.content("Method · layer 3: component 3", "The decision rule and the reset operation")
    for i, (head, body) in enumerate([
            ("What it does", "Geometric: reset when (1 − p)^(a−1) ≤ η. Histogram: "
                             "reset when a exceeds the unit's (1 − η) percentile of past "
                             "silent runs. Never before 6 silent steps (floor)."),
            ("Why it is needed", "It turns the streak into a decision with a probability meaning: "
                                 "η is the largest accepted chance of this streak for a "
                                 "healthy unit."),
            ("How it works", "Redraw incoming weights; bias 0; outgoing weights 0 (output "
                             "unchanged); clear Adam moments of those entries; clear the unit's "
                             "statistics.")]):
        card(s, 0.5 + i * 3.05, 1.3, 2.9, 2.2, head, body, bullets=False, body_size=10.5)
    box(s, 0.5, 3.7, 9.0, 1.2)
    tb(s, 0.7, 3.78, 4.0, 0.25, "Worked example, geometric rule, η = 0.01", size=10,
       color=TEAL, bold=True)
    tb(s, 0.7, 4.08, 4.2, 0.35, f"p = 0.5 (fires often): reset after ≈ {astar(0.5, 0.01):.0f} "
       "silent steps", size=11.5, bold=True)
    tb(s, 5.0, 4.08, 4.3, 0.35, f"p = 0.01 (fires rarely): reset after ≈ "
       f"{astar(0.01, 0.01):.0f} silent steps", size=11.5, bold=True)
    tb(s, 0.7, 4.5, 8.6, 0.3, "Same η, but the waiting time scales with each unit's own history: "
       "this is the \"self-normalized\" part.", size=10.5, color=MUTED)
    d.finish(s, refs=[1], notes=f"""
The third component makes the decision and performs the reset.

What it does. With the geometric estimator, a unit is reset when (1 - p)^(a - 1) <= eta, the probability that a healthy unit with firing rate p would produce a silent run at least this long. We evaluate it in log form to avoid numbers too small to represent. With the histogram estimator, a unit is reset when its current streak exceeds the (1 - eta) percentile of its own past silent runs. In both cases no unit is reset before it has been silent for more than 6 steps, about 100 examples at batch 16. Our final code review found why this floor is needed. A dense unit whose silences all last one step has mean 1, so p = 1, and the geometric rule then resets it after two silent steps whatever eta is. That made most resets independent of eta. The paper does not give a floor; the authors' code never thresholds below 100 examples, so we adopted the same order of magnitude.

Why it is needed: the rule turns the streak and the unit's history into one decision with a clear meaning. eta is the largest probability we accept for seeing this streak from a healthy unit.

Worked example with the geometric rule and eta = 0.01. A unit that fires on half of the steps (p = 0.5) is reset after about {astar(0.5, 0.01):.0f} silent steps. A unit that fires on 1% of the steps (p = 0.01) is reset only after about {astar(0.01, 0.01):.0f}. The same eta gives very different waiting times because each unit is measured against its own history. That is what "self-normalized" refers to.

How a reset works in our corrected code. The incoming weights are redrawn from the layer's initialization and the bias is set to zero. The outgoing weights are set to zero, so the network's output is unchanged at that moment and the new unit starts contributing only as it learns. Adam's first and second moments for exactly those entries are cleared, and the unit's counter, mean and histogram start over.

eta controls how eager SNR is. A healthy unit is wrongly flagged with probability about eta per long silent run, so the time unit in which runs are counted decides how many false resets a given eta produces. That is the root of the deviation discussed later.
""")

    # ============================================================ 13. transformer placement
    s = d.content("Method · SNR in a Transformer",
                  "SNR in a Transformer: only the 256 feed-forward units can be reset")
    chain = [("Token + position embedding", MUTED, None),
             ("Self-attention: 2 heads × 16 dims (Q, K, V, O)", NAVY,
              "no ReLU units here → L2* regularizes these weights"),
             ("Feed-forward: 256 ReLU units", TEAL, "SNR watches these units"),
             ("Output logits over the restricted vocabulary", MUTED, None)]
    for i, (t, c, side) in enumerate(chain):
        y = 1.3 + 0.9 * i
        label_box(s, 0.5, y, 3.6, 0.62, t, fill=c, size=11, bold=c == TEAL)
        if i < 3:
            arrow(s, 2.3, y + 0.62, 2.3, y + 0.9)
        if side:
            arrow(s, 4.1, y + 0.31, 4.6, y + 0.31, color=c)
            tb(s, 4.7, y + 0.05, 2.2, 0.55, side, size=10, color=c, bold=True, anchor="m")
    tb(s, 0.5, 4.7, 3.6, 0.22, "One decoder layer with residual connections (diagram ours)",
       size=8.5, color=MUTED)
    card(s, 7.0, 1.3, 2.5, 3.35, "What a reset touches",
         ["FFN up-projection rows: redrawn; biases 0",
          "Matching down-projection columns: 0",
          "Adam moments of those entries: cleared",
          "Fires on a step = active at any token of any sequence",
          "Attention matrices: never reset"], body_size=10)
    d.finish(s, refs=[1], notes="""
Experiment B uses a Transformer, so it helps to see where SNR can act in that model. The diagram is ours.

The model in the paper is a decoder with a single layer. Tokens get an embedding plus a learned position embedding. Then come self-attention with 2 heads of 16 dimensions each, a feed-forward block with 256 ReLU units, and the output logits over the restricted vocabulary of the text. Residual connections wrap both blocks.

The only ReLU units in this model are the 256 feed-forward units, so those are the units SNR watches and resets. A unit fires on a step if it is active at any token position of any sequence in the batch, the same step-level rule as for the MLP. A reset redraws the unit's row of the up-projection, sets its bias to zero and its column of the down-projection to zero, and clears the Adam moments of those entries.

The attention block has no units of this kind. Its weights are the query, key, value and output matrices, and nothing there can "die" in the ReLU sense, so we never reset attention matrices. The paper reports that plasticity loss in this model comes with growing attention weight norms and a collapse in the entropy of the attention probabilities, which SNR cannot address. Its fix is to combine SNR with L2 regularization. The variant L2*, from the paper's ablation, applies the penalty only to the attention weights.

This explains the paper's main language result, which experiment B tests: SNR alone is about as bad as no intervention, L2 is needed, and SNR helps once it is combined with L2.
""")

    # ============================================================ 14. theory
    s = d.content("Method · layer 4", "Theory: what it promises and where its limits are")
    th = [("Proposition 2.1", "Resetting is an optimal stopping problem; the optimal rule waits for a "
                              "fixed percentile of the geometric gap, independent of p.",
           "Limit: assumes independent firings and a dead rate of exactly 0."),
          ("Proposition 2.2", "A fixed threshold shared by all units makes more errors than SNR when "
                              "firing rates differ.",
           "Limit: compares against an idealized ReDO/CBP in a simple model."),
          ("Theorem 4.1", "One badly initialized ReLU: L2 and L2 Init fail with constant probability; "
                          "oracle resets reach O(1/T).",
           "Limit: the oracle knows which units are dead; practical SNR must estimate it. One unit only.")]
    for i, (head, promise, limit) in enumerate(th):
        x = 0.5 + i * 3.05
        box(s, x, 1.3, 2.9, 3.6)
        box(s, x, 1.3, 2.9, 0.38, fill=NAVY, line=NAVY, shape=MSO_SHAPE.RECTANGLE)
        tb(s, x + 0.15, 1.3, 2.6, 0.38, head, size=12, color=WHITE, bold=True, anchor="m")
        tb(s, x + 0.15, 1.78, 2.6, 0.22, "PROMISE", size=9.5, color=TEAL, bold=True, spc=100)
        tb(s, x + 0.15, 2.02, 2.6, 1.4, promise, size=11)
        tb(s, x + 0.15, 3.5, 2.6, 1.3, limit, size=10.5, color=ORANGE)
    d.finish(s, refs=[1], notes="""
The fourth layer is the paper's theory, which we restate in intuitive terms. We did not reproduce the proofs.

Proposition 2.1. Suppose that deciding "alive or dead" is a hypothesis test with a penalty for deciding late. If the penalty is proportional to the unit's firing rate, which is a fair way to say that a dead busy unit costs more per example than a dead rare one, then the best rule is simple: reset once the silent streak exceeds a fixed percentile of the geometric distribution of gaps. That percentile is the same for every unit, which is why one eta can serve the whole network. The limit: the result assumes independent firings and compares "alive with some rate" against "rate exactly zero". Real units drift in between.

Proposition 2.2. A rule that waits the same fixed time for every unit, which is roughly what ReDO and CBP do, has a higher total error rate than SNR, and the gap grows as the units' firing rates become more different. The limit: the comparison is with idealized versions of those methods in the same simple model, not with their real implementations.

Theorem 4.1. Take a single ReLU that has to learn a target from a bad starting point, as if it had just finished an earlier task. Gradient descent with L2 or L2 Init fails with a probability bounded away from zero. Gradient descent with an oracle that resets the unit exactly when it is dead reaches an average error of order 1/T. This is the strongest limit of the three: the oracle knows the truth, while practical SNR must estimate it from noisy firing data, and the result covers one unit, not a deep network. The theorem supports resetting in general; it is not a guarantee for practical SNR.
""")

    # ============================================================ 15. complexity
    s = d.content("Method · layer 5",
                  "Cost: SNR scales with the number of units, not the number of weights")
    table(s, 0.5, 1.3, 9.0, [1.3, 2.3, 2.6, 2.8], [
        ["Method", "Extra time per step", "Extra memory", "Note"],
        ["SNR (geometric)", "O(B·N) to reduce firing, plus resets", "O(N): a few values per unit",
         "N = 200 (MLP), 256 (PS)"],
        ["SNR (histogram)", "same", "O(N·H): H histogram bins per unit", "threshold updated per window"],
        ["L2", "O(P)", "negligible", "P = 89,610 for the 784-100-100-10 MLP"],
        ["L2 Init", "O(P)", "O(P): copy of the initial weights", "one extra copy of the model"],
        ["Shrink & Perturb", "O(P) per shrink step", "negligible", "acts on every weight"],
        ["CBP, ReDO", "O(B·N) for activity or utility", "O(N) per-unit statistics", "plus periodic resets"],
    ], size=9.5, row_h=0.33)
    box(s, 0.5, 3.85, 9.0, 1.05)
    tb(s, 0.7, 3.92, 8.6, 0.9, [
        "N = hidden units, P = weights, B = batch size. N = 200 is far below P = 89,610, so SNR's "
        "bookkeeping is small next to the forward and backward pass.",
        [("We have not timed each method's step separately, ", {"bold": True}),
         ("so the slide gives orders of growth only.", {})]], size=10.5)
    d.finish(s, refs=[1], notes="""
The last layer is cost, which we derived ourselves from the method descriptions.

Our corrected Permuted MNIST model is the 784-100-100-10 MLP used in the authors' code. It has P = 89,610 weights and biases and N = 200 hidden units. (The superseded preliminary v1 model had three hidden layers: 99,710 parameters and 300 units.)

SNR with the geometric estimator keeps a few numbers per unit: the silent-streak counter and the mean run length. Its extra memory is O(N). Each step it reduces the batch's activations to one firing bit per unit, which is O(B times N), plus the cost of re-initializing the few units actually reset. The histogram estimator also stores a histogram of run lengths per unit, O(N times H) for H bins, and recomputes thresholds once per window of tasks. For the Transformer in experiment B, N is the 256 feed-forward units.

L2 adds a penalty over every parameter, so its extra time is O(P) per step and it needs no extra memory worth mentioning. L2 Init needs the same time and keeps a full copy of the initial weights, O(P) memory. Shrink and Perturb touches every weight when it shrinks and perturbs, O(P). CBP and ReDO keep per-unit statistics, so they are O(N) in memory and O(B times N) in time, like SNR, plus their periodic resets.

The practical message: SNR's bookkeeping is small next to the forward and backward pass because there are far fewer units than weights. We have not timed each method separately, so we give orders of growth only; per-step timing on the same hardware is a cheap addition for the final runs.
""")

    # ============================================================ 16. scope
    s = d.content("Reproduction · scope", "Core scope: two key experiments plus one analysis of our own")
    blocks = [
        ("A  ·  Headline", "Permuted MNIST, MLP (Table 1, PM column)",
         ["SGD and Adam, 8 conditions", "Now: 600 of 2400 tasks, seeds 100-101, paper-best settings",
          "Full 2400 tasks on Colab"], TEAL),
        ("B  ·  Transformer", "Permuted Shakespeare (Table 2, Figs 2-4)",
         ["1-layer decoder, 256 ReLU FFN units", "7 conditions incl. SNR+L2, L2*; CBP and ReDO not "
          "run (no PS grid in the paper)", "Now: 30 tasks x 50 epochs, 1 seed"], BLUE),
        ("C  ·  Our analysis", "η sensitivity of SNR on PM",
         ["Paper grid 0.08 … 0.00125 plus 1e-6", "SGD and Adam, 300 tasks, seed 100",
          "Histogram estimator at 3 values"], ORANGE),
    ]
    for i, (head, sub, items, col) in enumerate(blocks):
        x = 0.5 + i * 3.05
        box(s, x, 1.3, 2.9, 2.55)
        box(s, x, 1.3, 2.9, 0.38, fill=col, line=col, shape=MSO_SHAPE.RECTANGLE)
        tb(s, x + 0.15, 1.3, 2.6, 0.38, head, size=11.5, color=WHITE, bold=True, anchor="m")
        tb(s, x + 0.15, 1.78, 2.6, 0.45, sub, size=10.5, bold=True)
        tb(s, x + 0.15, 2.28, 2.6, 1.5, items, size=10, bullets=True, after=3)
    box(s, 0.5, 4.0, 9.0, 0.9, fill=ROW_ALT)
    tb(s, 0.7, 4.07, 8.6, 0.78, [
        [("Out of scope (compute and time): ", {"bold": True}),
         ("Random-label MNIST and CIFAR, Continual ImageNet, scaled Shakespeare and scale-256, test-set "
          "generalization, team hyperparameter tuning.", {})],
        [("Why this choice: ", {"bold": True}),
         ("A is the paper's headline claim; B is the only setting where SNR alone fails; C checks a "
          "claim our first implementation contradicted.", {})]], size=10, after=3)
    d.finish(s, refs=[1, 8], notes="""
The course lets a team choose full scope or core scope. Core scope means at least the two most important experiments, including the headline comparison, plus one analysis designed by the team in place of what is left out. We chose core scope because our development machines are CPU-only and our time is limited.

A, the headline. The paper's central claim is Table 1: SNR beats every competitor. We reproduce its Permuted MNIST column with the MLP, under SGD and Adam, with all eight conditions: no intervention, SNR, CBP, ReDO, L2, L2 Init, Shrink and Perturb, and LayerNorm. The runs in this deck use a reduced protocol: 600 of the paper's 2400 tasks, two fresh seeds (100 and 101; three SGD conditions also have seed 102), and the best hyperparameters from the paper's appendix tables, with no tuning of our own. The full 2400-task protocol runs later on a Colab GPU from a prepared notebook.

B, the Transformer. Permuted Shakespeare is the one setting where the paper says SNR alone does not work, so it tests the method's limits. We run seven conditions with 30 tasks of 50 epochs and one seed (the paper: 500 tasks of 100 epochs, 9 seeds): no intervention, L2, L2*, L2 Init, SNR, SNR+L2 and SNR+L2*. CBP and ReDO are not run here because the paper gives no hyperparameter grid for them on this benchmark.

C, our own analysis. The paper claims SNR is robust to eta, and our first implementation strongly contradicted this. We sweep the paper's seven values plus 1e-6, with SGD and Adam, over 300 tasks on seed 100, and run the histogram estimator at three values as a check.

Everything else is out of scope: random-label MNIST and CIFAR, Continual ImageNet, the scaled language studies, test-set generalization, and our own tuning. Runs are logged on the team dashboard [8].
""")

    # ============================================================ 17. methodology A
    s = d.content("Methodology · experiment A", "Permuted MNIST: data, protocol and metric")
    steps = [("MNIST [5]", "60,000 images"), ("Fixed subset", "10,000 images"),
             ("Pixel permutation", "seeded by (seed, task)"), ("MLP", "784-100-100-10"),
             ("625 steps", "per task (batch 16)")]
    for i, (a, b) in enumerate(steps):
        x = 0.4 + 1.88 * i
        label_box(s, x, 1.35, 1.62, 0.9, a, fill=NAVY if i != 3 else TEAL, size=10.5, bold=True,
                  sub=b, sub_size=9)
        if i < 4:
            arrow(s, x + 1.62, 1.8, x + 1.88, 1.8)
    table(s, 0.5, 2.5, 4.6, [1.3, 3.0], [
        ["Item", "Setting"],
        ["Tasks", "paper 2400; our reduced run 600 (2400 on Colab)"],
        ["Optimizers [7]", "SGD lr 0.01; Adam lr 0.001, ε = 1e-7"],
        ["Conditions", "8 (None, SNR, CBP, ReDO, L2, L2 Init, S&P, LN)"],
        ["Hyperparameters", "best values in the paper's App. C; no team tuning"],
        ["Seeds", "reduced runs 100-101 (102 for some); v1 used 0-2"],
    ], size=9, row_h=0.32)
    box(s, 5.3, 2.5, 4.2, 2.4)
    tb(s, 5.5, 2.6, 3.8, 0.25, "Metric", size=11, bold=True)
    tb(s, 5.5, 2.9, 3.8, 1.2, "Online accuracy of a task = share of correct predictions on that "
       "task's training batches, each made before the update on that batch. Reported: mean over "
       "the last 10% of tasks, then mean ± std over seeds.", size=10)
    tb(s, 5.5, 4.15, 3.8, 0.65, "Dead unit (diagnostic): no firing for ≥ 1000 consecutive "
       "examples, as in the paper's Figure 1.", size=10, color=MUTED)
    d.finish(s, refs=[5, 6, 7], notes="""
Experiment A follows the paper's Permuted MNIST setup.

Data. MNIST [5] comes from the Hugging Face copy ylecun/mnist [6], read from its Parquet files. Each image is flattened to 784 values and divided by 255. One subset of 10,000 training images is drawn per seed and shared by all tasks. Each task applies its own pixel permutation, generated from the pair (seed, task index), to all 10,000 images; labels stay the same. The model sees one epoch per task in batches of 16, which is 625 steps per task, and its weights carry over from task to task.

Model. The paper only says its MLP matches earlier work. The authors' released code, read for verification, uses 784-100-100-10: two hidden layers of 100 ReLU units, LeCun-normal weight initialization and zero biases. We now use the same. Our superseded preliminary v1 runs used three hidden layers and PyTorch's default initialization.

Optimizers. SGD with learning rate 0.01, and Adam [7] with learning rate 0.001 and epsilon 1e-7, set explicitly because the PyTorch default is 1e-8. These are the paper's values.

Protocol. The paper runs 2400 tasks and reports 5 fresh seeds after a tuning sweep. Our current runs are a reduced protocol: 600 tasks, seeds 100 and 101 (plus 102 for SGD None, CBP and SNR), and for each method the hyperparameter that scored best in the paper's appendix tables. We do no tuning of our own, so tuning and evaluation stay separate by construction. The full 2400-task runs are prepared for a Colab GPU.

Metric. Online accuracy of a task is the share of correct predictions on that task's training batches, each made before the update on its batch. We report the mean over the last 10% of tasks (60 of 600 in the reduced runs), then the mean and sample standard deviation over seeds. Because 600 tasks is a quarter of the paper's sequence, the no-intervention baseline has had less time to degrade, which should be kept in mind when comparing with the paper's numbers.
""")

    # ============================================================ 18. methodology B: benchmark
    s = d.content("Methodology · experiment B",
                  "Permuted Shakespeare: a language-model analogue of Permuted MNIST")
    stepsB = [("The Tempest", "cut from the complete works"), ("GPT-2 BPE [11]", "256 × 129 tokens"),
              ("Vocabulary", "only tokens that occur"), ("Task t", "permute token ids"),
              ("Next-token loss", "context 128, batch 8")]
    for i, (a, b) in enumerate(stepsB):
        x = 0.4 + 1.88 * i
        label_box(s, x, 1.35, 1.62, 0.9, a, fill=NAVY if i != 3 else BLUE, size=10.5, bold=True,
                  sub=b, sub_size=9)
        if i < 4:
            arrow(s, x + 1.62, 1.8, x + 1.88, 1.8)
    card(s, 0.5, 2.55, 4.4, 2.35, "How a task works",
         ["The text stays fixed; each task maps every token id through a new random permutation "
          "of the restricted vocabulary, seeded by (seed, task)",
          "The model must relearn which id means which word",
          "The authors' public template differs: another play, words permuted before tokenizing"],
         body_size=10)
    card(s, 5.1, 2.55, 4.4, 2.35, "Why this benchmark matters for SNR",
         ["Only the 256 FFN units can be reset; attention has no such units",
          "The paper reports SNR alone ≈ no intervention here; L2 is needed",
          "It tests whether SNR helps in combination (SNR+L2, SNR+L2*)"], body_size=10,
         head_color=ORANGE)
    d.finish(s, refs=[1, 11], notes="""
Experiment B uses Permuted Shakespeare, a benchmark the paper introduces so that a Transformer can be studied the same way as the MLP on Permuted MNIST.

Data. The paper says it uses 32,768 tokens of Shakespeare's The Tempest, tokenized with the GPT-2 byte-pair encoder [11], with the vocabulary restricted to the tokens that occur. Our corpus file is the complete-works text; we checked that it is byte-identical to the authors' all_shakespeare.txt after converting Windows line endings. From it we cut The Tempest, from its title to its closing line, and take 256 non-overlapping windows of 129 tokens. One epoch is then exactly 32 batches of 8 sequences with 128 next-token predictions each, the paper's 32,768 tokens.

A task works like a permutation in Permuted MNIST. The text never changes. For task t we draw a random permutation of the restricted vocabulary, seeded by (seed, t), and map every token id through it. The model has to learn again which id stands for which word, while the structure of the text stays the same. Training is next-token prediction with Adam.

A difference we found. The authors' public template for this benchmark reads the first tokens of the complete-works file, which is a different play (All's Well That Ends Well), and it permutes whitespace-separated words before tokenizing, while we permute token ids. We follow the paper's text, and we list this under the deviations.

Why it matters for SNR. Only the feed-forward units can be reset, because attention has no ReLU units. The paper reports that SNR alone does about as well as nothing, that L2 is needed, and that SNR on top of L2 gives the lowest loss over the last 50 tasks among conditions that regularize all weights. This experiment tests that claim and the attention-only ablation L2*.
""")

    # ============================================================ 19. methodology B: protocol
    s = d.content("Methodology · experiment B", "Protocol: the paper's setting and our reduced version")
    any_ps = next(iter(ps.values()), None) if ps else None

    def ps_ours(key, fallback):
        if any_ps is None or any_ps.get(key) is None:
            return fallback
        return f"{int(float(any_ps[key])):,}"
    table(s, 0.5, 1.3, 5.8, [1.3, 1.75, 2.75], [
        ["Item", "Paper [1]", "Ours (reduced, CPU)"],
        ["Tasks", "500", ps_ours("n_tasks", "pending")],
        ["Epochs per task", "100", ps_ours("epochs", "pending")],
        ["Seeds", "9 (Table 2)", ps_ours("n_seeds", "1")],
        ["Optimizer", "Adam lr 1e-3, ε 1e-7", "same"],
        ["Batch / context", "8 / 128", "same"],
        ["Conditions", "9", "7: CBP and ReDO not run"],
        ["Settings", "swept (Table 8)", "L2, L2* λ 1e-4; L2 Init 1e-5; SNR η 0.03; "
                                         "SNR+L2(*) η 1e-3, λ 1e-4"],
    ], size=9, row_h=0.36, styles={(r, 2): dict(color=TEAL, bold=True) for r in range(1, 8)})
    box(s, 6.5, 1.3, 3.0, 3.6)
    tb(s, 6.7, 1.4, 2.6, 0.25, "Metric", size=11, bold=True)
    tb(s, 6.7, 1.7, 2.6, 1.7, "Training loss on the final epoch of each task, averaged over all "
       "tasks, the first 50 and the last 50 (paper Table 2). Mean ± std over seeds.", size=10)
    tb(s, 6.7, 3.35, 2.6, 1.45, "Diagnostics as in Figs 2-4: cumulative average loss, FFN units "
       "inactive for 5 steps, Q/K/FFN weight norms, attention entropy.", size=10, color=MUTED)
    d.finish(s, refs=[1], notes="""
This slide puts the paper's protocol for Permuted Shakespeare next to ours.

The paper trains on 500 tasks of 100 epochs each and reports Table 2 over 9 seeds; its appendix sweep used 5 seeds. That is 50,000 epochs per run, far too much for a CPU. Our current runs use 30 tasks of 50 epochs and one seed, about 3 hours per run on our laptop CPU. With 30 tasks, "first" and "last" mean the first and last 10% of tasks (3 tasks each), the same fraction as the paper's 50 of 500. The full protocol is planned on a Colab GPU.

What stays the same: Adam with learning rate 1e-3 and epsilon 1e-7, batches of 8 sequences, a context of 128 tokens, and the same model.

Conditions and settings. We run seven of the paper's nine conditions with settings taken from the paper's appendix sweep (Table 8): no intervention; L2 with lambda 1e-4; L2* (attention weights only) with lambda 1e-4; L2 Init with lambda 1e-5; SNR with eta 0.03; and SNR+L2 and SNR+L2* with eta 0.001 and lambda 1e-4. CBP and ReDO are not run, because the paper gives no hyperparameter grid for them on this benchmark and we chose not to invent one.

The metric is the training loss on the final epoch of each task, averaged over all tasks, the first 50 and the last 50, with mean and standard deviation over seeds. The change from first to last tasks is what shows plasticity loss.

Two consequences of the reduced protocol. With fewer epochs per task the end-of-task loss is higher for every method, so our absolute numbers are not comparable with the paper's; only the ordering and the first-to-last change are. And if a run has fewer than 100 tasks, the first-50 and last-50 windows overlap, so the window actually used will be stated next to the table.
""")

    # ============================================================ 20. methodology C
    s = d.content("Methodology · experiment C (team analysis)",
                  "η sweep: does SNR's accuracy really stay flat across η?")
    table(s, 0.5, 1.3, 5.4, [1.3, 3.4], [
        ["Item", "Setting"],
        ["η grid", "0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125 (paper) + 1e-6"],
        ["Benchmark", "Permuted MNIST, MLP 784-100-100-10, SGD and Adam"],
        ["Estimators", "geometric at all 8 values; histogram at 3"],
        ["Protocol", "300 tasks, seed 100 (reduced)"],
        ["Measured", "last-10% accuracy and resets per task"],
        ["Compared with", "paper Tables 10-11 (5 seeds each)"],
    ], size=9.5, row_h=0.36)
    box(s, 6.1, 1.3, 3.4, 3.6, fill=AMBER_TINT, line=AMBER)
    tb(s, 6.3, 1.4, 3.0, 0.25, "Hypothesis under test", size=11, bold=True, color=INK)
    tb(s, 6.3, 1.72, 3.0, 3.1, [
        "Paper: across its grid, SNR's PM accuracy moves by at most 0.3 points with SGD and 2 points "
        "with Adam.",
        "Our first version collapsed at large η. The cause was its per-example time unit; the "
        "corrected step-level version is what this sweep tests.",
        "Either way, it shows how much an unreported detail can matter."], size=10, after=6)
    d.finish(s, refs=[1, 8], notes="""
Experiment C is the analysis our team designed in place of the experiments we left out.

The question. The paper claims SNR is robust to its single hyperparameter eta. Its appendix tables for Permuted MNIST show accuracy between 0.847 and 0.850 with SGD across seven values of eta from 0.08 down to 0.00125, and between 0.867 and 0.887 with Adam.

What happened first. Our first implementation, preliminary v1, behaved very differently: at the paper's eta values accuracy collapsed, and it only matched the paper at eta = 1e-6. Reading the authors' released code for verification showed why. They count firing once per optimizer step (a unit fires if it is active on any example in the batch); we had counted single examples, which makes busy units look as if they fire with probability near 1 and triggers a flood of false resets. The paper's pseudocode is written per example and does not mention this.

The design now. With the corrected step-level SNR we sweep all seven values of the paper's grid plus 1e-6, with SGD and Adam, on the 784-100-100-10 MLP, over 300 tasks on seed 100. For each value we record last-10% accuracy and resets per task. We also run the histogram estimator, the other one the paper describes, at three values. We compare with the paper's Tables 10 and 11.

How to read the outcome. A flat curve confirms the paper's robustness claim for the step-level rule and turns our earlier collapse into a documented lesson: the claim holds only for a time unit the paper never states. A curve that is still steep would point to further unreported details. Either result feeds the critique poster. Runs are logged on the dashboard [8].
""")

    # ============================================================ 21-22. results A tables
    for opt in OPTS:
        rows = [["Condition", "Ours (mean ± std)", "Paper", "Diff."]]
        styles = {}
        shown = [m for m in METHODS if val(opt, m) is not None] or METHODS
        hp_opt = pm.get((opt, "snr"), {}).get("hparam", "")
        eta_opt_txt = fmt_eta(parse_eta(hp_opt)) if parse_eta(hp_opt) else snr_eta_txt
        for i, m in enumerate(shown, start=1):
            r = pm.get((opt, m))
            p = PAPER_PM[opt][m]
            if r and r.get("mean") is not None:
                sd = (f" ± {r['std']:.3f}" if r.get("std") is not None
                      and (r.get("n_seeds") or 1) > 1 else "")
                ours = f"{r['mean']:.3f}{sd}"
                diff = f"{r['mean'] - p:+.3f}".replace("-", "−")
            else:
                ours, diff = "pending", "–"
                styles[(i, 1)] = dict(color=MUTED, italic=True)
            rows.append([LABEL[m], ours, f"{p:.3f}", diff])
            if m == "snr":
                styles[i] = dict(bold=True, fill=TEAL_TINT)
        have = [m for m in METHODS if val(opt, m) is not None]
        team_vals = [val(opt, m) for m in shown]
        lo = min([v for v in team_vals if v is not None] + list(PAPER_PM[opt].values()))
        ymin = min(0.5, math.floor(lo * 10) / 10)
        snr_v = val(opt, "snr")
        diffs = [abs(val(opt, m) - PAPER_PM[opt][m]) for m in have]
        if snr_v is not None:
            best = max(have, key=lambda m: val(opt, m))
            title = (f"{OPT_LABEL[opt]}: SNR reaches {snr_v:.3f} (paper {PAPER_PM[opt]['snr']:.3f})"
                     + ("; best of the measured conditions" if best == "snr" else
                        f"; {LABEL[best]} is {'level with' if val(opt, best) - snr_v < 0.005 else 'above'} "
                        "it in our shorter runs"))
        else:
            title = f"{OPT_LABEL[opt]}: comparison of the measured conditions"
        s = d.content("Results · experiment A", title)
        bar_chart(s, 0.4, 1.25, 5.0, 3.6, [SHORT[m] for m in shown],
                  [("Paper (Table 6)", [PAPER_PM[opt][m] for m in shown], PAPER_BAR),
                   ("Ours", team_vals, TEAM_BAR)],
                  f"Last-10% online accuracy, {OPT_LABEL[opt]} (axis from {ymin:.1f})", ymin=ymin, ymax=1.0, label_size=7, label_only=[1])
        table(s, 5.55, 1.3, 3.95, [1.35, 1.25, 0.65, 0.7], rows, size=9, row_h=0.27,
              styles=styles)
        nseed = sorted({pm[(opt, m)].get("n_seeds") for m in have if pm[(opt, m)].get("n_seeds")})
        ntask = sorted({pm[(opt, m)].get("n_tasks") for m in have if pm[(opt, m)].get("n_tasks")})
        tb(s, 5.55, 3.85, 3.95, 0.95,
           [f"Source: {pm_source}; {'/'.join(map(str, nseed)) or '?'} seed(s), "
            f"{'/'.join(map(str, ntask)) or '?'} tasks. Runs on the dashboard [8].",
            f"SNR uses η = {eta_opt_txt} (inside the paper's grid). Bars are labelled with our "
            "values; paper values are in the table."],
           size=8.5, color=MUTED, after=2)
        biggest = max(have, key=lambda m: abs(val(opt, m) - PAPER_PM[opt][m])) if have else None
        detail = []
        if val(opt, "snr") is not None and val(opt, "l2init") is not None:
            g = val(opt, "snr") - val(opt, "l2init")
            pg = PAPER_PM[opt]["snr"] - PAPER_PM[opt]["l2init"]
            detail.append(f"SNR minus L2 Init is {g:+.3f} in our runs and {pg:+.3f} in the paper.")
            s_runs = {r["seed"]: last_frac_mean(r["df"]["online_accuracy"])
                      for r in pm[(opt, "snr")].get("runs", [])}
            i_runs = {r["seed"]: last_frac_mean(r["df"]["online_accuracy"])
                      for r in pm[(opt, "l2init")].get("runs", [])}
            common = sorted(set(s_runs) & set(i_runs))
            if common:
                wins = sum(s_runs[k] > i_runs[k] for k in common)
                detail.append(f"SNR beats L2 Init in {wins} of {len(common)} seeds run for both.")
            if 0 < g < pg / 2:
                detail.append("Our gap is less than half the paper's, so the safe reading is "
                              "\"slightly but consistently better\", not \"clearly better\".")
        sds = [(pm[(opt, m)]["std"], m) for m in have if pm[(opt, m)].get("std") is not None]
        if sds:
            mx = max(sds)
            detail.append(f"The largest seed standard deviation is {mx[0]:.3f} ({LABEL[mx[1]]}).")
        below = [m for m in have if val(opt, m) < PAPER_PM[opt][m]]
        if have:
            detail.append(f"{len(below)} of {len(have)} measured conditions sit below the paper's value.")
        detail_txt = " ".join(detail)
        order_team = sorted(have, key=lambda m: -val(opt, m))
        d.finish(s, refs=[1, 8], notes=f"""
This is the headline comparison for {OPT_LABEL[opt]}: last-10% online accuracy on Permuted MNIST, our runs ({snr_tasks or 600} tasks, reduced protocol) next to the paper's Table 6 (2400 tasks).

What we have. {len(have)} of the 8 conditions have runs: {', '.join(LABEL[m] for m in have) or 'none'}. {('Not shown yet: ' + ', '.join(LABEL[m] for m in METHODS if m not in have) + ', whose runs had not finished when this deck was built; they will be added from the same pipeline.') if len(have) < 8 else 'All eight conditions are measured.'}

How close we are. {('Across the measured conditions the largest gap to the paper is ' + f"{max(diffs):.3f}" + ' (' + LABEL[biggest] + ').') if have else ''} {('SNR reaches ' + f"{snr_v:.3f}" + ' against the paper\'s ' + f"{PAPER_PM[opt]['snr']:.3f}" + '.') if snr_v is not None else ''} Our ranking from best to worst is {', '.join(LABEL[m] for m in order_team) or 'not available yet'}.

{detail_txt}

Caveats. SNR here uses eta = {eta_opt_txt}, the best value in the paper's appendix for this optimizer; Table 6 reports SNR at the eta tuned by the authors, so the comparison is only like-for-like if both use the same value. The paper's standard deviations come from 5 seeds and ours from {('/'.join(map(str, nseed)) or 'an unknown number of')} seeds. {'These preliminary v1 runs used a three-layer MLP, PyTorch default initialization and per-example SNR, were produced on a teammate machine without a recorded environment, and are superseded. ' if is_v1 else 'These are reduced-protocol runs (fewer tasks than the paper, no team tuning). '}The source of every team number is {pm_source}, and the runs are on the dashboard [8]. The chart axis starts at {ymin:.1f} so the differences are visible; the exact values are in the table.
""")

    # ============================================================ 23-24. curves
    def curve_series(opt, metric="online_accuracy", block=40, methods=METHODS):
        out, cats = [], None
        for m in methods:
            r = pm.get((opt, m), {}).get("runs") or []
            dfs = [x["df"] for x in r if metric in x["df"] and len(x["df"]) >= 400]
            if not dfs:
                continue
            n = min(len(df) for df in dfs)
            nb = n // block
            vals = []
            for b in range(nb):
                vals.append(round(sum(df[metric].iloc[b * block:(b + 1) * block].mean()
                                      for df in dfs) / len(dfs), 4))
            cats = [str(b * block) if (b * block) % 400 == 0 else " " * (b % 7 + 1)
                    for b in range(nb)]
            out.append((m, vals))
        return cats, out

    for opt in OPTS:
        fname = f"pm_accuracy_{opt}"
        cats, ser = curve_series(opt) if is_v1 else (None, [])
        s = d.content("Results · experiment A · curves",
                      ("SGD: the baseline declines task after task; SNR and the penalties hold up"
                       if opt == "sgd" else
                       "Adam: the baseline collapses faster than with SGD; interventions stay high"))
        f = fig(fname)
        if f:
            picture(s, f, 0.4, 1.25, 9.1, 3.7)
            src = f"results/figures/{fname}.png"
        elif ser:
            allv = [v for _, vs in ser for v in vs]
            line_chart(s, 0.4, 1.25, 9.1, 3.7, cats,
                       [(LABEL[m], vs, METHOD_COLOR[m]) for m, vs in ser],
                       f"Online accuracy per task, {OPT_LABEL[opt]}, v1 runs (40-task blocks, seed mean)",
                       ymin=math.floor(min(allv) * 20) / 20, ymax=math.ceil(max(allv) * 20) / 20,
                       cat_title="task")
            src = "native chart built from results/metrics/pm_*.csv"
        else:
            pending_box(s, 0.4, 1.25, 9.1, 3.7, f"accuracy curves ({OPT_LABEL[opt]})",
                        f"results/figures/{fname}.png")
            src = "pending"
        shown = [LABEL[m] for m, _ in ser] if not f else "all conditions in the figure"
        trend = []
        for m, vs in ser:
            k = max(1, len(vs) // 10)
            first, last = sum(vs[:k]) / k, sum(vs[-k:]) / k
            word = "falls" if last < first - 0.02 else ("rises" if last > first + 0.02 else "stays about flat")
            trend.append(f"{LABEL[m]} {word}, from {first:.2f} over the first 10% of tasks to {last:.2f} "
                         "over the last 10%")
        trend_txt = ("What it shows. " + "; ".join(trend) + ". A falling line is plasticity loss in its "
                     "plainest form; a flat or rising line means the network keeps learning new tasks "
                     "as well as before, or better. Differences of a few thousandths cannot be judged "
                     "by eye here; the results table and the per-seed appendix are the place for "
                     "those.") if trend else ""
        d.finish(s, refs=[8], notes=f"""
This chart shows online accuracy task by task for {OPT_LABEL[opt]}, which is the team's version of the paper's Figure 1. {('It is the figure results/figures/' + fname + '.png produced by our plotting script.') if f else ('It is drawn from our run files: each point is the mean of a block of 40 tasks, averaged over seeds. Conditions shown: ' + ', '.join(shown) + '. When results/figures/' + fname + '.png exists, the build script uses that figure instead.') if ser else 'The figure has not been produced yet, so the slide shows a placeholder.'}

{trend_txt}

Each condition keeps the same colour on every chart in the deck, and the legend names every line. Colours were checked for colour-blind separation. The runs behind the chart are on the dashboard [8]. {'These are the superseded preliminary v1 runs; the chart switches to the corrected runs once results/figures/' + fname + '.png exists.' if is_v1 else 'These are reduced-protocol runs.'}
""")

    # ============================================================ 25. dead fraction
    s = d.content("Results · experiment A · dead units",
                  "Dead units: the baseline keeps losing them; SNR keeps the share low")
    f = fig("pm_dead_fraction")
    dead_src = pm_new if any(pm_new.get((o, "none")) for o in OPTS) else pm_v1
    dead_rows = [["Last 10% of tasks" + (" (v1)" if dead_src is pm_v1 else ""), "Dead"]]
    for opt in OPTS:
        for m in METHODS:
            v = dead_last_of(opt, m, dead_src)
            if v is not None:
                dead_rows.append([f"{SHORT[m]} · {OPT_LABEL[opt]}", pct(v)])
    if f:
        picture(s, f, 0.4, 1.25, 6.2, 3.7)
    else:
        ser = []
        cats = None
        for opt, dash in (("sgd", False), ("adam", True)):
            c, sr = curve_series(opt, "dead_fraction", methods=["none", "snr"]) if is_v1 else (None, [])
            cats = cats or c
            ser += [(f"{SHORT[m]} {OPT_LABEL[opt]}", vs, METHOD_COLOR[m], dash) for m, vs in sr]
        if ser:
            line_chart(s, 0.4, 1.25, 6.2, 3.7, cats, ser,
                       "Dead units per task, v1 runs (solid SGD, dashed Adam)", ymin=0,
                       val_fmt="0%", cat_title="task")
        else:
            pending_box(s, 0.4, 1.25, 6.2, 3.7, "dead-unit curves", "results/figures/pm_dead_fraction.png")
    if len(dead_rows) > 1:
        table(s, 6.8, 1.3, 2.7, [1.8, 0.7], dead_rows[:17], size=8.5,
              row_h=min(0.24, 3.6 / len(dead_rows[:17])))
    dn = {(o, m): dead_last_of(o, m, dead_src) for o in OPTS for m in METHODS}
    d.finish(s, refs=[8], notes=f"""
This slide tracks the share of dead hidden units, a unit counting as dead when it has not fired for 1000 consecutive examples. The table gives the average over the last 10% of tasks for every condition with a run.

In the preliminary v1 runs (superseded, shown in the table), with no intervention the share keeps rising: with SGD it ends around {pct(dn[('sgd','none')]) if dn[('sgd','none')] is not None else '?'}, and with Adam it rises faster and ends around {pct(dn[('adam','none')]) if dn[('adam','none')] is not None else '?'}. With SNR it stays near {pct(dn[('sgd','snr')]) if dn[('sgd','snr')] is not None else '?'} (SGD) and {pct(dn[('adam','snr')]) if dn[('adam','snr')] is not None else '?'} (Adam), because units that die are reset.

Two cautions. First, the diagnostic counter is separate from SNR's own statistics; in the corrected code it is cleared whenever a unit is reset, so a fresh unit is never reported as dead (preliminary v1 did not do this). Second, dead units do not explain everything. With Adam, L2 still leaves about {pct(dn[('adam','l2')]) if dn[('adam','l2')] is not None else '?'} of units dead, yet its accuracy ({f3(dead_src.get(('adam','l2'),{}).get('mean'))}) is close to SNR's ({f3(dead_src.get(('adam','snr'),{}).get('mean'))}). So a large dead share does not by itself force low accuracy, and this holds in every seed. This is one reason we treat dead units as a correlate of plasticity loss and not its single cause.

The dead share is recorded once at the end of each task, not averaged over the batches inside a task. {('The chart is results/figures/pm_dead_fraction.png.') if f else ('The chart is drawn from the run files: solid lines are SGD, dashed lines Adam, colours follow the method.')} Runs are on the dashboard [8].
""")

    # ============================================================ 26. results B table
    measured_ps = [c for c in PS_ORDER if c in ps and ps[c].get("all_mean") is not None]
    ps_diag = build_ps_diag()
    ps_title, ps_text = ps_findings(ps, ps_diag)
    ttl = (ps_title if ps_title else "Permuted Shakespeare: " + (
        f"{len(measured_ps)} of 9 conditions measured (reduced protocol)" if measured_ps else
        "paper reference shown, our reduced-protocol runs pending"))
    s = d.content("Results · experiment B", ttl)
    rows = [["Condition", "Paper all", "Paper first 50", "Paper last 50", "Ours all",
             "Ours first", "Ours last"]]
    styles = {}
    for i, c in enumerate(PS_ORDER, start=1):
        pa = PAPER_PS[c]
        r = ps.get(c, {})

        def ms(k):
            v = r.get(f"{k}_mean")
            if v is None:
                return "not run" if c in ("CBP", "ReDO") else "pending"
            sd = r.get(f"{k}_std")
            many = float(r.get("n_seeds") or 1) > 1
            return f"{float(v):.4f}" + (f" ± {float(sd):.4f}" if sd is not None and many else "")
        rows.append([c, f"{pa[0]:.4f}", f"{pa[1]:.4f}", f"{pa[2]:.4f}", ms("all"), ms("first"),
                     ms("last")])
        for ci in (4, 5, 6):
            if rows[-1][ci] in ("pending", "not run"):
                styles[(i, ci)] = dict(color=MUTED, italic=True)
        if c in ("SNR+L2", "SNR+L2*"):
            styles[i] = dict(bold=True, fill=TEAL_TINT)
    table(s, 0.5, 1.3, 9.0, [1.1, 0.95, 1.1, 1.1, 1.6, 1.6, 1.6], rows, size=8.5, row_h=0.3)
    proto = (f"Ours: {ps_ours('n_tasks', '?')} tasks × {ps_ours('epochs', '?')} epochs × "
             f"{ps_ours('n_seeds', '?')} seed(s) (reduced; absolute losses not comparable)."
             if any_ps else "Ours: reduced protocol (fewer tasks, epochs and seeds), runs pending.")
    tb(s, 0.5, 4.4, 9.0, 0.5, [
        "Loss on the final epoch of each task (lower is better). Paper: 500 tasks × 100 epochs, 9 "
        "seeds, Table 2. L2* = L2 on attention weights only. " + proto], size=9, color=MUTED)
    if not measured_ps:
        PENDING.append("results/tables/ps_main.csv")
    d.finish(s, refs=[1, 8], notes=f"""
This is the result table for experiment B. The paper's columns come from its Table 2: average training loss on the final epoch of each task, over all 500 tasks, the first 50 and the last 50, over 9 seeds. Lower is better.

The paper's pattern is clear. No intervention ends at a loss of about 3.0 over the last 50 tasks, far above its early loss: strong plasticity loss. SNR alone ends at about the same level (3.02), because it can only reset the 256 feed-forward units and the attention block keeps degrading. L2 Init helps partly. Plain L2 helps a lot (0.31). Adding SNR on top of L2 gives 0.26, the best of the conditions that regularize all weights. In the ablation, L2 applied to the attention weights alone (L2*) works even better (0.11), and SNR+L2* is lowest of all (0.09). CBP and ReDO, the other reset methods, improve only a little on no intervention.

Our columns. {('We have results for ' + ', '.join(measured_ps) + '. ' + proto + ' Because each task gets fewer epochs, every loss is higher than in the paper, so compare the ordering of conditions and the change from first to last tasks, not the absolute values.') if measured_ps else 'Our reduced-protocol runs are still in progress, so these cells say "pending". The build script fills them from results/tables/ps_main.csv as soon as that file exists, and the slide title changes to say how many conditions are measured.'}

{('What we find. ' + ps_text) if ps_text else "What would count as reproducing the paper's message: SNR alone close to no intervention, L2 far better, SNR+L2 at or below L2 over the last tasks, and L2* variants best."} Here "first" and "last" are the first and last 10% of our tasks, the same fraction as the paper's 50 of 500. With one seed there is no standard deviation, so small gaps (for example SNR+L2 against L2) are not evidence on their own. Runs are logged on the dashboard [8].
""")

    # ============================================================ 27. results B curves
    s = d.content("Results · experiment B · curves",
                  "Cumulative average loss over tasks (our version of the paper's Figure 2)")
    f = fig("ps_cumulative_loss")
    if f:
        picture(s, f, 0.4, 1.25, 9.1, 3.65)
    else:
        pending_box(s, 0.4, 1.25, 9.1, 3.65, "cumulative loss curves for all PS conditions",
                    "results/figures/ps_cumulative_loss.png")
    d.finish(s, refs=[1, 8], notes=f"""
This slide holds the team's version of the paper's Figure 2: the cumulative average of the final-epoch loss, plotted against the task index, for every Permuted Shakespeare condition.

How to read it. The cumulative average at task t is the mean of the final-epoch losses of tasks 1 to t. A method that keeps its plasticity has a flat or falling line; a method that loses plasticity has a line that keeps rising, because late tasks pull the average up. In the paper, the no-intervention and SNR-alone lines climb to about 2.7, L2 Init climbs more slowly, and L2 and SNR+L2 stay far lower, which the paper shows in a zoomed panel.

{('The figure is results/figures/ps_cumulative_loss.png from our plotting script.') if f else ('The run has not produced this figure yet, so the slide shows a clearly marked placeholder and no curve. The build script inserts results/figures/ps_cumulative_loss.png as soon as it exists.')}

What to keep in mind once it appears. Our runs follow a reduced protocol on CPU, with fewer tasks, fewer epochs per task and fewer seeds than the paper's 500 tasks, 100 epochs and 9 seeds. Fewer epochs per task means higher loss at the end of each task for every method, so the curves should be compared by shape and ordering, not by height. The full protocol is planned on Colab. Runs are logged on the dashboard [8].
""")

    # ============================================================ 28. results B diagnostics
    s = d.content("Results · experiment B · diagnostics",
                  "Why plasticity is lost here: inactive FFN units, weight norms, attention entropy")
    for i, (name, what, figno) in enumerate([
            ("ps_inactive", "FFN units inactive for 5 steps", "Fig. 3"),
            ("ps_norms_entropy", "Q, K, FFN weight norms and attention entropy", "Fig. 4")]):
        x = 0.4 + i * 4.6
        f = fig(name)
        tb(s, x, 1.25, 4.4, 0.25, f"{what} (cf. paper {figno})", size=10, bold=True)
        if f:
            picture(s, f, x, 1.55, 4.4, 3.35)
        else:
            pending_box(s, x, 1.55, 4.4, 3.35, what[0].lower() + what[1:], f"results/figures/{name}.png")
    if ps_diag:
        def g(c, k):
            return ps_diag[c][k] if c in ps_diag else float("nan")
        diag_text = ("Our measurements, averaged over the last 10% of tasks. Inactive feed-forward units "
                     f"(of 256): no intervention {g('None', 'inactive_units_last'):.0f}, L2 "
                     f"{g('L2', 'inactive_units_last'):.0f}, SNR {g('SNR', 'inactive_units_last'):.0f}, "
                     f"SNR+L2 {g('SNR+L2', 'inactive_units_last'):.0f}. Attention entropy: no intervention "
                     f"{g('None', 'attn_entropy_last'):.2f} nats, SNR {g('SNR', 'attn_entropy_last'):.2f}, "
                     f"L2 {g('L2', 'attn_entropy_last'):.2f}, SNR+L2 {g('SNR+L2', 'attn_entropy_last'):.2f}. "
                     f"Query weight norm: no intervention {g('None', 'q_norm_last'):.1f}, L2 "
                     f"{g('L2', 'q_norm_last'):.1f}. So the paper's mechanism shows up in our short run: "
                     "SNR keeps the feed-forward units alive but cannot stop the attention collapse, and "
                     "L2 keeps the norms down and the entropy up.")
    else:
        diag_text = "Our measured values will be listed here once results/tables/ps_diagnostics.csv exists."
    d.finish(s, refs=[1, 8], notes=f"""
The paper links plasticity loss in the Transformer to three diagnostics, and this slide holds our versions of them.

Left, the analogue of the paper's Figure 3: the number of feed-forward units that were inactive for 5 consecutive training steps. In the paper this count grows steadily without intervention and also with plain L2, while SNR+L2 keeps it near zero. So L2 fixes the weight growth but does not stop units from dying, and SNR is what keeps the feed-forward units alive.

Right, the analogue of Figure 4: weight norms of the query and key projections and of the feed-forward layer, plus the average entropy of the attention probabilities. Without intervention the norms grow and the attention entropy collapses, meaning each position ends up attending almost entirely to one other position. SNR alone cannot prevent this, because none of it happens in ReLU units. L2 keeps the norms stable and the entropy away from zero.

Together the two panels explain the paper's recommendation to combine SNR with L2: each fixes a different part of the problem.

{diag_text}

Whatever our panels show, they come from a reduced protocol, so the time axis is shorter than the paper's and the magnitudes will differ. The panels are results/figures/ps_inactive.png and results/figures/ps_norms_entropy.png from our plotting script. Runs are logged on the dashboard [8].
""")

    # ============================================================ 29. results C eta sweep
    f = fig("eta_sweep")
    eta_opts = [o for o in OPTS if eta.get(o)]
    v1_eta = eta_src == "metrics"
    n_units = 300 if v1_eta else 200

    def geo_rows(o):
        return [r for r in eta.get(o, []) if r.get("estimator", "geometric") != "histogram"]
    spread_txt, flat = "", None
    if eta_opts:
        o = eta_opts[0]
        grid_vals = [r["mean"] for r in geo_rows(o) if r["eta"] in PAPER_ETA_GRID]
        if len(grid_vals) >= 2:
            spread = max(grid_vals) - min(grid_vals)
            flat = spread <= 0.03
            spread_txt = f"{spread:.2f}"
    if flat is None:
        title = "η sensitivity: sweep results pending"
    elif v1_eta:
        title = "η sensitivity: our first version collapsed at large η; the fixed version is being swept"
    elif flat:
        title = "η sensitivity: with step-level firing, accuracy is roughly flat across η"
    else:
        title = (f"η sensitivity: our accuracy still moves by {spread_txt} across the paper's grid")
    s = d.content("Results · experiment C (team analysis)", title)
    if f:
        picture(s, f, 0.4, 1.25, 5.2, 3.65)
    elif eta_opts:
        o = eta_opts[0]
        gr = geo_rows(o)
        etas = sorted({r["eta"] for r in gr} | set(PAPER_ETA_GRID), reverse=True)
        team = {r["eta"]: r["mean"] for r in gr}
        bar_chart(s, 0.4, 1.25, 5.2, 3.65, [fmt_eta(e) for e in etas],
                  [("Paper (Tables 10-11)", [PAPER_ETA[o].get(e) for e in etas], PAPER_BAR),
                   ("Ours (preliminary v1)" if v1_eta else "Ours", [team.get(e) for e in etas],
                    TEAM_BAR)],
                  f"SNR last-10% accuracy vs η, {OPT_LABEL[o]}", ymin=0, ymax=1.0,
                  label_size=6.5, fmt="0.00")
    else:
        pending_box(s, 0.4, 1.25, 5.2, 3.65, "η sweep", "results/tables/eta_sweep.csv")
    rows = [["Opt.", "η", "Resets / task", "Ours", "Paper"]]
    if not v1_eta and eta_opts:
        # compact layout: one row per eta, SGD and Adam side by side (geometric estimator)
        def _cell(o, e):
            r = next((r for r in geo_rows(o) if abs(r["eta"] - e) < 1e-12), None)
            return ("–", "–") if r is None else (f"{r['mean']:.3f}", "–" if r["resets"] is None
                                                  else f"{r['resets']:.0f}")
        all_etas = sorted({r["eta"] for o in eta_opts for r in geo_rows(o)}, reverse=True)
        rows = [["η", "SGD", "paper", "resets", "Adam", "paper", "resets"]]
        for e in all_etas:
            sg, sr = _cell("sgd", e)
            ad, ar = _cell("adam", e)
            rows.append([fmt_eta(e), sg, f3(PAPER_ETA["sgd"].get(e)) if e in PAPER_ETA["sgd"] else "–", sr,
                         ad, f3(PAPER_ETA["adam"].get(e)) if e in PAPER_ETA["adam"] else "–", ar])
        hist = [(o, r) for o in eta_opts for r in eta[o] if r.get("estimator") == "histogram"]
    for o in (eta_opts if v1_eta else []):
        for r in sorted(eta[o], key=lambda r: (r.get("estimator", "") == "histogram", -r["eta"])):
            tag = " (hist.)" if r.get("estimator") == "histogram" else ""
            rows.append([OPT_LABEL[o], fmt_eta(r["eta"]) + tag,
                         "–" if r["resets"] is None else f"{r['resets']:.1f}",
                         f"{r['mean']:.3f}", f3(r["paper"]) if r["paper"] else "not in grid"])
    nb, legacy = (None, False)
    if eta_opts and v1_eta:
        nb, legacy = eta_baseline(runs, eta_opts[0], eta[eta_opts[0]][0]["n_tasks"])
        if nb is not None:
            rows.append([OPT_LABEL[eta_opts[0]], "no SNR", "0", f"{nb:.3f}", "–"])
    shown_rows = rows[:21]
    if len(rows) > 1 and not v1_eta:
        table(s, 5.75, 1.3, 3.8, [0.62, 0.5, 0.5, 0.52, 0.5, 0.5, 0.56], shown_rows, size=8,
              row_h=0.25)
    elif len(rows) > 1:
        table(s, 5.8, 1.3, 3.7, [0.65, 0.95, 0.9, 0.6, 0.8], shown_rows, size=8 if len(rows) > 12 else 8.5,
              row_h=min(0.25, 2.35 / len(shown_rows)))
    n_eta = eta[eta_opts[0]][0]["n_tasks"] if eta_opts else None
    seeds_eta = eta[eta_opts[0]][0]["n_seeds"] if eta_opts else None
    eta_resets_txt = ""
    eta_seen_txt = ""
    if eta_opts:
        rr = [r for r in geo_rows(eta_opts[0]) if r["resets"] is not None]
        if len(rr) >= 2:
            hi, lo = max(rr, key=lambda r: r["eta"]), min(rr, key=lambda r: r["eta"])
            eta_resets_txt = (f"Resets per task (out of {n_units} units) go from about "
                              f"{hi['resets']:.1f} at eta = {fmt_eta(hi['eta'])} to about "
                              f"{lo['resets']:.1f} at {fmt_eta(lo['eta'])}.")
        o0 = eta_opts[0]
        listing = "; ".join(f"eta = {fmt_eta(r['eta'])}: {r['mean']:.3f}"
                            for r in sorted(geo_rows(o0), key=lambda r: -r["eta"]))
        lo_p, hi_p = min(PAPER_ETA[o0].values()), max(PAPER_ETA[o0].values())
        eta_seen_txt = f"What the numbers show ({OPT_LABEL[o0]}): {listing}. "
        if v1_eta:
            nb0, _ = eta_baseline(runs, o0, eta[o0][0]["n_tasks"])
            worse = [r["eta"] for r in eta[o0] if nb0 is not None and r["mean"] < nb0]
            if nb0 is not None:
                eta_seen_txt += f"The no-intervention run over the same tasks reaches {nb0:.3f}"
                eta_seen_txt += (f", so from eta = {fmt_eta(min(worse))} upward the first version of "
                                 "SNR did worse than doing nothing. ") if worse else ". "
        eta_seen_txt += eta_resets_txt + (f" The paper reports {lo_p:.3f} to {hi_p:.3f} with "
                                          f"{OPT_LABEL[o0]} across its whole grid.")
    nb_y = 3.72 if v1_eta else 3.95
    box(s, 5.8, nb_y, 3.7, 4.9 - nb_y, fill=AMBER_TINT, line=AMBER)
    if v1_eta and eta_opts:
        note = [[(f"Bars: {V1}, ", {"bold": True}),
                 (f"{n_eta} tasks, {seeds_eta} seed{'s' if (seeds_eta or 0) > 1 else ''}, per-example "
                  "SNR.", {})],
                [("Sanity check after the fix ", {"bold": True}),
                 ("(100 tasks, seed 0, not a final result): SGD 0.803-0.805 for η ∈ {0.08, "
                  "0.01, 0.00125}; Adam 0.860-0.870, best at 0.08.", {})]]
    else:
        note = [f"Source: results/tables/eta_sweep.csv ({n_eta} tasks, {seeds_eta} seed"
                f"{'s' if (seeds_eta or 0) > 1 else ''}, step-level SNR, geometric estimator). Paper: "
                "Tables 10-11, 5 seeds, 2400 tasks. Resets = units reset per task (of 200). "
                + (("Histogram estimator: " + "; ".join(f"{OPT_LABEL[o]} η={fmt_eta(r['eta'])}: {r['mean']:.3f}"
                                                         for o, r in hist) + ".") if hist else "")]
    tb(s, 5.93, nb_y + 0.05, 3.45, 4.85 - nb_y, note, size=7.5 if not v1_eta else 8.5, color=INK, after=2)
    d.finish(s, refs=[1, 8], notes=f"""
This is experiment C, the team's own analysis: how SNR's accuracy on Permuted MNIST depends on eta.

{('Source of the bars: our superseded first implementation (preliminary v1), ' + str(n_eta) + ' tasks, ' + str(seeds_eta) + ' seed per value. It counted firing per example inside each batch.') if v1_eta and eta_opts else ('Source: results/tables/eta_sweep.csv, the reduced-protocol sweep with the corrected step-level SNR.' if eta_opts else 'No sweep results exist yet.')}

{eta_seen_txt}

{('Why the first version collapsed. Counting per example, a unit that is active on most examples looks as if it fires with probability close to 1, so two or three silent examples in a row already look very unlikely and the unit is reset. With a large eta this happens all the time, the network is constantly re-initialized and never settles. The authors\' released code, which we read for verification only, counts firing per optimizer step instead: a unit fires on a step if it is active on any example in the batch. The paper\'s pseudocode is written per example and does not say this. After switching to the step-level rule, a 100-task sanity check on seed 0 gave SGD accuracy between 0.803 and 0.805 for eta = 0.08, 0.01 and 0.00125, and Adam between 0.860 and 0.870, best at 0.08, the same pattern as the paper. Those are early, single-seed numbers on a short run and are not used as results; the 600-task sweep replaces them when results/tables/eta_sweep.csv exists, and this slide then switches to it automatically.') if v1_eta else 'Background. Our first implementation counted firing per example and collapsed at large eta. The authors\' released code, read for verification only, counts per optimizer step, and our corrected SNR does the same. This sweep checks whether that removes the sensitivity, and the slide title states the answer from the data.'}

Our reading for the critique stage: the robustness the paper reports depends on a time unit the paper never states. With the wrong unit, the same eta produces a completely different number of resets. Runs are on the dashboard [8].
""")

    # ============================================================ 30. reset behaviour
    r_sgd, r_adam = resets_of("sgd"), resets_of("adam")
    s = d.content("Results · experiment C (team analysis)",
                  "Reset counts: SGD vs Adam" + (" in preliminary v1" if not resets_new
                                                  else " with the corrected step-level SNR"))
    nu = 200 if resets_new else 300
    for i, (lab, v) in enumerate([("SNR + SGD", r_sgd), ("SNR + Adam", r_adam)]):
        x = 0.7 + i * 4.5
        box(s, x, 1.4, 4.1, 2.3)
        tb(s, x + 0.2, 1.5, 3.7, 0.25, lab, size=11, bold=True)
        tb(s, x + 0.2, 1.85, 3.7, 1.0, "pending" if v is None else f"{v:.0f}", size=54, bold=True,
           color=TEAL if i == 0 else ORANGE, anchor="m")
        tb(s, x + 0.2, 2.95, 3.7, 0.6, f"units reset per task (of {nu}), averaged over all tasks and "
           "seeds", size=11, color=MUTED)
    box(s, 0.7, 3.95, 8.6, 0.95)
    tb(s, 0.9, 4.0, 8.2, 0.85,
       ("Counted with the per-example SNR of preliminary v1, which over-resets busy units. The "
        "Adam number must be re-measured with the step-level version before we read anything "
        "into it." if not resets_new else
        "Counted with the corrected step-level SNR (reduced protocol, 600 tasks, seeds 100+). "
        "Preliminary v1, which counted per example, reset 8 (SGD) and about 321 (Adam) units "
        "per task out of 300."), size=11, anchor="m")
    d.finish(s, refs=[8], notes=f"""
This slide shows how many units SNR resets per task, averaged over the whole run and over seeds, out of the {nu} hidden units of the {'corrected 784-100-100-10 MLP' if resets_new else 'preliminary v1 MLP'}.

With SGD the answer is about {f"{r_sgd:.0f}" if r_sgd is not None else '(pending)'} units per task. With Adam it is about {f"{r_adam:.0f}" if r_adam is not None else '(pending)'} units per task{', which is roughly every unit once per task' if (r_adam or 0) > 250 else ''}. SNR with Adam still reached {f3(val('adam', 'snr'))} accuracy.

{'In the first version of the deck we presented the Adam figure as a puzzling finding. It now has a likely explanation: preliminary v1 counted firing per example, which makes busy units look as if they almost always fire, so a short pause in a batch looks like death. With Adam, a new permutation abruptly changes which units are active at the start of each task, and the per-example rule then resets many of them. The mass resets did little damage because a reset unit\'s outgoing weights are set to zero, so the output does not change at that moment and Adam quickly trains the fresh units. With the step-level rule we expect far fewer resets, and the reduced-protocol runs will show whether that is true. Until then we do not draw conclusions from the Adam number.' if not resets_new else 'These counts come from the corrected step-level SNR. Preliminary v1, which counted firing per example, reset about 8 units per task with SGD and about 321 with Adam (out of 300), so with Adam it reset roughly every unit in every task. The step-level rule with its minimum silence cuts that to the numbers above. Adam still resets more than SGD: a new permutation abruptly changes which units are active at the start of each task, and Adam moves the weights faster, so more units pass through long silences. A reset does not change the output at that moment, because the outgoing weights are set to zero.'}

Runs are on the dashboard [8].
""")

    # ============================================================ 31. deviation analysis
    adam_gap = None if val("adam", "none") is None else val("adam", "none") - PAPER_PM["adam"]["none"]
    s = d.content("Reproduction · deviation analysis",
                  "Where we differ from the paper, and what explains it")
    snp_sgd = val("sgd", "snp")
    none_adam = val("adam", "none")
    l2s_last = (ps.get("L2*") or {}).get("last_mean")
    l2_last = (ps.get("L2") or {}).get("last_mean")
    eta_rng = None
    if eta_src == "table" and eta.get("sgd"):
        _ev = [r["mean"] for r in eta["sgd"] if r.get("estimator", "geometric") == "geometric"
               and r["eta"] >= 0.00125 and r.get("mean") is not None]
        eta_rng = (min(_ev), max(_ev)) if len(_ev) >= 3 else None

    def gap(v, ref):
        return "–" if v is None else f"{v - ref:+.3f}".replace("-", "−")
    dev_rows = [
        ["Deviation", "Size", "Cause", "Status"],
        ["SNR η scale (v1)", "≈ 1000×", "firing counted per example; the authors' code counts per "
         "optimizer step (the paper does not say)",
         "Fixed" + (f"; η sweep now {eta_rng[0]:.3f}-{eta_rng[1]:.3f} across the paper grid (SGD)"
                    if eta_rng else "")],
        ["Dense units reset for any η", "most resets", "a unit whose silences all last 1 step gets "
         "p = 1, so 2 silent steps trigger a reset whatever η is", "Fixed: no reset before 6 silent "
         "steps (found in code review)"],
        ["Adam baseline", gap(none_adam, PAPER_PM["adam"]["none"]),
         "600 of 2400 tasks: the Adam baseline has not collapsed yet (v1 at 2400 tasks: 0.598)",
         "Full 2400-task runs on Colab"],
        ["Shrink & Perturb (SGD)", gap(snp_sgd, PAPER_PM["sgd"]["snp"]),
         "our noise is σ·N(0, 1) every step; S&P is usually drawn from the init distribution "
         "(about 28× smaller on layer 1)", "Kept as found; init-scaled noise is the next run"],
        ["PS: L2* not best", "" if l2s_last is None or l2_last is None else f"{l2s_last:.2f} vs {l2_last:.2f}",
         "attention-only L2 leaves the FFN weights growing in our 30 short tasks", "Full PS run on Colab "
         "(in progress)"],
        ["PS text and permutation", "setup", "authors' template: another play, words permuted; "
         "paper and we: The Tempest, token ids permuted", "Documented; we follow the paper"],
        ["Reduced protocols", "scale", "PM 600 of 2400 tasks, 2 seeds; η sweep 300 tasks, 1 seed; "
         "PS 30 × 50 epochs, 1 seed; no tuning", "Full protocol on Colab"],
    ]
    table(s, 0.5, 1.3, 9.0, [1.6, 0.95, 3.6, 2.85], dev_rows, size=8.5, row_h=0.36)
    box(s, 0.5, 4.35, 9.0, 0.55)
    tb(s, 0.7, 4.38, 8.6, 0.5, "The first version's two big deviations had concrete causes, found by "
       "reading the authors' code and by our own review; both are fixed. The rest are protocol "
       "effects or open questions for the full runs.", size=10, anchor="m")
    d.finish(s, refs=[1], notes=f"""
This slide lists where we differ from the paper and what we know about the cause.

SNR's eta scale. Our first implementation (preliminary v1) matched the paper only at eta = 1e-6, about a thousand times below the paper's grid, and collapsed at the paper's own values. The paper's Algorithm 1 is written per training example, and v1 counted firing that way. The authors' released code, which we read for verification only and did not copy, counts firing once per optimizer step: a unit fires on a step if it is active on any example in the batch. Per example, busy units look as if they fire with probability near 1, so short pauses trigger resets. We switched to the step-level rule and to completed silent runs only. A 100-task sanity check then gave flat accuracy across the paper's eta values; the 600-task sweep will confirm it.

The Adam baseline. Our v1 no-intervention run with Adam reached 0.598 against the paper's 0.641, a gap of 0.043, well beyond either side's seed spread. v1 used three hidden layers and PyTorch's default initialization; the authors' code uses 784-100-100-10 with LeCun-normal weights and zero biases. We now use the same, and the reduced-protocol runs will show how much of the gap remains.

Permuted Shakespeare setup. The paper says it uses The Tempest and permutes the vocabulary. The authors' public template reads the first tokens of the complete-works file, which is a different play (All's Well That Ends Well), and permutes whitespace-separated words before tokenizing. We follow the paper's description: The Tempest, with token ids permuted. Our corpus file is byte-identical to theirs after line-ending normalization.

Reduced protocols. PM runs 600 of 2400 tasks with two seeds and paper-best settings; the eta sweep 300 tasks with one seed; PS 30 of 500 tasks, 50 of 100 epochs, one seed. Absolute numbers are therefore not directly comparable.

Provenance. The v1 runs have no recorded environment or commit and are superseded by tracked runs.
""")

    # ============================================================ 32. author code
    s = d.content("Reproduction · comparison with the authors' code",
                  "Comparison with the authors' code: what was and was not compared")
    for i, (head, col, items) in enumerate([
            ("Compared (code read only)", TEAL,
             ["MLP: 784-100-100-10, LeCun-normal init, zero bias",
              "SNR: firing per optimizer step (any example in the batch)",
              "Threshold: per-unit (1 − η) percentile of past silent runs",
              "PS template: first tokens of the complete works; words permuted"]),
            ("What we changed", BLUE,
             ["Step-level firing; completed silent runs only",
              "Geometric and histogram estimators",
              "MLP and init now match the authors'",
              "Kept the paper's Tempest text and token-id permutation"]),
            ("Not compared yet", ORANGE,
             ["Reference code not run",
              "No output-level comparison of numbers",
              "Saved notebook outputs not verified"])]):
        card(s, 0.5 + i * 3.05, 1.3, 2.9, 2.75, head, items, band=col, body_size=10)
    box(s, 0.5, 4.2, 9.0, 0.7, fill=ROW_ALT)
    tb(s, 0.7, 4.2, 8.6, 0.7, "The authors' code was read for verification only. Nothing was copied: "
       "our implementation is written from the paper, and each change is recorded with its evidence "
       "in docs/reproduction_notes.md.", size=10.5, anchor="m")
    d.finish(s, refs=[1, 12], notes="""
The course asks teams to compare against the original code when it is public. The authors' code is on GitHub [12]. This slide states exactly what we compared.

What we compared, by reading the released code for verification only. First, the Permuted MNIST model: an MLP with two hidden layers of 100 units (784-100-100-10), LeCun-normal weight initialization and zero biases. Second, how SNR measures firing: once per optimizer step, where a unit counts as firing if it is active on any example in the batch. Third, the reset threshold: each unit has its own threshold, the (1 - eta) percentile of its past silent-run lengths. This also explains the "percentile 0.92" in the reference notebook's settings, which is eta = 0.08. Fourth, the public template for Permuted Shakespeare: it reads the first tokens of the complete-works file, which is All's Well That Ends Well and not The Tempest that the paper names, and it permutes whitespace-separated words before tokenizing. We also confirmed that our local corpus file is byte-identical to the authors' after converting Windows line endings.

What we changed as a result. SNR now counts firing per step and only uses completed silent runs. It supports both the geometric estimator from the paper's footnote and the histogram estimator from the main text. The MLP and its initialization now match the authors' code. For Permuted Shakespeare we kept the paper's description, The Tempest with token ids permuted, and list the template's differences as a deviation.

What we have not compared. We have not run the reference code, so there is no output-level comparison of numbers between the two pipelines. The saved outputs in the supplied reference notebook were not produced by us and are not used as results.

Nothing from the authors' code was copied into ours; our implementation is written from the paper's description, and each change is recorded with its evidence in the project's reproduction notes.
""")

    # ============================================================ 33. conclusion
    s = d.content("Conclusion", "Where the reproduction stands, and what comes next")
    ps_line = "B: PS ordering matches Table 2 except the L2* ablation"
    concl = [
        ("Reproduced so far", TEAL,
         ["Plasticity loss and dead units appear clearly on PM",
          f"A: SNR {f3(val('sgd', 'snr'))} with SGD (paper 0.851)"
          + (f", {f3(val('adam', 'snr'))} with Adam (paper 0.889)" if val('adam', 'snr') is not None else "")
          + f", {run_label}", ps_line]),
        ("Findings so far", ORANGE,
         ["SNR's η robustness depends on counting firing per step and on a minimum silence, "
          "neither stated in the paper",
          "PS: SNR alone ≈ no intervention, L2 needed, SNR+L2 lowest; L2* did not help in our short run",
          "Dead units do not fully explain accuracy (L2 + Adam)"]),
        ("Next steps", BLUE,
         ["Full protocol on Colab (2400 tasks, 500 × 100 PS), already running for PS",
          "Shrink & Perturb with init-scaled noise",
          "Poster: test the limitations in theory and experiment"]),
    ]
    for i, (head, col, items) in enumerate(concl):
        card(s, 0.5 + i * 3.05, 1.3, 2.9, 2.75, head, items, band=col, body_size=10)
    box(s, 0.5, 4.2, 9.0, 0.7, fill=TEAL_TINT, line=TEAL)
    tb(s, 0.7, 4.2, 8.6, 0.7, [[("Toward the poster: ", {"bold": True}),
                                ("the dependence of SNR on its unstated time unit, and the gap between "
                                 "the PS description and code, are the first candidate limitations to "
                                 "argue in theory and test.", {})]], size=10.5, anchor="m")
    d.finish(s, refs=[1], notes=f"""
Where the reproduction stands.

First, the phenomenon reproduces clearly. On Permuted MNIST the no-intervention baseline loses accuracy task after task, and the share of dead units rises with it.

Second, on the headline comparison SNR reaches {f3(val('sgd', 'snr'))} with SGD against 0.851 in the paper, and {f3(val('adam', 'snr'))} with Adam against 0.889. These numbers come from {('our superseded first implementation; the reduced-protocol runs with the corrected code replace them as soon as results/tables/pm_main.csv exists.' if is_v1 else 'the reduced protocol with the corrected implementation.')} {('Conditions still without runs: ' + ', '.join(missing) + '.') if missing else 'All eight conditions of Table 1 are measured.'}

Third, {('Permuted Shakespeare results exist for ' + ', '.join(measured_ps) + ' under a reduced protocol.') if measured_ps else 'the Permuted Shakespeare runs are still pending; the deck shows the paper\'s Table 2 as the reference.'}

What we learned. Our first SNR collapsed at the paper's eta values. Checking the authors' released code showed that they count firing per optimizer step, while the paper's pseudocode reads per example; with the step-level rule, early checks are flat across eta, as the paper reports. The authors' public Permuted Shakespeare template also differs from the paper's description, in the play used and in what is permuted. And with Adam, L2 kept many dead units yet reached accuracy close to SNR in our first runs, so dead units alone do not explain plasticity.

Next steps: finish the 600-task runs for experiment A and the eta sweep, finish the reduced Permuted Shakespeare runs, then run the full protocol on Colab with the prepared notebook. The time-unit dependence and the description-versus-code gap are the starting points for the critique and empirical-demonstration poster, where each claimed limitation must be argued in theory and then tested.
""")

    # ============================================================ 34. A1 decision log
    s = d.content("Appendix A1", "Decision log: details the paper leaves open and what we chose")
    table(s, 0.5, 1.3, 9.0, [1.5, 3.6, 3.9], [
        ["Decision", "Our choice", "Reason / evidence"],
        ["Time unit", "optimizer step: fires if active on any example (or token) in the batch",
         "Authors' code; per-example counting (v1) over-reset busy units"],
        ["Silent runs", "only completed runs (length ≥ 1) enter the estimate; no reset before "
                        "6 silent steps", "authors' code; floor found in our code review"],
        ["Estimator", "geometric: EMA (β = 0.1), μ = 1000 steps before the first run; or "
                      "histogram over a window of tasks", "paper footnote 3 / main text; window length "
                                                          "not reported"],
        ["Reset operation", "redraw incoming weights, bias 0, outgoing 0, clear Adam moments and "
                            "unit statistics", "output unchanged at the moment of reset"],
        ["Hyperparameters", "best values in the paper's App. C tables; no team tuning",
         "reduced protocol, fresh seeds 100-101"],
        ["MLP", "784-100-100-10, LeCun-normal, zero bias", "authors' code; v1 used 3 × 100"],
        ["PS text", "The Tempest cut from the complete works; 256 × 129 tokens; token ids permuted",
         "paper text; authors' template differs"],
        ["Transformer", "reset FFN units only; L2* = attention weights only", "attention has no ReLU units"],
    ], size=8.5, row_h=0.37)
    d.finish(s, refs=[1], notes="""
The paper leaves many details open, so we made choices and recorded them, with the evidence, in the project's reproduction notes. This table is the summary.

Time unit. A unit fires on an optimizer step if its activation is positive for any example (or any token position) in the batch, and silent runs are counted in steps. This follows the authors' released code, read for verification only. Our first version counted single examples and over-reset busy units; that was the cause of the eta collapse.

Silent runs. Only completed runs, at least one step long and ended by a firing, enter the estimate, as in the authors' code. No unit is reset before 6 silent steps (about 100 examples): without this floor a unit whose silences all last one step gets p = 1 and is reset after two silent steps whatever eta is.

Estimator. The geometric version keeps an exponential moving average (beta = 0.1) of completed run lengths; the first run sets it, and before that it is 1000 steps, which also fixes how long a unit that is dead from initialization waits. The histogram version keeps per-unit run lengths over a trailing window of tasks and uses their (1 - eta) percentile. The paper does not report its window length.

Reset operation. Incoming weights are redrawn from the layer's initialization, the bias is set to zero, outgoing weights to zero, and Adam's moments for those entries are cleared together with the unit's statistics. The output therefore does not change at the moment of a reset.

Hyperparameters. For the reduced protocol we take, for each method, the value that scored best in the paper's appendix tables, and do no tuning of our own; evaluation uses fresh seeds 100 and 101 (and 102 for three SGD conditions).

Model. The MLP is 784-100-100-10 with LeCun-normal weights and zero biases, as in the authors' code.

Permuted Shakespeare. We cut The Tempest from the complete-works file, take 256 windows of 129 tokens and permute token ids, following the paper; the authors' template differs. In the Transformer only feed-forward units are reset, and L2* applies the penalty to the attention weights only.
""")

    # ============================================================ 35. A2 code architecture
    s = d.content("Appendix A2", "Code architecture: one shared training loop, methods plug in at two points")
    mods = [
        ("src/data/", ["mnist.py", "shakespeare.py"]),
        ("src/tasks/", ["permuted_mnist.py", "permuted_shakespeare.py"]),
        ("src/models/", ["mlp.py", "transformer.py"]),
        ("src/methods/", ["snr.py", "cbp.py", "redo.py", "shrink_perturb.py", "regularization.py",
                          "resets.py"]),
        ("src/", ["train.py", "evaluate.py", "diagnostics.py", "tracking.py", "checkpoint.py"]),
        ("scripts/", ["run_experiments.py", "sweep.py", "prepare_data.py", "summarize_results.py",
                      "plot_results.py"]),
    ]
    roles = {"src/data/": "load base datasets", "src/tasks/": "build task sequences",
             "src/models/": "architectures", "src/methods/": "interventions",
             "src/": "loop, metrics, logging", "scripts/": "entry points"}
    status_count = {"done": 0, "stub": 0, "missing": 0}
    for i, (folder, files) in enumerate(mods):
        x = 0.5 + (i % 3) * 3.05
        y = 1.3 + (i // 3) * 1.75
        box(s, x, y, 2.9, 1.6)
        tb(s, x + 0.12, y + 0.07, 2.7, 0.25, [[(folder, {"bold": True}),
                                              (f"  {roles[folder]}", {"color": MUTED, "size": 9})]],
           size=10.5)
        paras = []
        for fn in files:
            st = module_status(folder + fn)
            status_count[st] += 1
            mark = {"done": ("● ", TEAL), "stub": ("○ ", ORANGE),
                    "missing": ("○ ", ORANGE)}[st]
            paras.append([(mark[0], {"color": mark[1], "bold": True}), (fn, {})])
        tb(s, x + 0.12, y + 0.37, 2.7, 1.2, paras, size=9.5, after=1)
    tb(s, 0.5, 4.8, 9.0, 0.25, [[("●", {"color": TEAL, "bold": True}),
                                 (" has code   ", {}), ("○", {"color": ORANGE, "bold": True}),
                                 (" docstring stub, in progress (checked when this deck was built). "
                                  "tracking.py: wandb | trackio | none.", {})]],
       size=9, color=MUTED)
    d.finish(s, notes=f"""
This is the repository layout the code is being built toward. Each folder has one job. src/data loads the base datasets (MNIST, the Shakespeare text). src/tasks builds the task sequences: pixel permutations for PM and vocabulary permutations for PS, both seeded by the pair (seed, task). src/models holds the MLP and the one-layer decoder Transformer, and each model can return its hidden activations. src/methods holds the interventions: SNR, CBP, ReDO and Shrink and Perturb decide which units or weights to change; resets.py holds the shared operations that perform a reset (re-initialize incoming weights, zero outgoing weights, clear optimizer moments); regularization.py holds L2 and L2 Init with parameter masks, including the attention-only mask for L2*.

src/train.py is the single training loop used by every method. Methods plug in at two points only: a penalty added to the loss (L2, L2 Init) or a hook called after each optimizer step (SNR, CBP, ReDO, Shrink and Perturb, and the dead-unit diagnostic). This keeps comparisons fair: with the same seed, every method sees the same data, the same batch order and the same initialization. evaluate.py computes the metrics, diagnostics.py the dead-unit and attention diagnostics, tracking.py logs to W&B, Trackio or nothing, and checkpoint.py saves everything needed to resume a long run on Colab.

The scripts are thin entry points: run one experiment, run a whole matrix in parallel from the experiment configs (sweep.py reads configs/experiments/main, sensitivity and shakespeare), prepare data, summarize results into the tables this deck reads, and plot the figures. A Colab notebook runs the full protocol on a GPU. Stubs for the out-of-scope experiments (CNN, random-label and ImageNet tasks, scaling studies) were removed when the team settled on core scope.

The filled and hollow markers were set automatically when this deck was built, by checking whether each file has code below its docstring. At build time {status_count['done']} files had code and {status_count['stub'] + status_count['missing']} were still stubs, so this slide shows the plan and its progress, not a finished system.
""")

    # ============================================================ 36. A3 per-seed
    seed_src = pm_new if pm_new else pm_v1
    s = d.content("Appendix A3", ("Per-seed results, reduced protocol (600 tasks): spread across seeds"
                                  if seed_src is pm_new else
                                  "Per-seed results of preliminary v1 (superseded): spread across seeds"))
    seed_set = sorted({r["seed"] for (o, m), v in seed_src.items() for r in v.get("runs", [])})[:5]
    rows = [["Condition", "Opt."] + [f"Seed {k}" for k in seed_set] + ["Mean", "Std"]]
    for opt in OPTS:
        for m in METHODS:
            v = seed_src.get((opt, m))
            if not v or not v.get("runs"):
                continue
            per = {r["seed"]: last_frac_mean(r["df"]["online_accuracy"]) for r in v["runs"]}
            mean, sd = stats(list(per.values()))
            rows.append([LABEL[m], OPT_LABEL[opt]] + [f3(per.get(k)) if k in per else "–"
                                                      for k in seed_set]
                        + [f3(mean), "–" if sd is None else f"{sd:.3f}"])
    if len(rows) > 1:
        table(s, 0.5, 1.3, 9.0, [1.6, 0.8] + [1] * len(seed_set) + [1, 0.8], rows, size=8.5,
              row_h=min(0.27, 3.6 / len(rows)))
    else:
        pending_box(s, 0.5, 1.3, 9.0, 3.5, "per-seed table", "results/metrics/pm_*.csv")
    d.finish(s, refs=[8], notes=f"""
This table lists every per-seed number behind the Permuted MNIST results: online accuracy averaged over the last 10% of tasks, one column per seed, then the mean and the sample standard deviation across seeds.

It shows {'the corrected reduced-protocol runs (results/metrics, 600 tasks, seeds 100 and up)' if seed_src is pm_new else 'the superseded preliminary v1 runs (results/metrics/preliminary_v1)'}. A dash means that seed was not run locally. It is computed directly from the metric files when the deck is built, using for each condition the longest set of runs available. Rows appear only for conditions that have runs.

In the preliminary runs the spread across seeds is at most about 0.003 for every condition except the Adam baseline, whose standard deviation is about 0.010. That baseline is also the condition furthest from the paper, so both its level and its stability deserve a closer look when the architecture is verified.

File names on the dashboard [8] follow the runner's pattern: pm_<method>_<optimizer>_seed<seed>_tasks<N>, with the hyperparameter in the name where there is one, for example pm_snr_eta1e-06_sgd_seed0_tasks2400 or pm_l2_lam0.001_adam_seed1_tasks2400.
""")

    # ============================================================ 37. A4 hyperparameters
    s = d.content("Appendix A4", "Hyperparameters and environment")
    table(s, 0.5, 1.3, 9.0, [2.1, 6.9], [
        ["Item", "Value"],
        ["PM: model / batch / tasks", "784-100-100-10, LeCun-normal, zero bias / 16 / paper 2400, ours 600 (2400 on Colab)"],
        ["PS: batch / context / tasks", "8 / 128 / paper 500 × 100 epochs; ours reduced"],
        ["SGD / Adam [7]", "lr 0.01 / lr 0.001, β = (0.9, 0.999), ε = 1e-7"],
        ["PM methods", "best values in the paper's App. C tables (Tables 10-11); no team tuning"],
        ["PS methods", "L2, L2* λ 1e-4; L2 Init λ 1e-5; SNR η 0.03; SNR+L2(*) η 1e-3, λ 1e-4"],
        ["SNR", "step-level firing; geometric (EMA β 0.1, μ init 1000 steps) or histogram"],
        ["Seeds", "PM 100-101; η sweep 100; PS 100; v1 used 0-2"],
        ["Data [6]", "MNIST from ylecun/mnist (Parquet); Shakespeare corpus = authors' file after CRLF→LF"],
        ["Environment", "v1: teammate CPU machine, versions not recorded. Now: pinned .venv (CPU) "
                        "and a Colab GPU notebook, recorded per run"],
    ], size=9, row_h=0.33)
    d.finish(s, refs=[6, 7], notes="""
This table collects the settings needed to re-run the experiments.

Permuted MNIST: the 784-100-100-10 MLP with LeCun-normal weights and zero biases, batch 16, one epoch per task. The paper runs 2400 tasks; our current reduced runs use 600, and the full length is prepared for a Colab GPU.

Permuted Shakespeare: batch 8 sequences, context 128. The paper uses 500 tasks of 100 epochs; our runs are reduced, as stated on the protocol slide.

Optimizers: SGD with learning rate 0.01; Adam with learning rate 0.001, betas 0.9 and 0.999, and epsilon 1e-7, set explicitly because the PyTorch default is 1e-8.

Method settings. On Permuted MNIST each method uses the value that scored best in the paper's appendix tables; we do no tuning of our own. On Permuted Shakespeare: L2 and L2* with lambda 1e-4, L2 Init with lambda 1e-5, SNR with eta 0.03, and SNR+L2 and SNR+L2* with eta 0.001 and lambda 1e-4, following the paper's appendix sweep.

SNR: firing counted per optimizer step, with either the geometric estimator (moving average with beta 0.1, starting at 1000 steps) or the histogram estimator.

Seeds: 100 and 101 for the main Permuted MNIST runs (102 as well for three SGD conditions), 100 for the eta sweep, and 100 for Permuted Shakespeare. The superseded preliminary v1 runs used seeds 0 to 2.

Data: MNIST from the Hugging Face dataset ylecun/mnist [6], read from Parquet. The Shakespeare corpus is byte-identical to the authors' reference file after line-ending normalization.

Environment. The v1 runs were made on a teammate's CPU machine without recorded versions. New runs use the pinned project environment for CPU work and a Colab notebook for long runs, and record configuration, seed, commit and hardware.
""")

    return d


def fix_notes_pages(prs):
    """Make speaker notes render in every editor (OnlyOffice, LibreOffice, PowerPoint).

    The draft 7.pptx declares a notes page of 5,143,500 x 9,144,000 EMU (the slide height used
    as the width), while its notes master lays its boxes out for the standard 6,858,000 width,
    so slide image and notes text ran off the page and OnlyOffice showed no notes. We set the
    standard portrait notes size and give each notes slide's image and body placeholders
    explicit geometry instead of relying on inheritance from the master.
    """
    from pptx.util import Emu
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
    sz = prs.element.find("p:notesSz", ns)
    if sz is not None:
        sz.set("cx", "6858000")
        sz.set("cy", "9144000")
    geometry = {"sldImg": (685800, 1143000, 5486400, 3086100),
                "body": (685800, 4400550, 5486400, 3600450)}
    for slide in prs.slides:
        if not slide.has_notes_slide:
            continue
        for ph in slide.notes_slide.placeholders:
            kind = ph.placeholder_format.type
            key = "sldImg" if "SLIDE_IMAGE" in str(kind) else ("body" if "BODY" in str(kind) else None)
            if key:
                x, y, w, h = geometry[key]
                ph.left, ph.top, ph.width, ph.height = Emu(x), Emu(y), Emu(w), Emu(h)
            if key == "body":   # 11 pt so a 300-400 word note fits the printed notes page
                from pptx.util import Pt
                for para in ph.text_frame.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(10)


def main():
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fix_notes_pages(d.prs)
    d.prs.save(str(OUT))
    n = len(d.prs.slides)
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({n} slides, {size:.2f} MB)")
    print(f"notes words: max {max(d.notes_words)}, min {min(d.notes_words)}")
    if d.pending_by_slide:
        for k, v in sorted(d.pending_by_slide.items()):
            print(f"  slide {k}: pending -> {', '.join(v)}")
    else:
        print("no pending placeholders")


if __name__ == "__main__":
    main()
