"""Checks for the experiment entry points: option parsing, run naming and sweep expansion."""
from scripts.run_experiments import condition_name, parse_args
from scripts.sweep import expand, output_path, to_cli


def test_to_cli_handles_flags_and_values():
    assert to_cli({"layer_norm": True, "lam": 0.001, "penalty_mask": "attention"}) == [
        "--layer-norm", "--lam", "0.001", "--penalty-mask", "attention"]
    assert to_cli({"layer_norm": False}) == []


def test_benchmark_defaults():
    pm = parse_args(["--benchmark", "pm"])
    ps = parse_args(["--benchmark", "ps"])
    assert (pm.optimizer, pm.lr, pm.epochs, pm.batch_size) == ("sgd", 0.01, 1, 16)
    assert (ps.optimizer, ps.lr, ps.epochs, ps.batch_size) == ("adam", 1e-3, 100, 8)


def test_condition_names_distinguish_masks_and_composition():
    a = parse_args(["--reset", "snr", "--eta", "0.001", "--penalty", "l2", "--lam", "1e-4"])
    b = parse_args(["--reset", "snr", "--eta", "0.001", "--penalty", "l2", "--lam", "1e-4",
                    "--penalty-mask", "attention"])
    assert condition_name(a) == "snr_eta0.001+l2_lam0.0001"
    assert condition_name(b) == "snr_eta0.001+l2star_lam0.0001"
    assert condition_name(parse_args(["--layer-norm"])) == "ln"


def test_sweep_expands_seed_major(tmp_path):
    cfg = tmp_path / "x.yaml"
    cfg.write_text("common: {tasks: 3}\nseeds: [1, 2]\nruns:\n  - {reset: none}\n  - {reset: snr}\n")
    names = [output_path(argv).name for argv in expand(cfg)]
    assert names == ["pm_none_sgd_seed1_tasks3.csv", "pm_snr_eta0.01_sgd_seed1_tasks3.csv",
                     "pm_none_sgd_seed2_tasks3.csv", "pm_snr_eta0.01_sgd_seed2_tasks3.csv"]


def test_experiment_configs_parse():
    for cfg in ("main", "sensitivity", "shakespeare"):
        jobs = expand(f"configs/experiments/{cfg}.yaml")
        assert jobs and all(parse_args(argv) for argv in jobs)


def _fake_run(dirpath, name, meta, rows):
    import json
    import pandas as pd
    pd.DataFrame(rows).to_csv(dirpath / f"{name}.csv", index=False)
    (dirpath / f"{name}.json").write_text(json.dumps(meta))


def test_summary_keeps_sweep_and_eval_apart(tmp_path, monkeypatch):
    """Review finding I2: eval runs at another task count leaked into the eta sweep table."""
    import pandas as pd
    import scripts.summarize_results as sr
    metrics, tables = tmp_path / "m", tmp_path / "t"
    metrics.mkdir()
    base = dict(benchmark="pm", reset="snr", penalty="none", penalty_mask="all", layer_norm=False,
                optimizer="sgd", seed=100, estimator="geometric", lam=0.001)
    rows = [{"task": t, "online_accuracy": 0.5, "resets": 1} for t in range(10)]
    _fake_run(metrics, "a", dict(base, eta=0.005, tasks=10, phase="eval"), rows)
    _fake_run(metrics, "b", dict(base, eta=0.005, tasks=5, phase="sweep"), rows[:5])
    _fake_run(metrics, "c", dict(base, eta=0.005, tasks=5, phase="smoke", reset="snr"), rows[:5])
    monkeypatch.setattr(sr, "METRICS", metrics)
    monkeypatch.setattr(sr, "TABLES", tables)
    sr.main()
    eta = pd.read_csv(tables / "eta_sweep.csv")
    assert eta["n_tasks"].tolist() == [5] and eta["n_seeds"].tolist() == [1]
    pm = pd.read_csv(tables / "pm_main.csv")
    assert pm["n_tasks"].tolist() == [10]


def test_combined_interventions_are_labelled():
    from scripts.summarize_results import pm_method
    meta = dict(reset="snr", penalty="l2", layer_norm=False, eta=0.01, lam=0.001)
    assert pm_method(meta)[0] == "SNR+L2"
