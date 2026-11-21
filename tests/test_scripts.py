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
