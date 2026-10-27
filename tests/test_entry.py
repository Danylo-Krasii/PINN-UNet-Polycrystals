"""`uv run train` (src/train.py), the configs and src/settings.py."""
import shlex
from pathlib import Path

import pytest

from src import names, settings, train
from src.training import single_pass

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "train").glob("*.yaml"))


def write(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return path


def test_every_key_becomes_an_option_with_dashes():
    cfg = dict(model="single_pass_unet", configuration="labels-only", max_rewinds=20, midpoint_guard=True, other=False, init=None, steps=5, tags=["a", "b"])
    assert train.options(cfg) == ["--configuration", "labels-only", "--max-rewinds", "20", "--midpoint-guard", "--steps", "5", "--tags", "a", "b"]
    assert train.options(dict(midpoint_guard=False, other=False)) == ["--no-midpoint-guard"]
    for bad in (dict(a={"b": 1}), dict(a=[]), {1: "x"}):
        with pytest.raises(SystemExit):
            train.options(bad)


def test_an_unknown_model_stops_the_run():
    with pytest.raises(SystemExit):
        train.command(dict(model="kind"))
    with pytest.raises(SystemExit):
        train.command({})
    assert train.command(dict(model="single_pass_unet", seed=3))[1:] == ["-m", "src.training.single_pass", "--seed", "3"]


def test_the_configs_of_the_paper_are_the_eight_runs_and_the_trainer_takes_every_option(capsys):
    assert [p.name for p in CONFIGS] == sorted(f"{m}-{c}.yaml" for m, c in [("copper", c) for c in names.CONFIGURATIONS] + [("zener-8", "labels-only"), ("zener-8", "labels-energy-adaptive")])
    for path in CONFIGS:
        assert train.main(["--config", str(path), "--dry-run"]) == 0
        line = capsys.readouterr().out.strip()
        args = single_pass.parse(shlex.split(line)[3:])
        cfg = settings.load_config(path)
        assert (args.configuration, args.material, args.seed, args.steps, args.patience, args.min_stop_step, args.max_rewinds, args.midpoint_guard) == (
            cfg["configuration"], cfg["material"], cfg["seed"], 96000, 6000, 12000, 20, True)
        assert args.peak == (2.5e-4 if args.configuration == "energy-only" else 1.25e-4) and args.device == "cuda"
        assert args.init == str(settings.CHECKPOINTS / f"initial-{args.material}-s{args.seed}.pt") and args.out == str(settings.ROOT / "outputs" / "runs")
        assert Path(args.data_dir) == settings.DATA
        if args.configuration in ("labels-energy-fixed", "labels-energy-lagrangian"):
            assert args.lambda_f == 3.0454335833546127
        if args.configuration == "labels-energy-lagrangian":
            assert args.label_target == 0.027132353081610314


def test_set_changes_a_key_and_a_open_value_stops_a_real_run(tmp_path, capsys):
    path = write(tmp_path, "model: single_pass_unet\nconfiguration: labels-only\nseed: 0\ndevice: cpu\nlambda_f: '{LAMBDA_F}'\n")
    assert train.main(["--config", str(path), "--dry-run", "--set", "seed=4", "--set", "device=null", "--set", "midpoint_guard=true"]) == 0
    captured = capsys.readouterr()
    assert "--seed 4" in captured.out and "--device" not in captured.out and "--midpoint-guard" in captured.out and "{LAMBDA_F}" in captured.out and "open values" in captured.err
    with pytest.raises(SystemExit):
        train.main(["--config", str(path)])
    with pytest.raises(SystemExit):
        train.main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"])


def test_the_placeholders_expand_to_the_paths_of_the_settings():
    assert settings.expand("{ROOT}/outputs") == f"{settings.ROOT}/outputs" and settings.expand("{DATA}") == str(settings.DATA)
    assert settings.expand(["{CHECKPOINTS}/a.pt", 3]) == [f"{settings.CHECKPOINTS}/a.pt", 3] and settings.expand("{seed}") == "{seed}"
    assert settings.DATA == settings.ROOT / "data" or "PINN_UNET_DATA" in __import__("os").environ


def test_the_overrides_of_load_config(tmp_path):
    path = write(tmp_path, "a: 1\nb: x\n")
    assert settings.load_config(path, ["a=true", "b=null", "c=text", "d=False"]) == {"a": True, "c": "text", "d": False}
    for bad in (["novalue"], ["=3"]):
        with pytest.raises(ValueError):
            settings.load_config(path, bad)
    with pytest.raises(ValueError):
        settings.load_config(write(tmp_path, "- a\n- b\n"))
    with pytest.raises(SystemExit):
        settings.check_keys("m", {"a": 1, "z": 2}, ("a", "b"), ("a",))
    settings.check_keys("m", {"a": 1}, ("a", "b"), ("a",))


def test_the_command_trains_from_a_config(data_dir, tmp_path):
    """`uv run train --config ...` calls the trainer in a process of its own: two steps of copper-labels-only on the synthetic volumes."""
    sets = dict(device="cpu", data_dir=data_dir, out=tmp_path, steps=2, init="null", n_train=4, n_val=2, batch=2, val_every=1, val_batch=2, warmup_steps=1)
    argv = ["--config", str(ROOT / "configs/train/copper-labels-only.yaml")] + [a for k, v in sets.items() for a in ("--set", f"{k}={v}")]
    assert train.main(argv) == 0
    assert sorted(p.name for p in (tmp_path / "copper-labels-only-s0").iterdir()) == ["checkpoint.pt", "summary.json", "trace.jsonl"]
