"""The trainer: a short run of every configuration on synthetic volumes, the files it writes, and the quantities it reports."""
import json

import pytest
import torch

from src.data import volumes
from src.models import unet
from src.names import CONFIGURATIONS
from src.training import balance, single_pass
from tests import synth


def argv(data_dir, out, cfg, steps=3, **extra):
    a = ["--configuration", cfg, "--seed", "1", "--steps", str(steps), "--device", "cpu", "--out", str(out), "--data-dir", str(data_dir), "--n-train", "4", "--n-val", "2",
         "--batch", "2", "--warmup-steps", "2", "--weight-warmup", "2", "--val-every", "1", "--val-batch", "2", "--balance-every", "2", "--sched-patience", "5"]
    if cfg == "labels-energy-lagrangian":
        a += ["--label-target", "0.1"]
    for k, v in extra.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return a


@pytest.mark.parametrize("cfg", CONFIGURATIONS)
def test_a_short_run_writes_its_files(cfg, data_dir, tmp_path):
    summary = single_pass.train(single_pass.parse(argv(data_dir, tmp_path, cfg)))
    run = tmp_path / f"copper-{cfg}-s1"
    assert sorted(p.name for p in run.iterdir()) == ["checkpoint.pt", "summary.json", "trace.jsonl"]
    assert summary["steps"] == 3 and summary["params"] == 1_445_106 and summary["stop_reason"] is None and isinstance(summary["stalled"], bool)
    rows = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    assert rows and all(r["step"] <= 3 for r in rows)
    assert json.loads((run / "summary.json").read_text())["best_step"] == summary["best_step"]
    _, ck = unet.load_checkpoint(run / "checkpoint.pt")
    assert ck["configuration"] == cfg and ck["best_step"] == summary["best_step"] and ck["material"] == "copper" and "data_dir" not in ck["settings"]
    with pytest.raises(SystemExit):
        single_pass.train(single_pass.parse(argv(data_dir, tmp_path, cfg)))


def test_a_run_uses_the_label_only_where_the_configuration_has_labels(data_dir, tmp_path, monkeypatch):
    seen = []
    real = volumes.load_volumes
    monkeypatch.setattr(volumes, "load_volumes", lambda folder, seeds, labels=True, scales=False: seen.append((labels, scales)) or real(folder, seeds, labels, scales))
    for cfg in ("labels-only", "energy-only", "labels-energy-fixed"):
        single_pass.train(single_pass.parse(argv(data_dir, tmp_path, cfg, steps=1)))
    assert seen == [(True, False), (False, True), (True, True)]


def test_the_frozen_weight_of_the_checkpoint_is_the_weight_of_the_trace_row_at_the_kept_step(data_dir, tmp_path):
    for cfg, key in (("labels-energy-adaptive", "lam"), ("labels-energy-uncertainty", "lam_applied"), ("labels-energy-lagrangian", "lam_applied")):
        summary = single_pass.train(single_pass.parse(argv(data_dir, tmp_path, cfg, steps=6, weight_warmup=2)))
        rows = [json.loads(line) for line in (tmp_path / f"copper-{cfg}-s1" / "trace.jsonl").read_text().splitlines()]
        kept = [r for r in rows if "event" not in r and r["step"] == summary["best_step"]]
        if summary["best_step"] == 0:
            assert summary["weight"] is None
            continue
        assert len(kept) == 1 and kept[0][key] == summary["weight"] >= 0


def test_the_initial_weights_are_those_of_the_file_and_the_name_takes_the_material_and_the_seed(data_dir, tmp_path):
    torch.manual_seed(5)
    donor = unet.Surrogate(synth.stats())
    unet.save_checkpoint(tmp_path / "initial-copper-s1.pt", donor, configuration="initial-weights", material="copper", seed=1, best_step=0, weight=None, settings={})
    a = single_pass.parse(argv(data_dir, tmp_path / "runs", "labels-only", steps=1, peak=1e-30, init=tmp_path / "initial-{material}-s{seed}.pt"))
    assert a.init == str(tmp_path / "initial-copper-s1.pt")
    single_pass.train(a)
    _, ck = unet.load_checkpoint(tmp_path / "runs" / "copper-labels-only-s1" / "checkpoint.pt")
    assert ck["best_step"] == 0
    kept = torch.load(tmp_path / "runs" / "copper-labels-only-s1" / "checkpoint.pt", weights_only=True)["state"]
    assert all(torch.equal(kept[k], v) for k, v in donor.state_dict().items() if k.startswith("net."))


def test_the_options_that_do_not_belong_to_a_configuration_are_refused(data_dir, tmp_path):
    for bad in (["--configuration", "labels-only", "--lambda-f", "3"], ["--configuration", "labels-only", "--label-target", "0.1"], ["--configuration", "energy-only", "--guard", "2"],
                ["--configuration", "labels-energy-lagrangian"], ["--configuration", "labels-energy-fixed", "--uncertainty-lr", "0.1"], ["--configuration", "labels-only", "--clip", "0"],
                ["--configuration", "labels-only", "--clip", "nan"], ["--configuration", "labels-only", "--guard", "0"], ["--configuration", "labels-energy-adaptive", "--balance-alpha", "1.5"]):
        with pytest.raises(SystemExit):
            single_pass.parse(bad)


def test_the_defaults_are_the_recipe_of_the_paper():
    a = single_pass.parse(["--configuration", "labels-energy-fixed"])
    assert (a.steps, a.patience, a.min_stop_step, a.max_rewinds, a.midpoint_guard, a.batch, a.peak, a.warmup_steps, a.lambda_f) == (96000, 6000, 12000, 20, True, 8, 1.25e-4, 1000, 3.0454335833546127)
    assert single_pass.parse(["--configuration", "energy-only"]).peak == 2.5e-4
    assert single_pass.parse(["--configuration", "labels-only", "--no-midpoint-guard", "--patience", "0"]).patience is None
    assert single_pass.parse(["--configuration", "labels-energy-uncertainty"]).uncertainty_lr == balance.UNCERTAINTY_LR
