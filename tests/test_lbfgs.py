"""The L-BFGS phase: the frozen weight, the chunked loss and two calls on synthetic volumes."""
import json

import numpy as np
import pytest
import torch

from src.data import volumes
from src.models import unet
from src.training import lbfgs, single_pass


def trained(data_dir, tmp_path, cfg="labels-energy-fixed", **extra):
    a = ["--configuration", cfg, "--seed", "1", "--steps", "3", "--device", "cpu", "--out", str(tmp_path / "runs"), "--data-dir", str(data_dir), "--n-train", "4", "--n-val", "2",
         "--batch", "2", "--warmup-steps", "2", "--weight-warmup", "2", "--val-every", "1", "--val-batch", "2", "--peak", "1e-3"]
    single_pass.train(single_pass.parse(a))
    return tmp_path / "runs" / f"copper-{cfg}-s1" / "checkpoint.pt"


def run(data_dir, tmp_path, source, *more):
    a = lbfgs.parse(["--source", str(source), "--data-dir", str(data_dir), "--out", str(tmp_path / "lbfgs"), "--device", "cpu", "--dtype", "float64", "--n-train", "4", "--n-val", "2",
                     "--max-iter", "2", "--max-calls", "2", *more])
    return lbfgs.run(a)


def test_two_calls_keep_the_state_with_the_lowest_validation_error(data_dir, tmp_path):
    source = trained(data_dir, tmp_path)
    summary = run(data_dir, tmp_path, source)
    rows = [json.loads(line) for line in (tmp_path / "lbfgs" / "lbfgs-copper-labels-energy-fixed-s1" / "trace.jsonl").read_text().splitlines()]
    assert [r["call"] for r in rows] == [0, 1, 2] and summary["stop_reason"] == "max-calls" and summary["calls"] == 2
    assert rows[0]["iterations"] == 0 and 1 <= rows[1]["iterations"] <= 2
    assert rows[2]["loss"] <= rows[1]["loss"] <= rows[0]["loss"] + 1e-12
    assert summary["weight"] == pytest.approx(3.0454335833546127) and summary["best_val"] == min(r["val"] for r in rows)
    model, ck = unet.load_checkpoint(tmp_path / "lbfgs" / "lbfgs-copper-labels-energy-fixed-s1" / "checkpoint.pt")
    assert ck["best_step"] == summary["best_call"]
    with pytest.raises(SystemExit):
        run(data_dir, tmp_path, source)


def test_the_volumes_are_those_of_the_source_unless_the_options_name_others(data_dir, tmp_path):
    source = trained(data_dir, tmp_path)
    a = lbfgs.parse(["--source", str(source), "--data-dir", str(data_dir)])
    assert (a.first_seed, a.n_train, a.n_val) == (None, None, None)
    summary = run(data_dir, tmp_path, source)
    assert summary["calls"] == 2


def test_the_loss_is_the_same_for_any_chunk_size(data_dir):
    got = volumes.load_volumes(data_dir / "copper", [1000, 1001, 1002, 1003], labels=True, scales=True)
    torch.manual_seed(0)
    model = unet.Surrogate(volumes.train_stats(got["C"], got["A"])).double()
    X, Y, S = got["C"].double(), got["A"].double(), got["S"]
    for p in model.net.out.parameters():
        torch.nn.init.normal_(p, std=0.02)
    ids = np.arange(4)
    grads = []
    for chunk in (4, 3, 1):
        r = lbfgs.full_batch(model, X, Y, S, ids, 2.5, chunk=chunk)
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        assert r["loss"] == pytest.approx(r["D"] + 2.5 * r["LE"], rel=1e-12)
        if chunk == 4:
            first = r
        assert r["loss"] == pytest.approx(first["loss"], rel=1e-12)
    assert torch.allclose(grads[0], grads[1], rtol=1e-9, atol=1e-12 * float(grads[0].abs().max())) and torch.allclose(grads[0], grads[2], rtol=1e-9, atol=1e-12 * float(grads[0].abs().max()))
    assert lbfgs.full_batch(model, X, Y, None, ids, None, grad=False)["LE"] is None


def test_the_frozen_weight_comes_from_the_source_or_from_the_option():
    base = dict(settings=dict(lambda_f=3.0, weight_warmup=1000), best_step=5000, weight=None)
    assert lbfgs.kept_weight(dict(base, configuration="labels-only")) is None
    assert lbfgs.kept_weight(dict(base, configuration="labels-energy-fixed")) == 3.0
    assert lbfgs.kept_weight(dict(base, configuration="labels-energy-adaptive", weight=3.4)) == 3.4
    assert lbfgs.kept_weight(dict(base, configuration="labels-energy-lagrangian", weight=0.4)) == 0.4
    assert lbfgs.kept_weight(dict(base, configuration="labels-energy-uncertainty", weight=None), 2.5) == 2.5 and lbfgs.kept_weight(dict(base, configuration="labels-energy-fixed"), 2.0) == 2.0
    for bad in (dict(configuration="energy-only"), dict(configuration="labels-energy-uncertainty"), dict(configuration="labels-energy-adaptive", weight=3.0, best_step=999),
                dict(configuration="labels-energy-lagrangian", weight=0.4, best_step=999), dict(configuration="labels-energy-adaptive", weight=float("nan"))):
        with pytest.raises(SystemExit):
            lbfgs.kept_weight(dict(base, **bad))
    for bad in (dict(configuration="energy-only"), dict(configuration="labels-only")):
        with pytest.raises(SystemExit):
            lbfgs.kept_weight(dict(base, **bad), 3.0)
