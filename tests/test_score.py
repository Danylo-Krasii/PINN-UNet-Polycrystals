"""The scorer and the predictor on synthetic volumes."""
import json

import numpy as np
import pytest
import torch

from src import infer
from src.data import volumes
from src.evaluation import score
from src.inference import predict
from src.models import unet
from tests import synth


@pytest.fixture()
def initial(data_dir):
    torch.manual_seed(0)
    got = volumes.load_volumes(data_dir / "copper", [1000, 1001], labels=True)
    return unet.Surrogate(volumes.train_stats(got["C"], got["A"]))


@pytest.fixture()
def weights_file(initial, tmp_path):
    unet.save_checkpoint(tmp_path / "w.pt", initial, configuration="initial-weights", material="copper", seed=0, best_step=0, weight=None, settings={})
    return tmp_path / "w.pt"


def test_the_untrained_network_scores_the_uniform_strain_guess(initial, data_dir):
    seeds = [1000, 1001, 1002]
    got = score.score(initial.double(), data_dir / "copper", seeds)
    num = den = 0.0
    for seed in seeds:
        voxels, C_grain, A = synth.volume(seed)
        C6 = C_grain[voxels]
        num += ((C6 - A) ** 2).sum()
        den += (A ** 2).sum()
        assert got["per_volume"][seed] == pytest.approx(np.sqrt(((C6 - A) ** 2).sum() / (A ** 2).sum()), rel=1e-13)
    assert got["pooled"] == pytest.approx(np.sqrt(num / den), rel=1e-13)
    assert list(got["per_volume"]) == seeds


def test_the_scorer_writes_a_json_file_and_never_overwrites_it(weights_file, data_dir, tmp_path):
    out = tmp_path / "new folder" / "score.json"
    argv = ["--checkpoint", str(weights_file), "--data-dir", str(data_dir), "--material", "zener-8", "--sets", "val", "--out", str(out)]
    assert score.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["material"] == "zener-8" and doc["dtype"] == "float64" and doc["sets"]["val"]["seeds"] == [1140, 1159] and len(doc["sets"]["val"]["per_volume"]) == 20
    assert doc["sets"]["val"]["pooled"] == pytest.approx(score.score(unet.load_checkpoint(weights_file, torch.float64)[0], data_dir / "zener-8", volumes.seed_lists()["val"])["pooled"])
    with pytest.raises(SystemExit):
        score.main(argv)


def test_the_predictor_gives_the_layout_of_the_labels_and_the_entry_point_names_the_output(weights_file, data_dir, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"model: single_pass_unet\ncheckpoint: {weights_file}\ndtype: float64\ndevice: cpu\n")
    assert infer.main(["--config", str(cfg), "--input", str(data_dir / "copper" / "sve_1000.hdf5"), "--output", str(tmp_path / "out")]) == 0
    A = np.load(tmp_path / "out" / "pred_1000.npz")["A"]
    voxels, C_grain, _ = synth.volume(1000)
    assert A.shape == (8, 8, 8, 6, 6) and A.dtype == np.float64 and np.array_equal(A, C_grain[voxels])
    np.savez(tmp_path / "volume.npz", C66=C_grain[voxels])
    assert infer.main(["--config", str(cfg), "--input", str(tmp_path / "volume.npz"), "--output", str(tmp_path / "out"), "--set", "dtype=float32"]) == 0
    assert np.load(tmp_path / "out" / "pred_volume.npz")["A"].dtype == np.float32


def test_the_predictor_refuses_a_missing_key_a_wrong_key_and_a_wrong_dtype(data_dir):
    path = str(data_dir / "copper" / "sve_1000.hdf5")
    for options in ({"checkpoint": "x"}, {"checkpoint": "x", "dtype": "float64", "steps": 4}, {"checkpoint": "x", "dtype": "float16"}):
        with pytest.raises(SystemExit):
            predict.run(options, path)
