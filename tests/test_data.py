"""The volumes, the seed lists, the standardization, the symmetries and the pooled error."""
import numpy as np
import pytest
import torch

from src.data import symmetries, volumes
from tests import synth


def test_the_seed_lists_are_the_split_of_the_corpus():
    lists = volumes.seed_lists()
    assert [len(lists[k]) for k in ("train", "val", "test")] == [140, 20, 40]
    assert (lists["train"][0], lists["train"][-1], lists["val"][0], lists["val"][-1], lists["test"][0], lists["test"][-1]) == (1000, 1139, 1140, 1159, 1160, 1199)
    assert volumes.seed_lists(5, 2, 1, 0) == {"train": [5, 6], "val": [7], "test": []}


def test_the_loader_returns_channels_first_float32_arrays_and_the_scale(data_dir):
    got = volumes.load_volumes(data_dir / "copper", [1000, 1001], labels=True, scales=True)
    voxels, C_grain, A = synth.volume(1000)
    assert got["C"].shape == (2, 21, 8, 8, 8) and got["A"].shape == (2, 36, 8, 8, 8) and got["S"].shape == (2, 6)
    assert got["C"].dtype == torch.float32 and got["A"].dtype == torch.float32 and got["S"].dtype == torch.float64
    C6 = C_grain[voxels]
    assert np.array_equal(got["C"][0].numpy(), volumes.upper_triangle(C6).astype(np.float32).transpose(3, 0, 1, 2))
    assert np.array_equal(got["A"][0].numpy(), A.astype(np.float32).reshape(8, 8, 8, 36).transpose(3, 0, 1, 2))
    c6, a = volumes.read_volume(data_dir / "copper", 1000)
    assert np.array_equal(c6, C6) and np.array_equal(a, A)


def test_the_loader_without_labels_never_reads_the_label(tmp_path):
    synth.write_volumes(tmp_path / "copper", [1000, 1001], labels=False)
    got = volumes.load_volumes(tmp_path / "copper", [1000, 1001], labels=False, scales=True)
    assert got["A"] is None and got["S"].shape == (2, 6)
    with pytest.raises(KeyError):
        volumes.load_volumes(tmp_path / "copper", [1000], labels=True)


def test_the_standardization_is_averaged_over_the_channel_orbits(data_dir):
    got = volumes.load_volumes(data_dir / "copper", [1000, 1001, 1002], labels=True)
    stats = volumes.train_stats(got["C"], got["A"])
    assert stats["xm"].shape == (1, 21, 1, 1, 1) and stats["ys"].shape == (1, 36, 1, 1, 1)
    oct_ = symmetries.Octahedral("cpu")
    assert sorted(sum(oct_.orbits21, [])) == list(range(21)) and len(oct_.orbits21) < 21
    for orbit in oct_.orbits21:
        assert len(set(stats["xm"].flatten()[orbit].tolist())) == 1
    diag = got["C"][:, volumes.DIAGONAL].double().mean()
    assert stats["c0"] == float(diag)
    no_labels = volumes.train_stats(got["C"])
    assert float(no_labels["ym"].abs().max()) == 0.0 and float(no_labels["ys"].min()) == 1.0 and torch.equal(no_labels["xm"], stats["xm"])


def test_the_cube_group_has_48_elements_and_the_channel_maps_follow_its_product():
    group = symmetries.octahedral_group()
    assert len(group) == 48 and len({q.tobytes() for q in group}) == 48 and np.array_equal(group[0], np.eye(3, dtype=int))
    oct_ = symmetries.Octahedral("cpu")
    for _, _, M21, M36 in oct_.ops:
        assert torch.allclose(M36 @ M36.T, torch.eye(36), atol=1e-5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        g, h = (int(k) for k in rng.integers(0, 48, size=2))
        product = group[g] @ group[h]
        k = next(i for i, q in enumerate(group) if np.array_equal(q, product))
        assert torch.allclose(oct_.ops[g][2] @ oct_.ops[h][2], oct_.ops[k][2], atol=1e-5)
        assert torch.allclose(oct_.ops[g][3] @ oct_.ops[h][3], oct_.ops[k][3], atol=1e-5)


def test_the_identity_acts_as_the_identity_and_an_unlabelled_batch_has_no_target():
    oct_ = symmetries.Octahedral("cpu")
    x = torch.randn(2, 21, 4, 4, 4)
    y = torch.randn(2, 36, 4, 4, 4)
    xa, ya = oct_.apply(x, y, 0)
    assert torch.allclose(xa, x, atol=1e-6) and torch.allclose(ya, y, atol=1e-6)
    assert oct_.apply(x, None, 5)[1] is None


def test_a_symmetry_moves_the_mean_stiffness_like_a_rotated_tensor(data_dir):
    """The mean of the 21 channels after a cube operation is the operation's channel map applied to the mean before: C' = R C R^T."""
    got = volumes.load_volumes(data_dir / "copper", [1000], labels=False)
    oct_ = symmetries.Octahedral("cpu")
    for g in (1, 7, 30):
        x, _ = oct_.apply(got["C"].double(), None, g)
        before = got["C"].double().mean((0, 2, 3, 4))
        assert torch.allclose(x.mean((0, 2, 3, 4)), oct_.ops[g][2].double() @ before, rtol=1e-4)


def test_the_pooled_error_sums_squares_over_all_volumes():
    p = volumes.Pooled()
    rng = np.random.default_rng(0)
    refs = [rng.normal(size=(4, 4, 4, 6, 6)) for _ in range(3)]
    preds = [r + 0.1 * rng.normal(size=r.shape) for r in refs]
    per = [p.add(a, b) for a, b in zip(preds, refs)]
    expected = np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(preds, refs)) / sum((b ** 2).sum() for b in refs))
    assert p.value == pytest.approx(expected, rel=1e-14)
    assert per[0] == pytest.approx(np.sqrt(((preds[0] - refs[0]) ** 2).sum() / (refs[0] ** 2).sum()), rel=1e-14)
