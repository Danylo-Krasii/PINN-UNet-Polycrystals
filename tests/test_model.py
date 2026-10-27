"""The single-pass U-Net: its size, its shapes, its start at A = C, its gradients and its files."""
import numpy as np
import pytest
import torch

from src.data import volumes
from src.models import unet
from tests import synth


@pytest.fixture(scope="module")
def batch():
    vols = [synth.volume(s) for s in (1, 2)]
    C6 = np.stack([c[v] for v, c, _ in vols])
    c21 = torch.from_numpy(np.ascontiguousarray(volumes.upper_triangle(C6).transpose(0, 4, 1, 2, 3))).float()
    A36 = torch.from_numpy(np.ascontiguousarray(np.stack([a.reshape(a.shape[:3] + (36,)) for _, _, a in vols]).transpose(0, 4, 1, 2, 3))).float()
    return c21, A36, C6


@pytest.fixture()
def model(batch):
    torch.manual_seed(0)
    return unet.Surrogate(volumes.train_stats(batch[0], batch[1]))


def test_the_network_has_1445106_weights_and_39_input_channels(model):
    assert unet.count_params(model) == 1_445_106
    assert model.net.enc[0][0].weight.shape == (32, 39, 3, 3, 3)
    assert model.net.out.weight.shape == (18, 32, 1, 1, 1)


def test_the_untrained_network_returns_w_zero_and_the_stiffness(model, batch):
    c21, _, C6 = batch
    out = model(c21)
    assert float(out["w"].abs().max()) == 0.0
    assert out["w"].shape == (2, 18, 8, 8, 8) and out["b"].shape == (2, 18, 8, 8, 8) and out["A"].shape == (2, 36, 8, 8, 8)
    expected = torch.from_numpy(C6.reshape(C6.shape[:4] + (36,))).float().permute(0, 4, 1, 2, 3)
    assert torch.allclose(out["A"], expected, rtol=1e-6)


def test_the_output_has_zero_mean_and_the_optional_outputs(model, batch):
    c21 = batch[0]
    for p in model.net.out.parameters():
        torch.nn.init.normal_(p, std=0.05)
    out = model(c21, stress=False, out_of_balance=True)
    assert "A" not in out and out["out_of_balance"].shape == out["w"].shape
    assert float(out["w"].mean((2, 3, 4)).abs().max()) < 1e-5 * float(out["w"].abs().max())
    assert "out_of_balance" not in model(c21)


def test_the_gradient_reaches_the_output_layer_first_and_then_every_weight(model, batch):
    c21, A36, _ = batch
    model.zero_grad()
    ((model(c21)["A"] - model.ym) / model.ys - (A36 - model.ym) / model.ys).pow(2).mean().backward()
    assert float(model.net.out.weight.grad.abs().max()) > 0
    assert float(model.net.enc[0][0].weight.grad.abs().max()) == 0
    torch.optim.SGD(model.parameters(), lr=1e-2).step()
    model.zero_grad()
    ((model(c21)["A"] - model.ym) / model.ys - (A36 - model.ym) / model.ys).pow(2).mean().backward()
    zero = [n for n, p in model.named_parameters() if n != "net.out.bias" and float(p.grad.abs().max()) == 0]
    assert zero == []


def test_the_force_channels_have_unit_size_for_each_load_case(model, batch):
    """The network reads b / c0 divided by its root-mean-square size for each volume and load case, whatever the size of the stiffness."""
    seen = []
    model.net.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    model(batch[0])
    model(batch[0] * 10.0)
    assert len(seen) == 2 and seen[0].shape[1] == 39
    for x in seen:
        rms = unet.load_rms(x[:, 21:])
        assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-5)


def test_a_grid_that_the_pooling_cannot_halve_three_times_is_refused(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 21, 12, 12, 12))


def test_a_checkpoint_round_trips_and_a_file_of_the_initial_weights_loads(model, tmp_path):
    unet.save_checkpoint(tmp_path / "a.pt", model, configuration="labels-only", material="copper", seed=0, best_step=0, weight=None, settings={})
    loaded, ck = unet.load_checkpoint(tmp_path / "a.pt", torch.float64)
    assert ck["configuration"] == "labels-only" and ck["c0"] == model.c0 and next(loaded.parameters()).dtype == torch.float64
    fresh = unet.Surrogate(synth.stats())
    unet.load_initial_weights(fresh, tmp_path / "a.pt")
    assert all(torch.equal(fresh.state_dict()[k], v) for k, v in model.state_dict().items() if k.startswith("net."))
    assert float(fresh.xs.max()) == 1.0 and fresh.c0 == 1.0


def test_a_file_with_another_first_layer_is_refused(model, tmp_path):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["net.enc.0.0.weight"] = torch.zeros(32, 40, 3, 3, 3)
    torch.save(dict(state=state, c0=1.0), tmp_path / "wide.pt")
    with pytest.raises(RuntimeError):
        unet.load_initial_weights(model, tmp_path / "wide.pt")
