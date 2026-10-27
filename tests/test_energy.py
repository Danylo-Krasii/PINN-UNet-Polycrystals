"""The energy loss, its scale and the tests of energy-only."""
import math

import numpy as np
import pytest
import torch

from src.data import volumes
from src.models import unet
from src.training import energy
from tests import synth


def stiffness(seeds):
    return np.stack([c[v] for v, c, _ in (synth.volume(s) for s in seeds)])


def test_the_scale_is_the_voigt_reuss_gap_and_zero_for_a_homogeneous_volume():
    C6 = stiffness([1, 2])
    S = energy.scale_S(C6)
    assert S.shape == (2, 6) and bool((S > 0).all())
    n = C6.shape[1]
    voigt = C6.mean((1, 2, 3))
    reuss = np.linalg.inv(np.linalg.inv(C6).mean((1, 2, 3)))
    expected = 0.5 * n ** 3 * (np.diagonal(voigt, axis1=-2, axis2=-1) - np.diagonal(reuss, axis1=-2, axis2=-1))
    assert np.allclose(S.numpy(), expected, rtol=1e-10)
    uniform = np.array(np.broadcast_to(C6[0, :1, :1, :1], C6[0].shape))
    assert float(energy.scale_S(uniform).abs().max()) < 1e-9 * float(S.max())


def test_the_loss_is_the_mean_of_the_energy_over_the_scale_and_has_no_gradient_in_the_scale():
    Pi = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0]], dtype=torch.float64, requires_grad=True)
    S = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]], dtype=torch.float64, requires_grad=True)
    loss = energy.loss_E(Pi, S)
    assert float(loss) == -1.0
    loss.backward()
    assert S.grad is None and torch.allclose(Pi.grad, 1.0 / (12 * S.detach()))
    with pytest.raises(ValueError):
        energy.loss_E(Pi, S[:1])


def test_the_untrained_network_has_zero_energy_and_a_residual_of_one():
    seeds = [1, 2]
    C6 = stiffness(seeds)
    c21 = torch.from_numpy(np.ascontiguousarray(volumes.upper_triangle(C6).transpose(0, 4, 1, 2, 3))).float()
    torch.manual_seed(0)
    model = unet.Surrogate(volumes.train_stats(c21))
    out = energy.validate(model, c21, energy.scale_S(C6), batch=1)
    assert out["V_E"] == 0.0 and math.isclose(out["residual"], 1.0, rel_tol=1e-9)
    for p in model.net.out.parameters():
        torch.nn.init.normal_(p, std=0.02)
    out = energy.validate(model, c21, energy.scale_S(C6), batch=2)
    assert out["V_E"] != 0.0 and out["residual"] != 1.0


def test_better_is_a_relative_test_that_is_right_for_a_negative_criterion():
    assert energy.better(-1.0, math.inf, 1e-5) and not energy.better(math.nan, math.inf, 1e-5)
    assert energy.better(-1.0001, -1.0, 1e-5) and not energy.better(-1.000001, -1.0, 1e-5) and not energy.better(-0.9, -1.0, 1e-5)


def test_the_energy_test_and_the_residual_test():
    assert energy.e_guard(0.1, 0.5, 1.0) == (True, "energy")
    assert energy.e_guard(0.0, 0.5, 1.0) == (False, "")
    assert energy.e_guard(math.nan, 0.5, 1.0) == (True, "energy")
    assert energy.e_guard(-0.2, 3.0, 1.0) == (False, "")
    assert energy.e_guard(-0.2, 3.01, 1.0) == (True, "residual")
    assert energy.e_guard(-0.2, math.inf, 1.0) == (True, "residual")
    assert energy.e_guard(-0.2, 7.0, math.inf) == (False, "")
