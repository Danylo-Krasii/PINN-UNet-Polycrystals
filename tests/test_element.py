"""The SOLID185 element operator: known answers that do not need a network."""
import numpy as np
import pytest
import torch

from src.data import volumes
from src.solvers.fem import element_torch, solid185
from src.training import energy
from tests import synth


@pytest.fixture(scope="module")
def stiffness():
    voxels, C_grain, _ = synth.volume(3)
    C6 = torch.from_numpy(C_grain[voxels])[None]
    return C6


def test_the_strain_matrix_reproduces_a_uniform_strain():
    """A displacement u = E x on the corners of a voxel gives the strain E at every Gauss point (the patch test of the element)."""
    B = solid185.strain_matrix()
    E = np.array([[0.1, 0.02, 0.0], [0.02, -0.05, 0.03], [0.0, 0.03, 0.07]])
    corners = solid185.NODE_OFFSETS.astype(float)
    u = (corners @ E.T).reshape(24)
    eps = B @ u
    expected = np.array([E[0, 0], E[1, 1], E[2, 2], 2 * E[0, 1], 2 * E[1, 2], 2 * E[0, 2]])
    assert np.allclose(eps, expected[None], atol=1e-14)


def test_the_element_stiffness_is_symmetric_with_the_translations_in_its_null_space(stiffness):
    el = element_torch.Element(torch.float64)
    ke = el.ke(stiffness)
    assert float((ke - ke.transpose(-1, -2)).abs().max()) < 1e-3
    for c in range(3):
        t = torch.zeros(24, dtype=torch.float64)
        t[c::3] = 1.0
        assert float((ke @ t).abs().max()) < 1e-6 * float(ke.abs().max())
    assert float(torch.linalg.eigvalsh(ke[0, 0]).min()) > -1e-6 * float(ke.abs().max())


def test_a_homogeneous_medium_has_no_nodal_forces_and_the_stress_is_the_stiffness(stiffness):
    el = element_torch.Element(torch.float64)
    C6 = stiffness[:, :1, :1, :1].expand(1, 8, 8, 8, 6, 6).contiguous()
    assert float(el.rhs(C6).abs().max()) < 1e-6 * float(C6.abs().max())
    u = el.gather(torch.zeros(1, 18, 8, 8, 8, dtype=torch.float64))
    A = el.stress_u(C6, u, 8)
    assert torch.equal(A, C6.reshape(1, 8, 8, 8, 36).permute(0, 4, 1, 2, 3))


def test_the_gradient_of_the_energy_is_minus_the_out_of_balance_forces(stiffness):
    """Pi(w) = 1/2 w.K w - b.w = -1/2 w.(b + r) with r = b - K w, so that the gradient of Pi is K w - b = -r."""
    el = element_torch.Element(torch.float64)
    torch.manual_seed(0)
    c21 = torch.from_numpy(np.ascontiguousarray(volumes.upper_triangle(stiffness.numpy()).transpose(0, 4, 1, 2, 3)))
    C6 = element_torch.c21_to_c6(c21)
    ke, b = el.ke(C6), el.rhs(C6)
    w = torch.randn(1, 18, 8, 8, 8, dtype=torch.float64)
    w = (w - w.mean((2, 3, 4), keepdim=True)).requires_grad_()
    r = el.residual_u(ke, b, el.gather(w), 8)
    pi = energy.potential_energy(w, b, r)
    (grad,) = torch.autograd.grad(pi.sum(), w)
    direct = 0.5 * (w * el.apply(ke, w)).reshape(1, 3, 6, -1).sum((1, 3)) - (b * w).reshape(1, 3, 6, -1).sum((1, 3))
    assert torch.allclose(pi, direct, rtol=1e-10)
    r_val = el.residual_u(ke, b, el.gather(w.detach()), 8)
    assert torch.allclose(grad, -r_val, rtol=1e-8, atol=1e-8 * float(r_val.abs().max()))


def test_the_stiffness_channels_round_trip():
    c6 = torch.randn(1, 4, 4, 4, 6, 6, dtype=torch.float64)
    c6 = c6 + c6.transpose(-1, -2)
    c21 = torch.from_numpy(np.ascontiguousarray(volumes.upper_triangle(c6.numpy()).transpose(0, 4, 1, 2, 3)))
    assert torch.equal(element_torch.c21_to_c6(c21), c6)
