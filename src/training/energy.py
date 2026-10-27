"""The discrete potential energy of the element problem, the energy loss L_E, its scale S, and the tests of energy-only training.

For volume v and Mandel load case j the potential energy of a fluctuation w is Pi_vj(w) = 1/2 w_j.K w_j - b_j.w_j = -1/2 w_j.(b_j + r_j),
where b are the nodal forces of the unit strain and r = b - K w the out-of-balance nodal forces, in Pa times the voxel volume. Its minimum is the
solution of the element problem. The scale S_vj = 1/2 n^3 (<C>_jj - (<C^-1>^-1)_jj) is the gap between the Voigt and the Reuss stiffness of the
volume for the load case; it follows from the stiffness alone. L_E is the mean over the volumes and the six load cases of Pi / S and needs no label.

Energy-only training validates with V_E, the mean of Pi / S of the prediction in float64, and with the residual, the mean of the relative norms
|b - K w| / |b| of the out-of-balance nodal forces of the prediction.
"""
import math

import numpy as np
import torch

from src.solvers.fem import element_torch

THR_PLATEAU = 5.2e-6
THR_BEST = 1.8e-5
GUARD_FACTOR = 3.0
VAL_DTYPE = torch.float64
_ELEMENTS = {}


def scale_S(C6):
    """(B, 6) float64: S_vj = 1/2 n^3 ((<C>)_jj - ((<C^-1>)^-1)_jj) per volume and Mandel load case, from the stiffness C6 (B, n, n, n, 6, 6) in Pa."""
    C = torch.as_tensor(C6).to(torch.float64)
    if C.dim() == 5:
        C = C[None]
    n = C.shape[1]
    voigt = C.mean((1, 2, 3))
    reuss = torch.linalg.inv(torch.linalg.inv(C).mean((1, 2, 3)))
    return 0.5 * n ** 3 * (torch.diagonal(voigt, dim1=-2, dim2=-1) - torch.diagonal(reuss, dim1=-2, dim2=-1))


def potential_energy(w, b, out_of_balance):
    """(B, 6) float64: Pi_vj(w) = -1/2 sum over the nodes and components of w (b + r), with r the out-of-balance nodal forces."""
    f64 = torch.float64
    B = b.shape[0]
    return -0.5 * (w.to(f64) * (b.to(f64) + out_of_balance.to(f64))).reshape(B, 3, 6, -1).sum((1, 3))


def loss_E(Pi, S):
    """0-d float64: the mean over the volumes and the 6 load cases of Pi / S; S (B, 6) float64 is data and carries no gradient."""
    if S.shape != Pi.shape:
        raise ValueError(f"S has shape {tuple(S.shape)} and Pi has shape {tuple(Pi.shape)}")
    return (Pi / S.detach().to(device=Pi.device, dtype=Pi.dtype)).mean()


def element64(device="cpu"):
    """The element in the validation dtype (float64), one per device."""
    key = (VAL_DTYPE, torch.device(device))
    if key not in _ELEMENTS:
        _ELEMENTS[key] = element_torch.Element(VAL_DTYPE, device)
    return _ELEMENTS[key]


def _energy_residual(el, c21, w):
    """(Pi, |b - K w| / |b|), both (B, 6) in the dtype of `el`, for the problem of the stiffness c21 (B, 21, n, n, n) and the field w."""
    C = element_torch.c21_to_c6(c21.to(el.dtype))
    ke, b = el.ke(C), el.rhs(C)
    w = w.to(el.dtype)
    Kw = el.apply(ke, w)
    B = w.shape[0]

    def red(x):
        return x.reshape(B, 3, 6, -1).sum((1, 3))

    r = b - Kw
    r = r - r.mean((2, 3, 4), keepdim=True)
    return 0.5 * red(w * Kw) - red(b * w), (red(r * r) / red(b * b)).sqrt()


@torch.no_grad()
def validate(model, c21, S, batch=4):
    """dict(V_E, residual) on the volumes c21 (V, 21, n, n, n) with the scales S (V, 6): V_E is the mean over the volumes and the 6 load cases
    of Pi / S of the prediction, residual the mean of the ratios |b - K w| / |b|, both evaluated in float64 for the stiffness that the network reads."""
    S = S.to(c21.device)
    el = element64(c21.device)
    pn, res = [], []
    for i in range(0, c21.shape[0], batch):
        x = c21[i:i + batch]
        Pi, ratio = _energy_residual(el, x, model(x, stress=False)["w"])
        pn.append((Pi / S[i:i + batch].to(Pi.dtype)).double().cpu())
        res.append(ratio.double().cpu())
    pn, res = torch.cat(pn).numpy(), torch.cat(res).numpy()
    return dict(V_E=np.float64(pn.mean()), residual=np.float64(res.mean()))


def better(v, best, thr):
    """v < best - thr * |best|, the relative test that is right for a negative criterion; any finite v beats best = +inf, a non-finite v beats nothing."""
    v, best = float(v), float(best)
    if not math.isfinite(v):
        return False
    if best == math.inf:
        return True
    return bool(math.isfinite(best) and v < best - thr * abs(best))


def e_guard(V_E, residual, lowest_residual):
    """(fire, cause): fires with cause "energy" (the energy test) if V_E is non-finite or above 0, else with cause "residual" (the residual test) if the
    residual is non-finite or above 3 times the lowest residual of the validations so far (inf before the first)."""
    V_E, residual, lowest = float(V_E), float(residual), float(lowest_residual)
    if not math.isfinite(V_E) or V_E > 0.0:
        return True, "energy"
    if not math.isfinite(residual) or residual > GUARD_FACTOR * lowest:
        return True, "residual"
    return False, ""


def load_perm(oct_, g):
    """(6,) long: S of the volume acted on by the cube operation g is S[..., load_perm(oct_, g)]; read off the diagonal rows of M21."""
    M21 = oct_.ops[g][2].double()
    i, j = np.triu_indices(6)
    diag = [k for k in range(21) if i[k] == j[k]]
    perm = []
    for k in diag:
        hit = (M21[k].abs() > 1e-9).nonzero().flatten().tolist()
        if len(hit) != 1 or hit[0] not in diag:
            raise ValueError(f"row {k} of M21 of operation {g} is not a signed permutation of the diagonal channels")
        perm.append(diag.index(hit[0]))
    return torch.tensor(perm)
