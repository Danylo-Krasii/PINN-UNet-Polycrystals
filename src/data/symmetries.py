"""The 48 symmetries of the cube as operations on (B, C, n, n, n) fields, and the standardization statistics averaged over their channel orbits.

A symmetry acts on the voxel grid and on the channels of a field: the 21 independent entries of the Mandel stiffness and the 36 entries of the
localization tensor A transform as C' = R C R^T, where R is the Mandel rotation of the cube operation. The Mandel order is xx, yy, zz, xy, yz, xz, with the
factor sqrt 2 on the shear components (`MANDEL_F`), so that the 6 x 6 rotation is orthogonal. Fields are transformed raw, before
standardization; the statistics are averaged over the channel orbits of the group, so that a standardized field transforms the same way.
"""
import itertools

import numpy as np
import torch

MANDEL_MAP = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
MANDEL_F = np.array([1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2), np.sqrt(2)])


def to_mandel2(T):
    """(..., 3, 3) -> (..., 6)."""
    return np.stack([MANDEL_F[I] * T[..., i, j] for I, (i, j) in enumerate(MANDEL_MAP)], -1)


def octahedral_group():
    """The 48 signed permutation matrices of the cube, proper and improper, as integer (3, 3) arrays with the identity first."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            Q = np.zeros((3, 3), dtype=int)
            for j in range(3):
                Q[perm[j], j] = signs[j]
            out.append(Q)
    out.sort(key=lambda Q: -int(np.array_equal(Q, np.eye(3, dtype=int))))
    return out


def mandel_rotation(Q):
    """The (6, 6) matrix R with mandel2(Q T Q^T) = R mandel2(T) for symmetric T; orthogonal in Mandel form."""
    R = np.zeros((6, 6))
    for I, (i, j) in enumerate(MANDEL_MAP):
        T = np.zeros((3, 3))
        if i == j:
            T[i, i] = 1.0
        else:
            T[i, j] = T[j, i] = 1.0 / np.sqrt(2)
        R[:, I] = to_mandel2(Q @ T @ Q.T)
    return R


def channel_maps(Q):
    """Linear maps on the 21 upper-triangle channels and on the 36 channels of A for C' = R C R^T."""
    R = mandel_rotation(Q)
    iu, ju = np.triu_indices(6)
    M21 = np.zeros((21, 21))
    for k, (i, j) in enumerate(zip(iu, ju)):
        E = np.zeros((6, 6))
        E[i, j] = E[j, i] = 1.0
        M21[:, k] = (R @ E @ R.T)[iu, ju]
    M36 = np.zeros((36, 36))
    for k in range(36):
        E = np.zeros(36)
        E[k] = 1.0
        M36[:, k] = (R @ E.reshape(6, 6) @ R.T).reshape(36)
    return M21, M36


def spatial_action(Q):
    """Axis permutation and flips that move a voxel field by Q: F'(x) = F(Q^T x). Flip the input axes first, then transpose."""
    p = [int(np.argmax(np.abs(Q[:, j]))) for j in range(3)]
    s = [int(Q[p[j], j]) for j in range(3)]
    inv = [p.index(k) for k in range(3)]
    flips = [j for j in range(3) if s[j] < 0]
    return inv, flips


class Octahedral:
    """The 48 cube symmetries as torch operations; `ops[g]` is (axis permutation, flipped axes, M21, M36)."""

    def __init__(self, dev="cpu"):
        self.ops = []
        for Q in octahedral_group():
            inv, flips = spatial_action(Q)
            M21, M36 = channel_maps(Q)
            self.ops.append((inv, flips, torch.tensor(M21, dtype=torch.float32, device=dev), torch.tensor(M36, dtype=torch.float32, device=dev)))
        self.orbits21 = self._orbits([o[2].cpu().numpy() for o in self.ops])
        self.orbits36 = self._orbits([o[3].cpu().numpy() for o in self.ops])

    @staticmethod
    def _orbits(mats):
        k = mats[0].shape[0]
        adj = np.zeros((k, k), dtype=bool)
        for M in mats:
            adj |= np.abs(M) > 1e-9
        seen, orbits = set(), []
        for i in range(k):
            if i in seen:
                continue
            stack, orb = [i], set()
            while stack:
                a = stack.pop()
                if a in orb:
                    continue
                orb.add(a)
                stack.extend(np.nonzero(adj[a] | adj[:, a])[0].tolist())
            seen |= orb
            orbits.append(sorted(orb))
        return orbits

    @staticmethod
    def orbit_stats(x, orbits):
        """Per-channel mean and standard deviation, then equalized within each orbit: both (1, C, 1, 1, 1)."""
        mu = x.mean((0, 2, 3, 4))
        var = x.var((0, 2, 3, 4))
        for orb in orbits:
            mu[orb] = mu[orb].mean()
            var[orb] = var[orb].mean()
        return mu.view(1, -1, 1, 1, 1), var.sqrt().view(1, -1, 1, 1, 1) + 1e-8

    def apply(self, X, Y, g):
        """Act on the stiffness X (21 channels) and on the target Y (36 channels, or None) by the group element g."""
        inv, flips, M21, M36 = self.ops[g]
        ax = [2 + k for k in flips]
        perm = [0, 1] + [2 + k for k in inv]

        def act(F, M):
            if ax:
                F = torch.flip(F, ax)
            F = F.permute(perm).contiguous()
            return torch.einsum("ij,bjxyz->bixyz", M.to(F.dtype), F)
        return act(X, M21), (None if Y is None else act(Y, M36))
