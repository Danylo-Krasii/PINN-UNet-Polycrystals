"""The SOLID185 element in its B-bar form on a periodic voxel grid, in torch.

Fields are batched and channels first. A displacement fluctuation w and the nodal forces b have the shape (B, 18, n, n, n), where channel
6 c + l is displacement component c of load case l (the six Mandel unit strains). Node (i, j, k) is the lowest corner of voxel [i, j, k].
The localization tensor A has the shape (B, 36, n, n, n), where channel 6 i + l is stress component i of load case l, and the Mandel
stiffness C6 has the shape (B, n, n, n, 6, 6).

`Element` holds the constants of the element and gives the stiffness of every voxel (`ke`), the nodal forces b of the unit strains (`rhs`), the
nodal forces K w of a fluctuation w (`apply`, `apply_u`), the out-of-balance nodal forces b - K w (`residual_u`) and the stress layer
A = C (I + mean strain of w) (`stress_u`). The `_u` forms take the corner displacements that `gather` returns, so that one gather serves
the stress layer and the out-of-balance forces. The layer has no trainable weights.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

from src.data.symmetries import MANDEL_F
from src.solvers.fem import solid185

OFFSETS = [tuple(int(v) for v in o) for o in solid185.NODE_OFFSETS]


def c21_to_c6(c21):
    """(B, 21, n, n, n) upper-triangle channels -> (B, n, n, n, 6, 6) symmetric Mandel stiffness."""
    c21 = c21.permute(0, 2, 3, 4, 1)
    i, j = np.triu_indices(6)
    out = c21.new_zeros(c21.shape[:-1] + (6, 6))
    out[..., i, j] = c21
    out[..., j, i] = c21
    return out


def a36_to_nnn66(A36):
    """(B, 36, n, n, n) -> (B, n, n, n, 6, 6) with [i, l] = stress i of load case l."""
    B, n = A36.shape[0], A36.shape[2]
    return A36.permute(0, 2, 3, 4, 1).reshape(B, n, n, n, 6, 6)


class Element:
    def __init__(self, dtype=torch.float64, device="cpu"):
        self.dtype, self.device = dtype, device
        B = solid185.strain_matrix()
        F = MANDEL_F

        def t(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

        self.T = t(np.einsum('gia,gjb->ijab', B, B) / 8.0).reshape(36, 576)
        self.Bmean = t(B.mean(0))
        self.Bmean_mandel_t = t((B.mean(0) / F[:, None]).T)
        self.FF = t(np.outer(F, F))
        self.Fv = t(F)

    def gather(self, w):
        """(B, 18, n, n, n) -> (B, n^3, 6, 24), contiguous: the 8 corner displacements of every voxel for every load case."""
        B, n = w.shape[0], w.shape[2]
        wp = Fn.pad(w, (0, 1, 0, 1, 0, 1), mode="circular")
        s = torch.stack([wp[:, :, ox:ox + n, oy:oy + n, oz:oz + n] for ox, oy, oz in OFFSETS], 1)
        return s.reshape(B, 8, 3, 6, n ** 3).permute(0, 4, 3, 1, 2).reshape(B, n ** 3, 6, 24).contiguous()

    def _scatter(self, f, n):
        """(B, n^3, 6, 24) element forces -> (B, 18, n, n, n) summed onto the periodic nodes."""
        B = f.shape[0]
        f = f.reshape(B, n, n, n, 6, 8, 3).permute(5, 0, 6, 4, 1, 2, 3)
        out = sum(torch.roll(f[a], (ox, oy, oz), (-3, -2, -1)) for a, (ox, oy, oz) in enumerate(OFFSETS))
        return out.reshape(B, 18, n, n, n)

    @staticmethod
    def _center(x):
        return x - x.mean((2, 3, 4), keepdim=True)

    def ke(self, C6):
        """(B, n, n, n, 6, 6) Mandel stiffness -> (B, n^3, 24, 24) element stiffness of every voxel (C is symmetrized, so ke is symmetric)."""
        C6 = C6.to(self.dtype)
        C6 = 0.5 * (C6 + C6.transpose(-1, -2))
        CV = (C6 / self.FF).reshape(C6.shape[:4] + (36,))
        return (CV @ self.T).reshape(C6.shape[0], -1, 24, 24)

    def apply_u(self, ke, u, n):
        """K w from the gathered corner displacements u of w: the element forces u ke, summed onto the nodes."""
        return self._scatter(torch.matmul(u, ke), n)

    def apply(self, ke, w):
        """K w, (B, 18, n, n, n)."""
        return self.apply_u(ke, self.gather(w), w.shape[2])

    def rhs(self, C6):
        """The nodal forces b of the six Mandel unit strains, (B, 18, n, n, n): minus the assembled force Bmean^T C_e E, with zero mean."""
        C6 = C6.to(self.dtype)
        B, n = C6.shape[0], C6.shape[1]
        CV = (C6 / self.FF).reshape(B, n ** 3, 6, 6)
        sig = (CV * self.Fv).transpose(-1, -2)
        return self._center(-self._scatter(sig @ self.Bmean, n))

    def residual_u(self, ke, b, u, n):
        """The out-of-balance nodal forces b - K w with the mean removed, from the gathered corner displacements u of w."""
        return self._center(b - self.apply_u(ke, u, n))

    def stress_u(self, C6, u, n):
        """The B-bar stress layer from the gathered corner displacements: A = C (I + mean strain of w), (B, 36, n, n, n)."""
        C6 = C6.to(self.dtype)
        B = C6.shape[0]
        eps = u @ self.Bmean_mandel_t
        A = C6.reshape(B, n ** 3, 6, 6)
        A = A + (A[:, :, :, None, :] * eps[:, :, None, :, :]).sum(-1)
        A = A.reshape(B, n, n, n, 6, 6)
        return A.reshape(A.shape[:4] + (36,)).permute(0, 4, 1, 2, 3)
