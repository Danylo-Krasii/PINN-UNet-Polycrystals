"""The volumes, the seed lists, the standardization and the pooled error.

A material folder holds one HDF5 file per volume, `sve_<seed>.hdf5`, with the datasets `voxels` (n, n, n) grain index of every voxel, `C_grain`
(number of grains, 6, 6) Mandel stiffness of every grain in Pa, and `A` (n, n, n, 6, 6) the label of the localization tensor, A[x, y, z, i, l] =
stress component i of load case l in Pa per unit Mandel strain. The seeds 1000 to 1199 are split into 140 training (1000 to 1139), 20 validation
(1140 to 1159) and 40 test volumes (1160 to 1199). The energy loss and its scale read `voxels` and `C_grain` only.

The error of a prediction is the pooled relative error sqrt(sum |A_pred - A_ref|^2 / sum |A_ref|^2) over the voxels, the 36 components and all
volumes of a list, with the sums in float64.
"""
from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.symmetries import Octahedral
from src.training import energy

FIRST, N_TRAIN, N_VAL, N_TEST = 1000, 140, 20, 40
DIAGONAL = [k for k, (i, j) in enumerate(zip(*np.triu_indices(6))) if i == j]


def seed_lists(first=FIRST, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST):
    """{"train", "val", "test"}: consecutive seed lists in this order, from `first`."""
    a, b = first + n_train, first + n_train + n_val
    return {"train": list(range(first, a)), "val": list(range(a, b)), "test": list(range(b, b + n_test))}


def volume_path(folder, seed):
    return Path(folder) / f"sve_{int(seed)}.hdf5"


def upper_triangle(C6):
    """The 21 independent Mandel components, (..., 6, 6) -> (..., 21)."""
    i, j = np.triu_indices(6)
    return C6[..., i, j]


def read_volume(folder, seed):
    """(C6, A), both float64 (n, n, n, 6, 6): the Mandel stiffness of the voxels and the label."""
    with h5py.File(volume_path(folder, seed), "r") as f:
        return f["C_grain"][...][f["voxels"][...]], f["A"][...].astype(np.float64)


def load_volumes(folder, seeds, labels=True, scales=False):
    """dict(seeds, C, A, S): C (V, 21, n, n, n) and A (V, 36, n, n, n) float32 channels first, S (V, 6) float64 the scales of the energy loss.
    A is None without `labels` (the file's `A` is not read), S is None without `scales`."""
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("no seeds")
    C = A = S = None
    for i, seed in enumerate(seeds):
        with h5py.File(volume_path(folder, seed), "r") as f:
            voxels, grains = f["voxels"][...], f["C_grain"][...]
            label = f["A"][...] if labels else None
        C6 = grains[voxels].astype(np.float64)
        if C is None:
            C = torch.empty((len(seeds), 21) + C6.shape[:3], dtype=torch.float32)
            A = torch.empty((len(seeds), 36) + C6.shape[:3], dtype=torch.float32) if labels else None
            S = torch.empty(len(seeds), 6, dtype=torch.float64) if scales else None
        C[i] = torch.from_numpy(upper_triangle(C6).astype(np.float32).transpose(3, 0, 1, 2).copy())
        if labels:
            A[i] = torch.from_numpy(label.astype(np.float32).reshape(label.shape[:3] + (36,)).transpose(3, 0, 1, 2).copy())
        if scales:
            S[i] = energy.scale_S(C6)[0]
    if scales and not bool((torch.isfinite(S) & (S > 0)).all()):
        raise ValueError("the scale S of the energy loss is not finite and positive for some volume")
    return dict(seeds=seeds, C=C, A=A, S=S)


def train_stats(C21, A36=None):
    """The standardization of the training volumes (channels-first arrays): xm, xs (1, 21, 1, 1, 1), ym, ys (1, 36, 1, 1, 1) and the stiffness scale c0,
    the mean of the six diagonal Mandel entries in Pa. Without labels ym = 0 and ys = 1 carry no statistic of A."""
    oct_ = Octahedral("cpu")
    xm, xs = Octahedral.orbit_stats(C21, oct_.orbits21)
    c0 = float(C21[:, DIAGONAL].double().mean())
    if A36 is None:
        return dict(xm=xm, xs=xs, ym=torch.zeros(1, 36, 1, 1, 1), ys=torch.ones(1, 36, 1, 1, 1), c0=c0)
    ym, ys = Octahedral.orbit_stats(A36, oct_.orbits36)
    return dict(xm=xm, xs=xs, ym=ym, ys=ys, c0=c0)


def _np64(x):
    return x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64)


def sums(P, R):
    """(sum |P - R|^2, sum |R|^2) in float64."""
    P, R = _np64(P), _np64(R)
    return float(((P - R) ** 2).sum()), float((R ** 2).sum())


class Pooled:
    """Running pooled relative error over volumes."""

    def __init__(self):
        self.num = self.den = 0.0

    def add(self, P, R):
        n, d = sums(P, R)
        self.num += n
        self.den += d
        return float(np.sqrt(n / d))

    @property
    def value(self):
        return float(np.sqrt(self.num / self.den))
