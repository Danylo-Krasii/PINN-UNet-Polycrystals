"""Synthetic volumes for the tests: random grain stiffnesses on a small periodic grid and a label near the stiffness. The files have the layout of docs/DATA.md."""
import h5py
import numpy as np
import torch

N = 8


def stats():
    """Standardization statistics that change nothing (zero mean, unit scale) for a model that the test fills by itself."""
    return dict(xm=torch.zeros(1, 21, 1, 1, 1), xs=torch.ones(1, 21, 1, 1, 1), ym=torch.zeros(1, 36, 1, 1, 1), ys=torch.ones(1, 36, 1, 1, 1), c0=1.0)


def volume(seed, n=N, grains=12):
    """(voxels, C_grain, A): the grain of every voxel, the Mandel stiffness of every grain in Pa, and a label A (n, n, n, 6, 6)."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(grains, 6, 6))
    C_grain = (M @ M.transpose(0, 2, 1) + 3.0 * np.eye(6)) * 2e10
    C_grain = 0.5 * (C_grain + C_grain.transpose(0, 2, 1))
    voxels = rng.integers(0, grains, size=(n, n, n)).astype(np.int32)
    C6 = C_grain[voxels]
    return voxels, C_grain, C6 + rng.normal(size=C6.shape) * 3e9


def write_volumes(folder, seeds, n=N, labels=True):
    folder.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        voxels, C_grain, A = volume(seed, n)
        with h5py.File(folder / f"sve_{seed}.hdf5", "w") as f:
            f["voxels"], f["C_grain"] = voxels, C_grain
            if labels:
                f["A"] = A
    return folder
