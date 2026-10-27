"""Prediction of the localization tensor A(x) of one periodic volume with a checkpoint (the predictor of `uv run infer`, src/infer.py).

The input is an HDF5 file with `voxels` and `C_grain` (or `C66`), or an .npz file with `C66`, or with `voxels` and `C_grain`; the stiffness is the Mandel
stiffness in Pa. A is (n, n, n, 6, 6) in Pa per unit Mandel strain, A[x, y, z, i, l] = stress component i of load case l, the layout of the labels. The stress
under a macroscopic strain E (Mandel, 6 components) is sigma = A : E.
"""
from pathlib import Path

import h5py
import numpy as np
import torch

from src import settings
from src.data import volumes
from src.models import unet
from src.solvers.fem import element_torch

OPTIONS = ("checkpoint", "dtype", "device")
DTYPES = {"float64": torch.float64, "float32": torch.float32}


def read_stiffness(path):
    """The Mandel stiffness of the voxels, (n, n, n, 6, 6), from an HDF5 or .npz file."""
    opener = np.load if Path(path).suffix == ".npz" else (lambda p: h5py.File(p, "r"))
    with opener(path) as f:
        if "C66" in f:
            return f["C66"][...]
        if "voxels" in f and "C_grain" in f:
            return f["C_grain"][...][f["voxels"][...]]
    raise SystemExit(f"{path} holds neither C66 nor voxels and C_grain")


def predict(model, C66, device="cpu"):
    """A (n, n, n, 6, 6) in the dtype of the model, from the Mandel stiffness C66 (n, n, n, 6, 6) in Pa."""
    dtype = next(model.parameters()).dtype
    c21 = torch.from_numpy(volumes.upper_triangle(np.asarray(C66))).to(dtype).permute(3, 0, 1, 2)[None].to(device)
    with torch.no_grad():
        A = model(c21)["A"]
    return element_torch.a36_to_nnn66(A)[0].cpu().numpy()


def run(options, path):
    """The prediction for `uv run infer`: A of the volume in `path` and a dict of what was used.

    The options are those of the config: checkpoint and dtype (float64 or float32), both required, and device (default cpu).
    """
    settings.check_keys("single_pass_unet", options, OPTIONS, ("checkpoint", "dtype"))
    if options["dtype"] not in DTYPES:
        raise SystemExit(f"dtype {options['dtype']!r} is not one of {sorted(DTYPES)}")
    device = options.get("device", "cpu")
    model, ck = unet.load_checkpoint(options["checkpoint"], DTYPES[options["dtype"]], device)
    return predict(model, read_stiffness(path), device), dict(checkpoint=options["checkpoint"], configuration=ck.get("configuration"), dtype=options["dtype"])
