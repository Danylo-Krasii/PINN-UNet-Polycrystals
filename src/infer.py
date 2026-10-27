"""Predict the stress-localization field A(x) of one periodic volume from a YAML config:

    uv run infer --config configs/infer/<name>.yaml --input <volume.hdf5> --output <folder>

The key `model:` of the config chooses the predictor (MODELS below); every other key is an option of that predictor, and --set key=value changes or adds
one (as for `uv run train`). The input is a corpus HDF5 file (voxels and C_grain) or an .npz (C66, or voxels and C_grain); the output folder gets
pred_<seed>.npz with the key A, (n, n, n, 6, 6) in Pa per unit Mandel strain, for an input named sve_<seed>.hdf5, and pred_<name>.npz for any other name;
a file that is already there is replaced, and the error stream says so.

    single_pass_unet  checkpoint and dtype (float64 or float32), both required; device (default cpu)
"""

import argparse
import importlib
import re
import sys
import time
from pathlib import Path

import numpy as np

from src import settings

MODELS = {"single_pass_unet": "src.inference.predict"}
SEED = re.compile(r"sve_(\d+)$")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="infer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/infer/<name>.yaml")
    ap.add_argument("--input", required=True, help="volume: corpus HDF5 (voxels, C_grain) or .npz")
    ap.add_argument("--output", required=True, help="folder for pred_<seed>.npz")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change or add one key of the config (repeatable)")
    a = ap.parse_args(argv)
    try:
        cfg = settings.load_config(a.config, a.set)
    except (OSError, ValueError) as e:
        ap.error(str(e))
    model = cfg.pop("model", None)
    if model not in MODELS:
        ap.error(f"model {model!r} is not one of {sorted(MODELS)}")
    t0 = time.time()
    A, info = importlib.import_module(MODELS[model]).run(cfg, a.input)
    stem = Path(a.input).stem
    found = SEED.fullmatch(stem)
    out = Path(a.output) / f"pred_{found.group(1) if found else stem}.npz"
    if out.exists():
        print(f"overwrites {out}", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, A=A)
    print(f"{model}: {a.input} grid {A.shape[:3]} -> {out} ({A.dtype}, {time.time() - t0:.2f} s)" + "".join(f"; {k} {v}" for k, v in info.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
