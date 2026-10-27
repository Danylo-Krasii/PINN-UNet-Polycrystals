"""The score of a checkpoint: the pooled relative error of A against the label in float64 on the CPU.

    CUDA_VISIBLE_DEVICES="" uv run score --checkpoint outputs/runs/copper-labels-only-s0/checkpoint.pt --sets train val test --out outputs/score.json

The model is rebuilt from the checkpoint, cast to float64 and run on the exact float64 stiffness of each volume, so the nodal forces and the stress layer
are in float64 too. For every set the file holds the pooled error sqrt(sum |A_pred - A_ref|^2 / sum |A_ref|^2) over the volumes (the sums over the voxels,
the 36 components and all volumes of the set) and the error of each volume. "train", "val" and "test" are the seed lists of `volumes.seed_lists`: 140, 20 and
40 volumes. The scored material is that of the checkpoint unless `--material` names another. A file of the initial weights scores the uniform-strain guess A = C.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from src import names, settings
from src.data import volumes
from src.inference import predict
from src.models import unet


def score(model, folder, seeds):
    """dict(pooled, per_volume {seed: error}) on the volumes `seeds` of the material folder; the model is float64."""
    pooled = volumes.Pooled()
    per = {}
    for s in seeds:
        C6, H = volumes.read_volume(folder, s)
        per[int(s)] = pooled.add(predict.predict(model, C6), H)
    return dict(pooled=pooled.value, per_volume=per)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=str(settings.DATA), help="the folder that holds the material folders (default: {DATA} of src/settings.py)")
    ap.add_argument("--material", choices=names.MATERIALS, default=None, help="default: the material of the checkpoint")
    ap.add_argument("--sets", nargs="+", default=["val", "test"], choices=names.VOLUME_SETS)
    ap.add_argument("--out", default=None, help="the JSON file of the result (never overwritten)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if out is not None:
        if out.exists():
            raise SystemExit(f"{out} exists: a result file is never overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
    model, ck = unet.load_checkpoint(a.checkpoint, torch.float64)
    material = a.material or ck.get("material", "copper")
    folder = Path(a.data_dir) / material
    lists = volumes.seed_lists()
    t0 = time.time()
    doc = dict(checkpoint=Path(a.checkpoint).name, material=material, configuration=ck.get("configuration"), seed=ck.get("seed"), best_step=ck.get("best_step"),
               dtype="float64", sets={})
    for name in a.sets:
        seeds = lists[name]
        doc["sets"][name] = dict(seeds=[seeds[0], seeds[-1]], **score(model, folder, seeds))
        print(name, round(doc["sets"][name]["pooled"], 6), flush=True)
    doc["seconds"] = time.time() - t0
    if out is not None:
        out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
