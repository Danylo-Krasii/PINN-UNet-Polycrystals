# Physics-informed training of a single-pass U-Net for stress localization in 3D polycrystals

A learned surrogate for the stress-localization tensor A(x) of a periodic 3D polycrystal. From the elastic stiffness of every voxel, one pass of a U-Net
predicts the displacement fluctuation of the cell, and a fixed stress layer of the ANSYS SOLID185 element (B-bar form) turns it into A(x). The stress under any applied
mean strain E then follows as sigma(x) = A(x) : E, so one prediction serves every load.

The network is trained with the label loss, with the discrete potential energy of the element problem, which needs no label, or with both. The weight of the energy is
fixed or set during training by one of three rules (gradient-norm balancing, uncertainty weighting, an augmented Lagrangian), and a full-batch L-BFGS phase can follow the
first-order training. A guard returns a run to its best state when the validation error jumps. This repository is the code of the paper; the trained weights and the
initial weights of its draws are published apart from it. [Placeholder: the citation of the paper.]

The corpus has 200 periodic polycrystals on a 32^3 voxel grid for copper and for a synthetic cubic crystal with a Zener ratio of 8 (zener-8). The labels solve the same
discrete problem as ANSYS Mechanical APDL to within 1e-8. On the 40 test volumes, the trained configurations reach mean errors of 3.245 to 3.404 % on copper, against
17.72 % for the prediction without training, and 11.72 and 12.16 % on zener-8, against 43.03 %. The best draws reach 3.152 % (copper) and 11.52 % (zener-8). The test volumes were
also read during the development of the method, so these errors are errors on a development test set.

## Quick start

    uv sync --all-extras                          # environment (docs/INSTALLATION.md)
    # put the volumes in data/ and the weights in checkpoints/ (docs/DATA.md, docs/TRAINING.md)

    uv run infer --config configs/infer/copper.yaml --input data/copper/sve_1160.hdf5 --output outputs/predictions
    uv run train --config configs/train/copper-labels-energy-adaptive.yaml --dry-run
    uv run train --config configs/train/copper-labels-energy-adaptive.yaml
    uv run score --checkpoint outputs/runs/copper-labels-energy-adaptive-s0/checkpoint.pt --sets train val test --out outputs/score.json
    uv run lbfgs --source outputs/runs/copper-labels-energy-adaptive-s0/checkpoint.pt
    uv run pytest

Data and weights: [Placeholder: where the data and the trained weights are obtained.] The data folder is `data/` or the folder in `PINN_UNET_DATA`; the weights folder is
`checkpoints/` or the folder in `PINN_UNET_CHECKPOINTS`.

## Layout

| folder | content |
|---|---|
| `src/` | the code: `models/` (the U-Net), `training/` (the trainer, the loss, the weight rules, the guard, L-BFGS), `inference/`, `solvers/` (the SOLID185 element), `data/` (the volumes, the symmetries), `evaluation/`, and the entry points `train.py` and `infer.py` |
| `configs/` | `train/` and `infer/`: one YAML file for each configuration of the paper |
| `docs/` | how to install, train, infer, the data and the names |
| `tests/` | unit tests with synthetic volumes (`uv run pytest`) |
| `data/`, `checkpoints/` | the volumes and the trained weights; git ignores their contents |

## Configurations

| configuration | loss | peak rate | seeds (copper) |
|---|---|---|---|
| labels-only | label loss | 1.25e-4 | 0, 1, 2 |
| energy-only | energy loss | 2.5e-4 | 1, 2, 3 |
| labels-energy-fixed | labels and energy, fixed weight 3.0454 | 1.25e-4 | 0, 1, 2 |
| labels-energy-adaptive | labels and energy, gradient-norm balancing | 1.25e-4 | 0, 1, 3 |
| labels-energy-uncertainty | labels and energy, uncertainty weighting | 1.25e-4 | 0, 1, 2 |
| labels-energy-lagrangian | energy subject to a bound on the label loss, augmented Lagrangian | 1.25e-4 | 0, 1, 2 |

On zener-8 the paper trained labels-only (seeds 1, 2, 3) and labels-energy-adaptive (seeds 0, 1, 2). The network has 39 input channels and 1,445,106 weights.

The initial weights of a run are the native initialization of torch under its seed, unless `init` names a file. The draws of the paper started from the files
`initial-<material>-s<seed>.pt`, so a run without them is a new draw and not a repeat of the paper ([docs/TRAINING.md](docs/TRAINING.md)).

## Docs

[Installation](docs/INSTALLATION.md) · [Data](docs/DATA.md) · [Training](docs/TRAINING.md) · [Inference](docs/INFERENCE.md) · [Names](docs/NAMES.md)

Author: Danylo Krasiy, NTU "KhPI". License: MIT.
