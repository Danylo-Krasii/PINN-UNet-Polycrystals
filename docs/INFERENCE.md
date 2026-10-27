# Inference

The model predicts A(x), the stress-localization tensor of every voxel of a periodic volume. The stress under any applied mean strain E then follows by one product,
sigma(x) = A(x) : E, so one prediction serves every load.

## Predict A(x)

    uv run infer --config configs/infer/copper.yaml --input data/copper/sve_1160.hdf5 --output outputs/predictions

| config | checkpoint |
|---|---|
| `configs/infer/copper.yaml` | `copper-labels-energy-adaptive-s3.pt`, the best copper draw of the paper (test error 3.152 %) |
| `configs/infer/zener-8.yaml` | `zener-8-labels-energy-adaptive-s1.pt`, the best zener-8 draw of the paper (test error 11.52 %) |

Both run on the CPU in float64. The input is a corpus file (`voxels` and `C_grain`, or `C66`) or an `.npz` file. The output folder gets `pred_<seed>.npz` with the key
`A`, of shape (n, n, n, 6, 6): `A[x, y, z, i, l]` is the Mandel stress component i, in Pa, for a unit Mandel strain l. The Mandel order is xx, yy, zz, xy, yz, xz, with
the factor sqrt 2 on the shear components. An existing output file is replaced, and the command says so. Change an option with `--set`, for example
`--set checkpoint=outputs/runs/copper-labels-only-s0/checkpoint.pt` or `--set dtype=float32`.

## The stress under a load

    import numpy as np

    A = np.load("outputs/predictions/pred_1160.npz")["A"]          # (32, 32, 32, 6, 6)
    E = np.array([1e-3, 0, 0, 0, 0, 0])                           # Mandel strain: 0.1 % along x
    sigma = np.einsum("xyzil,l->xyzi", A, E)                      # Mandel stress of every voxel, in Pa

## What the network computes

The input has 39 channels at every voxel: the 21 independent entries of the stiffness matrix of the voxel, standardized with the statistics of the training volumes, and
the 18 nodal forces b of the six unit strains, divided by their root-mean-square size for each load case. The output is the displacement fluctuation w, three
components for each unit strain, multiplied by the same size and with its spatial mean removed. A fixed layer without weights turns w into A: it takes the element-mean
B-bar stress of the SOLID185 element of w. The output layer starts at zero, so the untrained network returns w = 0 and A = C, the uniform-strain guess.

## The error

The error of a prediction is the pooled relative error

    err = sqrt( sum |A_pred - A_ref|^2 / sum |A_ref|^2 ),

with the sums over the volumes, their voxels and the 36 components of A, computed in double precision on the CPU (`uv run score`, [TRAINING.md](TRAINING.md)). On the 40 test
volumes the uniform-strain guess has the error 17.72 % on copper and 43.03 % on zener-8. A file of initial weights scores the uniform-strain guess.
