# Data

The corpus holds periodic polycrystals on a 32^3 voxel grid, for two cubic materials. The files are not in git. [Placeholder: where the data are obtained.]
Put them under `data/`, or point `PINN_UNET_DATA` at another folder.

    data/copper/sve_1000.hdf5 ... sve_1199.hdf5
    data/zener-8/sve_1000.hdf5 ... sve_1199.hdf5

## The volumes

A volume is a periodic Voronoi polycrystal of 1,638 grains, about 20 voxels per grain, with grain orientations drawn uniformly from the rotation group. The two
materials share the geometry and the orientations of each volume (the file `sve_<seed>.hdf5` of one material has the same grains as that of the other).

| material | c11, GPa | c12, GPa | c44, GPa | Zener ratio 2 c44 / (c11 - c12) |
|---|---|---|---|---|
| copper | 168.4 | 121.4 | 75.4 | 3.21 |
| zener-8 (synthetic) | 168.4 | 121.4 | 188.0 | 8.0 |

## The file

One HDF5 file for each volume, named `sve_<seed>.hdf5`. The code reads three datasets and nothing else.

| dataset | shape | dtype | content |
|---|---|---|---|
| `voxels` | (32, 32, 32) | int32 | the grain index of each voxel, zero-based; it indexes `C_grain`; the axes are x, y, z |
| `C_grain` | (1638, 6, 6) | float64 | the stiffness of each grain in Mandel notation, in Pa |
| `A` | (32, 32, 32, 6, 6) | float64 | the label, the localization tensor; `A[x, y, z, i, l]` is the stress component i in Pa for a unit Mandel strain l |

The Mandel order is xx, yy, zz, xy, yz, xz, with the factor sqrt 2 on the shear components. The energy loss and its scale read `voxels` and `C_grain` only, so the
configuration energy-only needs no label. The stress under a macroscopic strain E follows from `sigma(x) = A(x) : E` ([INFERENCE.md](INFERENCE.md)).

## The labels

The labels solve the discrete problem of the eight-node SOLID185 element of ANSYS in its B-bar form (KEYOPT(2) = 0), one element per voxel, under the six Mandel unit
strains. A matrix-free conjugate-gradient solver in double precision with a Fourier preconditioner on the same element runs to a relative tolerance of 1e-10. The
localization tensor of a voxel is the mean stress over the eight Gauss points of its element. On ten volumes of each material, five of the test set and five further
volumes that the corpus does not hold, with six load cases each, the pooled relative difference of A from ANSYS Mechanical APDL 2025 R2 was at most 9.78e-9, so the
labels solve the same discrete problem as ANSYS.

## The split

Each material has 200 volumes, seeds 1000 to 1199, in a fixed split.

| set | seeds | volumes |
|---|---|---|
| training | 1000 to 1139 | 140 |
| validation | 1140 to 1159 | 20 |
| test | 1160 to 1199 | 40 |

The test volumes were also read during the development of the method, so they form a development test set and not an untouched one. The seed of a training run
changes the initial weights and the draws of the batches, not the split.
