# Names

The code, the configs and the documents use one name for each thing.

## Configurations

| name | loss |
|---|---|
| labels-only | the label loss D |
| energy-only | the energy loss L_E; reads no label |
| labels-energy-fixed | D plus L_E at one fixed weight |
| labels-energy-adaptive | D plus L_E at a weight set by gradient-norm balancing |
| labels-energy-uncertainty | D plus L_E at a weight set by uncertainty weighting |
| labels-energy-lagrangian | L_E subject to D <= D_t by an augmented Lagrangian |

## Materials

| name | what it is |
|---|---|
| copper | Zener ratio 3.21 |
| zener-8 | synthetic: the c11 and c12 of copper, c44 = 188.0 GPa, Zener ratio 8.0 |

## Volume sets

| name | seeds | volumes |
|---|---|---|
| training (`train`) | 1000 to 1139 | 140 |
| validation (`val`) | 1140 to 1159 | 20 |
| test (`test`) | 1160 to 1199 | 40; read during the development of the method |

## Terms

| term | meaning |
|---|---|
| D | the label loss: the mean squared error of the standardized A against the label |
| L_E | the energy loss: the mean over the volumes and the load cases of the discrete potential energy of w divided by the Voigt-Reuss gap |
| b | the nodal forces that the six unit strains induce in the heterogeneous cell |
| out-of-balance nodal forces | b - K w, for the prediction w |
| validation test, midpoint test | the rewinds of the guard on the validation error |
| energy test, residual test | the rewinds of the guard of energy-only; the residual is the relative norm of the out-of-balance nodal forces |
| stalled draw | a draw whose best state lies at step 8,000 or earlier and whose steps were clipped more than half of the time |
| pooled relative error | the error of A over a set of volumes ([INFERENCE.md](INFERENCE.md)) |
| tag of a run | `<material>-<configuration>-s<seed>`, the name of its folder |
