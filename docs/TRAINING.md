# Training

One command trains the model. A YAML file in `configs/train/` holds the options of one configuration of the paper.

    uv run train --config configs/train/copper-labels-energy-adaptive.yaml --dry-run    # print the exact command, start nothing
    uv run train --config configs/train/copper-labels-energy-adaptive.yaml              # train (needs a GPU)

The command calls `python -m src.training.single_pass` with the options of the file. The key `model: single_pass_unet` chooses the trainer, and every other key is an option
of it, with the underscores of the key for the dashes of the option: `max_rewinds: 20` gives `--max-rewinds 20`. A key that the file does not set keeps the default of
the trainer. Change a key on the command line with `--set`, for example `--set seed=1`, `--set device=cpu` or `--set steps=200`; `--set key=null` removes a key. A false value gives no option,
except `midpoint_guard: false`, which gives `--no-midpoint-guard`; the cap of the weight is switched off with `weight_cap: none`.

## The six configurations

| configuration | loss | weight of the energy | peak rate | seeds of the paper (copper) |
|---|---|---|---|---|
| labels-only | D, the label loss | none | 1.25e-4 | 0, 1, 2 |
| energy-only | L_E, the energy loss | none | 2.5e-4 | 1, 2, 3 |
| labels-energy-fixed | D + w(t) lambda_F L_E | fixed, lambda_F = 3.0454 | 1.25e-4 | 0, 1, 2 |
| labels-energy-adaptive | D + w(t) lambda L_E | ratio of the gradient norms, measured every 100 steps, moving average of factor 0.9, cap 100 | 1.25e-4 | 0, 1, 3 |
| labels-energy-uncertainty | D + w(t) lambda L_E | uncertainty weighting, normalized so that the label term keeps unit weight | 1.25e-4 | 0, 1, 2 |
| labels-energy-lagrangian | energy minimized subject to D <= D_t | augmented Lagrangian, a gate and a multiplier update every 500 steps | 1.25e-4 | 0, 1, 2 |

On zener-8, labels-only has the seeds 1, 2 and 3 and labels-energy-adaptive the seeds 0, 1 and 2. A draw that stalled (see below) is replaced by the seed 3.
The files of `configs/train/` are `<material>-<configuration>.yaml`: the six copper configurations, and labels-only and labels-energy-adaptive on zener-8.

D is the mean squared error of the standardized A against the label. L_E is the mean over the volumes and the six load cases of the discrete potential energy of the
displacement fluctuation, divided by the gap between the Voigt and the Reuss stiffness of the volume for that load case; it needs no label. w(t) rises linearly over the
first 1,000 steps. The uncertainty weighting reads the energy gap P = L_E - L_E*, where L_E* is the energy of the label, because P is non-negative and has the same
gradient as L_E. The augmented Lagrangian minimizes the energy subject to D <= D_t; `label_target` is D_t, and until its gate opens (at step 6,000 at the latest) the loss
is that of labels-energy-fixed.

## The recipe

AdamW with zero weight decay, batches of 8 of the 140 training volumes, each batch transformed by one random symmetry of the cube and one random periodic shift. The
rate rises linearly over the first 1,000 steps to its peak, is halved when more than 20 consecutive validations bring no gain, and does not fall below 1e-5. Gradients
are clipped at a global norm of 1.0. Validation on the 20 validation volumes runs every 100 steps.

**Guard.** A run returns to its best state, and halves the rate, when a validation fails one of these tests:

- the validation test: the validation error is not finite or exceeds three times the best;
- the midpoint test: the validation error is above the midpoint between its value at the start (the untrained network) and its best value so far;
- for energy-only, which reads no label and keeps the state of the lowest energy on the validation volumes: the energy test (the energy is not finite or positive) and
  the residual test (the relative norm of the out-of-balance nodal forces of the prediction is not finite or exceeds three times its lowest value).

At most 20 rewinds are made (`max_rewinds`). `--no-midpoint-guard` (in a config, `midpoint_guard: false`) switches the midpoint test off. The draw of labels-only with the seed 0 of the paper
ran without the midpoint test; a replay of its trace shows that the test fires only together with the validation test, so the draw would not change.

**Stop rule.** A run stops 6,000 steps after its best state or its last rewind, never before step 12,000 and never after step 96,000. The kept model is its best state.

**Stall rule.** A draw is stalled if its best state lies at step 8,000 or earlier and more than half of its steps had their gradient clipped. The summary of a run says
whether it stalled (`stalled`); a stalled draw is reported and replaced by the next seed.

## The initial weights

The network starts from the native initialization of torch under the seed, unless `init` names a file. The configurations name the files `initial-<material>-s<seed>.pt`
of the weights folder, which hold the initial weights of the draws of the paper. The native initialization draws other numbers than the paper's draws did, so a run
without the file is a new draw and not a repeat. Start from the native initialization with `--set init=null`.

A seed fixes the batches, the symmetries and the shifts on any machine, because they are drawn with numpy only. The weights of a run follow from them up to the rounding of the
machine: another GPU or another library version changes the last digits of the float32 arithmetic, and a long run can drift from the run of another machine. The rule of
labels-energy-uncertainty amplifies the rounding of its inputs in its first steps, so two machines follow different values of its weight there.

## What a run writes

| file | content |
|---|---|
| `outputs/runs/<material>-<configuration>-s<seed>/checkpoint.pt` | the weights of the best validation state, the stiffness scale and the settings of the run; for the three configurations that set the weight by a rule, the weight of the energy at that state |
| `.../trace.jsonl` | one row for each validation (step, validation error, best, rate, loss, weight) and one for each rewind, with its cause |
| `.../summary.json` | the steps, the best step, the rewinds by cause, the clipped steps, the stop reason and `stalled` |

A run never overwrites a file.

## The quasi-Newton phase

    uv run lbfgs --source outputs/runs/copper-labels-energy-adaptive-s0/checkpoint.pt

Full-batch L-BFGS on the 140 training volumes, with the loss of the source configuration and its weight frozen, a history of 100 pairs, 20 iterations per call and a
strong-Wolfe line search, without augmentation. It stops after 10 calls without a new best validation error, or after the first call that ends past 2.5 hours
(`--limit-s 9000`), and keeps the call with the lowest validation error. The paper started it from the three trained draws of the configuration with labels that had the
lowest mean validation error, labels-energy-adaptive, and it gave no gain at this cost.

## Scoring

    CUDA_VISIBLE_DEVICES="" uv run score --checkpoint outputs/runs/copper-labels-energy-adaptive-s0/checkpoint.pt --sets train val test --out outputs/score.json

The pooled relative error of A in float64 on the CPU ([INFERENCE.md](INFERENCE.md)).
