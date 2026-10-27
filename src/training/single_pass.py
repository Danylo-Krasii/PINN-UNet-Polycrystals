"""Training of the single-pass U-Net surrogate in one of six configurations (Table 2 of the paper):

  labels-only                the label loss D, the mean squared error of the standardized A against the label;
  energy-only                the energy loss L_E alone; no label is read, and the best state is chosen by the energy on the validation volumes;
  labels-energy-fixed        D + w(t) lambda_F L_E with a fixed lambda_F;
  labels-energy-adaptive     D + w(t) lambda L_E, lambda by gradient-norm balancing;
  labels-energy-uncertainty  D + w(t) lambda L_E, lambda by uncertainty weighting;
  labels-energy-lagrangian   L_E subject to D <= D_t by an augmented Lagrangian (`--label-target` is D_t).

    uv run train --config configs/train/copper-labels-energy-adaptive.yaml [--set key=value ...] [--dry-run]
    python -m src.training.single_pass --configuration labels-energy-adaptive --seed 0 --material copper --data-dir data --out outputs/runs

The first command reads a YAML config (src/train.py) and calls the second with the options of the file; every option has the default below.

The recipe: AdamW with zero weight decay, batches of 8 training volumes, each transformed by one random symmetry of the cube and one random periodic
shift; the base rate rises linearly over `--warmup-steps` steps, is halved by the plateau rule, and the guard of `guard.py` returns the run to its
best state; gradients are clipped at a global norm of `--clip`. The run stops `--patience` steps after its best state or its last rewind, never before
`--min-stop-step` and never after `--steps`, and keeps its best state. The weight w(t) of the energy rises over `--weight-warmup` steps.

The files of a run are `<out>/<material>-<configuration>-s<seed>/{checkpoint.pt, trace.jsonl, summary.json}`. A run never overwrites a file. Without
`--init` the network starts from the native initialization of `torch` under the seed; `--init FILE` starts from the network weights of a checkpoint file
(`{material}` and `{seed}` in the name stand for those of the run). The draws are made with numpy only, so for one seed and one `--init` file they
do not depend on the machine. The checkpoint also holds the frozen weight of the energy at the kept state (`weight`), which the L-BFGS phase reads.
"""
import argparse
import copy
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from src import names, settings
from src.data import volumes
from src.data.symmetries import Octahedral
from src.models import unet
from src.training import balance, energy, guard

LAMBDA_F = 3.0454335833546127
PEAK = {"energy-only": 2.5e-4}
PEAK_DEFAULT = 1.25e-4


def positive(text):
    v = float(text)
    if not (math.isfinite(v) and v > 0):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive finite number")
    return v


def integer(least):
    def check(text):
        v = int(text)
        if v < least:
            raise argparse.ArgumentTypeError(f"{text!r} is below {least}")
        return v
    return check


def fraction(text):
    v = float(text)
    if not (math.isfinite(v) and 0 <= v < 1):
        raise argparse.ArgumentTypeError(f"{text!r} is not a number in [0, 1)")
    return v


def cap(text):
    return None if text.lower() == "none" else positive(text)


def add_common(ap):
    """The options that the trainer and the L-BFGS phase share: the data, the output folder, the device and the lists of volumes."""
    ap.add_argument("--data-dir", default=str(settings.DATA), help="the folder that holds the material folders (default: {DATA} of src/settings.py)")
    ap.add_argument("--out", default=str(settings.ROOT / "outputs" / "runs"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="float64 is for checks on the CPU")
    ap.add_argument("--first-seed", type=int, default=volumes.FIRST, help="the first volume of the training list")
    ap.add_argument("--n-train", type=integer(1), default=volumes.N_TRAIN)
    ap.add_argument("--n-val", type=integer(1), default=volumes.N_VAL)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configuration", required=True, choices=names.CONFIGURATIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--material", choices=names.MATERIALS, default="copper", help="the folder <data-dir>/<material> of the volumes")
    ap.add_argument("--init", default=None, help="a checkpoint file whose network weights are the initial weights; {material} and {seed} stand for those of the run")
    add_common(ap)
    ap.add_argument("--steps", type=integer(1), default=96000)
    ap.add_argument("--batch", type=integer(1), default=8)
    ap.add_argument("--peak", type=positive, default=None, help="the base rate after the warm-up (default 1.25e-4; energy-only 2.5e-4)")
    ap.add_argument("--warmup-steps", type=integer(0), default=1000, help="the linear warm-up of the rate (0: none)")
    ap.add_argument("--clip", type=positive, default=1.0)
    ap.add_argument("--sched-patience", type=integer(0), default=20)
    ap.add_argument("--val-every", type=integer(1), default=100)
    ap.add_argument("--val-batch", type=integer(1), default=4)
    ap.add_argument("--guard", type=positive, default=3.0, help="the validation test rewinds above this multiple of the best validation error")
    ap.add_argument("--max-rewinds", type=integer(0), default=20)
    ap.add_argument("--midpoint-guard", action=argparse.BooleanOptionalAction, default=True, help="the midpoint test (default: on)")
    ap.add_argument("--skip-rewind", type=integer(1), default=50, help="rewind after this many skipped (non-finite) steps in a row; stop as diverged when the rewinds are used up")
    ap.add_argument("--patience", type=integer(0), default=6000, help="stop after this many steps without a new best state or rewind (0: never)")
    ap.add_argument("--min-stop-step", type=integer(0), default=12000)
    ap.add_argument("--lambda-f", type=positive, default=None, help=f"labels-energy-fixed and labels-energy-lagrangian: the weight lambda_F (default {LAMBDA_F})")
    ap.add_argument("--uncertainty-lr", type=positive, default=None, help=f"labels-energy-uncertainty: the rate of the AdamW of s (default {balance.UNCERTAINTY_LR})")
    ap.add_argument("--label-target", type=positive, default=None, help="labels-energy-lagrangian: the target D_t of the label loss (required)")
    ap.add_argument("--weight-warmup", type=integer(0), default=1000, help="steps of the linear rise of the weight of the energy")
    ap.add_argument("--weight-cap", type=cap, default=100.0, help="labels-energy-adaptive and labels-energy-uncertainty: cap of the weight (a number, or none)")
    ap.add_argument("--balance-every", type=integer(1), default=100, help="labels-energy-adaptive: steps between two measurements of the gradient norms")
    ap.add_argument("--balance-alpha", type=fraction, default=0.9, help="labels-energy-adaptive: factor of the moving average")
    a = ap.parse_args(argv)
    cfg = a.configuration
    if cfg in ("labels-energy-fixed", "labels-energy-lagrangian"):
        a.lambda_f = LAMBDA_F if a.lambda_f is None else a.lambda_f
    elif a.lambda_f is not None:
        ap.error("--lambda-f is for labels-energy-fixed and labels-energy-lagrangian only")
    if cfg == "labels-energy-uncertainty":
        a.uncertainty_lr = balance.UNCERTAINTY_LR if a.uncertainty_lr is None else a.uncertainty_lr
    elif a.uncertainty_lr is not None:
        ap.error("--uncertainty-lr is for labels-energy-uncertainty only")
    if cfg == "labels-energy-lagrangian" and a.label_target is None:
        ap.error("labels-energy-lagrangian needs --label-target")
    if cfg != "labels-energy-lagrangian" and a.label_target is not None:
        ap.error("--label-target is for labels-energy-lagrangian only")
    if cfg == "energy-only" and a.guard != 3.0:
        ap.error("the residual test of energy-only is fixed at 3 times the lowest residual")
    if a.peak is None:
        a.peak = PEAK.get(cfg, PEAK_DEFAULT)
    a.patience = a.patience or None
    if a.init:
        a.init = a.init.format(material=a.material, seed=a.seed)
    return a


@torch.no_grad()
def evaluate(model, X, Y, ids, batch):
    """Pooled error of A over the volumes `ids`: the forward in the dtype of the data, the sums in float64."""
    num = den = 0.0
    for i in range(0, len(ids), batch):
        j = ids[i:i + batch]
        p = model(X[j])["A"]
        num += float(((p - Y[j]).double() ** 2).sum())
        den += float((Y[j].double() ** 2).sum())
    return float(np.sqrt(num / den))


def batch_terms(model, xb, yb, S_b):
    """(D, L_E) of one batch: D is None without a label (yb None) and L_E is None without the scales (S_b None)."""
    out = model(xb, stress=yb is not None, out_of_balance=S_b is not None)
    LE = D = None
    if S_b is not None:
        LE = energy.loss_E(energy.potential_energy(out["w"], out["b"], out["out_of_balance"]), S_b)
    if yb is not None:
        D = Fn.mse_loss((out["A"] - model.ym) / model.ys, (yb - model.ym) / model.ys)
    return D, LE


def shifted(x, shift):
    """x (B, C, n, n, n) rolled by `shift` voxels along the three spatial axes (a node field rolls like a voxel field)."""
    return torch.roll(x, tuple(int(s) for s in shift), (2, 3, 4))


def applied_lr(base, t, t0, warmup):
    """The rate at the training step t = 1, 2, ...: base x min(1, (t - t0) / warmup); the base without a warm-up (warmup 0)."""
    return base * min(1.0, (t - t0) / warmup) if warmup else base


def plain(a):
    """The settings of a run as plain values, without the paths and the device of the machine."""
    return {k: v for k, v in vars(a).items() if k not in ("data_dir", "out", "init", "device") and (v is None or isinstance(v, (bool, int, float, str)))}


def train(a, corpus=None, evaluate_fn=None):
    """Runs one training and returns the summary dict. `corpus` is what `volumes.load_volumes` returns for the training and validation seeds (plus `estar` for
    labels-energy-uncertainty); `evaluate_fn` replaces `evaluate` (labels: (model, X, Y, ids, batch) -> error) or `energy.validate` (energy-only:
    (model, c21, S, batch=) -> dict(V_E, residual)); the tests use it to script the validations."""
    cfg = a.configuration
    labelled, with_energy = cfg != "energy-only", cfg != "labels-only"
    evaluate_fn = evaluate_fn or (evaluate if labelled else energy.validate)
    dev, dtype = torch.device(a.device), getattr(torch, a.dtype)
    tag = f"{a.material}-{cfg}-s{a.seed}"
    out_dir = Path(a.out) / tag
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty: a run never overwrites a file")
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = volumes.seed_lists(a.first_seed, a.n_train, a.n_val, 0)
    n_tr, n_va = len(lists["train"]), len(lists["val"])
    tr, va = np.arange(n_tr), np.arange(n_tr, n_tr + n_va)
    if corpus is None:
        folder = Path(a.data_dir) / a.material
        corpus = volumes.load_volumes(folder, lists["train"] + lists["val"], labels=labelled, scales=with_energy)
        if cfg == "labels-energy-uncertainty":
            corpus["estar"] = balance.estar_of_seeds(folder, lists["train"])
    C, A, S = corpus["C"], corpus["A"], corpus["S"]
    stats = volumes.train_stats(C[tr], A[tr] if labelled else None)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = unet.Surrogate(stats)
    if a.init:
        unet.load_initial_weights(model, a.init)
    model = model.to(device=dev, dtype=dtype)
    X = C.to(device=dev, dtype=dtype)
    Y = A.to(device=dev, dtype=dtype) if labelled else None
    S_all = torch.as_tensor(S, dtype=torch.float64).to(dev) if with_energy else None
    n = X.shape[-1]
    oct_ = Octahedral(dev)
    perms = [energy.load_perm(oct_, g).to(dev) for g in range(len(oct_.ops))] if with_energy else None
    pars = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(pars, lr=a.peak, weight_decay=0.0)
    gain = (lambda v, best: v < best * (1 - 1e-4)) if labelled else (lambda v, best: energy.better(v, best, energy.THR_PLATEAU))
    plateau = guard.Plateau(gain, a.sched_patience)
    rule = balance.make_rule(cfg, pars, a) if with_energy and labelled else balance.Rule()
    estar = np.asarray(corpus["estar"], dtype=np.float64) if rule.needs_gap else None

    def improved(v, best):
        return v < best - 1e-5 if labelled else energy.better(v, best, energy.THR_BEST)

    def validate():
        if labelled:
            return evaluate_fn(model, X, Y, va, a.val_batch), None
        res = evaluate_fn(model, X[va], S_all[va], batch=a.val_batch)
        return float(res["V_E"]), float(res["residual"])

    s = types.SimpleNamespace(
        step=0, base=a.peak, t0=0, best=math.inf, best_step=0, best_state=None, best_opt=None, best_plateau=None, best_base=a.peak, best_rule=None, best_weight=None,
        bad=0, rewinds=0, nonfinite=0, skipped=0, clipped=0, lowest_res=math.inf, causes=Counter(),
        losses=[], mse_log=[], e_log=[], w_log=[])

    def keep_best():
        s.best_state = unet.state_copy(model)
        s.best_opt, s.best_plateau, s.best_base = copy.deepcopy(opt.state_dict()), plateau.state_dict(), s.base
        s.best_rule = rule.snapshot()

    keep_best()
    s.val_init, res_init = validate()
    s.best = s.val_init if np.isfinite(s.val_init) else math.inf
    s.lowest_res = res_init if res_init is not None and np.isfinite(res_init) else math.inf
    print(f"      step      0: val {s.val_init:.4g} (the initial weights, the first best state)", flush=True)
    trace = out_dir / "trace.jsonl"

    def emit(line):
        with open(trace, "a") as fh:
            fh.write(json.dumps(line) + "\n")

    def rewound(cause, **fields):
        lr = guard.rewind(model, opt, plateau, rule, s)
        s.rewinds += 1
        s.causes[cause] += 1
        emit(dict(step=s.step, event="rewind", cause=cause, best=s.best, best_step=s.best_step, lr=lr, t0=s.t0, **fields))
        return lr

    stop_reason = None
    stop = False
    while s.step < a.steps and not stop:
        b = np.random.choice(tr, size=min(a.batch, len(tr)), replace=False)
        g = int(np.random.randint(48))
        sh = np.random.randint(0, n, size=3)
        xb, yb = oct_.apply(X[b], Y[b] if labelled else None, g)
        xb = shifted(xb, sh)
        if labelled:
            yb = shifted(yb, sh)
        opt.zero_grad(set_to_none=True)
        S_b = S_all[torch.as_tensor(b, device=dev)][:, perms[g]] if with_energy else None
        D, LE = batch_terms(model, xb, yb, S_b)
        weight = None
        d_hat = p_hat = None
        if not with_energy:
            loss = D
        elif not labelled:
            loss = LE
        else:
            d_hat = float(D.detach())
            if rule.needs_gap:
                p_hat = balance.energy_gap(float(LE.detach()), float(estar[b].mean()))
            rule.begin(d_hat, p_hat)
            loss = rule.loss(D, LE, s.step)
            weight = rule.applied(s.step)
        if labelled:
            s.mse_log.append(float(D.detach()))
        if with_energy:
            s.e_log.append(float(LE.detach()))
        if weight is not None:
            s.w_log.append(float(weight))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pars, a.clip)
        if torch.isfinite(gn):
            opt.param_groups[0]["lr"] = applied_lr(s.base, s.step + 1, s.t0, a.warmup_steps)
            opt.step()
            s.clipped += int(gn > a.clip)
            s.skipped = 0
        else:
            s.nonfinite += 1
            s.skipped += 1
        rule.after_step(d_hat, p_hat)
        s.losses.append(float(loss.detach()))
        s.step += 1
        if s.skipped >= a.skip_rewind:
            if s.rewinds >= a.max_rewinds:
                stop, stop_reason = True, "diverged"
                emit(dict(step=s.step, event="stop", reason=stop_reason, skipped=s.skipped, rewinds=s.rewinds, best=s.best, best_step=s.best_step))
                print(f"      step {s.step:6d}: {s.skipped} skipped steps in a row and the {s.rewinds} rewinds are used up; the run stops as diverged", flush=True)
                continue
            lr = rewound("skipped", skipped=s.skipped)
            print(f"      step {s.step:6d}: {s.skipped} skipped steps in a row; rewound to best (step {s.best_step}), base rate now {lr:.2e} ({s.rewinds}/{a.max_rewinds})", flush=True)
            s.skipped = s.bad = 0
            continue
        if s.step % a.val_every == 0:
            v, res = validate()
            fire, cause = guard.check(v, s.best, s.val_init, s.rewinds, a.max_rewinds, a.guard, a.midpoint_guard, res, s.lowest_res)
            extra = {} if res is None else dict(residual=res, lowest_residual=s.lowest_res)
            if fire:
                lr = rewound(cause, val=v, **extra)
                print(f"      step {s.step:6d}: val {v:.4g} ({cause}) against best {s.best:.4g}; rewound to best, base rate now {lr:.2e} ({s.rewinds}/{a.max_rewinds})", flush=True)
                s.bad = s.skipped = 0
                continue
            if res is not None and np.isfinite(res):
                s.lowest_res = min(s.lowest_res, res)
            rule.at_row(s.step)
            lam = rule.lam_at(s.step)
            s.base = plateau.step(v, s.base)
            if improved(v, s.best):
                s.best, s.bad, s.best_step = v, 0, s.step
                keep_best()
                s.best_weight = balance.frozen_weight(cfg, lam, s.w_log, a.val_every)
            else:
                s.bad += a.val_every
            last = slice(-a.val_every, None)
            emit(dict(step=s.step, val=v, best=s.best, lr=s.base, loss=float(np.mean(s.losses[last])), mse=float(np.mean(s.mse_log[last])) if s.mse_log else None,
                      lam=None if lam is None else float(lam), **extra, **({"energy": float(np.mean(s.e_log[last]))} if s.e_log else {}),
                      **({"lam_applied": float(np.mean(s.w_log[last]))} if s.w_log else {}), **rule.fields()))
            if s.step % 500 == 0:
                print(f"      step {s.step:6d}/{a.steps}  val {v:.4f}  best {s.best:.4f}  base rate {s.base:.2e}", flush=True)
            if guard.should_stop(s.bad, s.step, a.patience, a.min_stop_step):
                stop, stop_reason = True, "patience"
    model.load_state_dict(s.best_state)
    summary = dict(tag=tag, configuration=cfg, material=a.material, seed=a.seed, steps=s.step, params=unet.count_params(model), best_step=s.best_step, best_val=s.best,
                   val_init=s.val_init, rewinds=s.rewinds, rewinds_by_cause=dict(s.causes), clipped=s.clipped, nonfinite=s.nonfinite, stop_reason=stop_reason,
                   stalled=bool(guard.is_stalled(s.step, s.best_step, s.clipped)), weight=s.best_weight, settings=plain(a))
    unet.save_checkpoint(out_dir / "checkpoint.pt", model, configuration=cfg, material=a.material, seed=a.seed, best_step=s.best_step, weight=s.best_weight,
                         settings=summary["settings"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    s = train(parse(argv))
    print(f"{s['tag']}: {s['steps']} steps, best state at step {s['best_step']} (validation {s['best_val']:.6g}), {s['rewinds']} rewinds, stop {s['stop_reason']}, stalled {s['stalled']}")
    return 3 if s["stop_reason"] == "diverged" else 0


if __name__ == "__main__":
    sys.exit(main())
