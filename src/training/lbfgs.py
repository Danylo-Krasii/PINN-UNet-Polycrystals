"""Full-batch L-BFGS after the first-order training of one trained draw.

    uv run lbfgs --source outputs/runs/copper-labels-energy-adaptive-s0/checkpoint.pt --data-dir data

The loss is the loss of the source configuration with its weight frozen, on the 140 training volumes without augmentation: D alone for labels-only,
D + lambda L_E for the combined configurations. lambda is lambda_F for labels-energy-fixed; for the other three it is the weight at the kept state of the
source, which the checkpoint of the trainer holds (`weight`: `lam` for labels-energy-adaptive, the mean of the applied weight over the last validation
interval for the other two); `--weight` gives it directly. The volumes are taken in chunks of 8 with gradient accumulation, a chunk weighted by its
size, so that the sum is the full-batch mean.

The optimizer is `torch.optim.LBFGS` with rate 1, a history of 100 pairs, a strong-Wolfe line search and 20 iterations per call; its state lasts from
call to call. Call 0 is the loaded source. After each call the job takes one forward-only evaluation of the loss and the validation error, and keeps
the state with the lowest validation error. It stops after the first call that ends past `--limit-s` seconds (reason "time"), after 10 calls in a row
without a new best ("patience"), or at a non-finite loss ("nonfinite"); `--max-calls` adds "max-calls". Only training and validation volumes are read.

The training volumes are the first `n_train` of the 140 training seeds and the validation volumes the first `n_val` of the 20 validation seeds; `first_seed`, `n_train`
and `n_val` are those of the source unless the options name others. The files of a run are
`<out>/lbfgs-<material>-<configuration>-s<seed>/{checkpoint.pt, trace.jsonl, summary.json}`; a run never overwrites a file. The
`best_step` of its checkpoint is the number of the best call (0 for the source).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src import names
from src.data import volumes
from src.models import unet
from src.training import single_pass

CONFIGURATIONS = names.LABELLED
CHUNK = 8
VAL_BATCH = 4
MAX_ITER = 20
LBFGS = dict(lr=1.0, history_size=100, tolerance_grad=1e-9, tolerance_change=0.0, line_search_fn="strong_wolfe")
LIMIT_S = 9000.0
PATIENCE = 10


class NonFinite(Exception):
    pass


def set_precision():
    """Deterministic cudnn without benchmark, and TF32 off for the convolutions and the matrix products."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def kept_weight(ck, override=None):
    """The weight of L_E in the loss of the source configuration with the weight frozen: `override` (the option --weight) or the weight of the source; None for labels-only,
    which has the label loss alone."""
    cfg = ck["configuration"]
    if cfg not in CONFIGURATIONS:
        raise SystemExit(f"{cfg}: not a configuration of the L-BFGS phase ({', '.join(CONFIGURATIONS)}); the best state of energy-only is chosen without labels")
    if cfg == "labels-only":
        if override is not None:
            raise SystemExit("labels-only has no energy term: drop --weight")
        return None
    if override is not None:
        return override
    if cfg == "labels-energy-fixed":
        return float(ck["settings"]["lambda_f"])
    weight = ck.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise SystemExit(f"the checkpoint has the weight {weight!r}, not a finite positive number; give --weight")
    if ck["best_step"] < ck["settings"]["weight_warmup"]:
        raise SystemExit(f"the kept step {ck['best_step']} is before the end of the warm-up of the weight: the weight of the checkpoint is not the full weight; give --weight")
    return float(weight)


def full_batch(model, X, Y, S, ids, weight, chunk=CHUNK, grad=True):
    """dict(loss, D, LE): the mean over the volumes `ids` of D + weight L_E, D and L_E, taken in chunks of `chunk` volumes and weighted by the size of the chunk
    over the number of volumes; without a weight the loss is D and LE is None. With `grad`, the gradient of the loss is left in the .grad of the weights,
    which are cleared first; without it nothing is recorded."""
    n = len(ids)
    total = dict(loss=0.0, D=0.0, LE=None if weight is None else 0.0)
    if grad:
        for p in model.parameters():
            p.grad = None
    with torch.set_grad_enabled(grad):
        for i in range(0, n, chunk):
            j = ids[i:i + chunk]
            share = len(j) / n
            mse, LE = single_pass.batch_terms(model, X[j], Y[j], None if weight is None else S[j])
            loss = mse if weight is None else mse + weight * LE
            if grad:
                (loss * share).backward()
            total["loss"] += float(loss) * share
            total["D"] += float(mse) * share
            if weight is not None:
                total["LE"] += float(LE) * share
    return total


def lbfgs_loop(params, full_loss, validate_fn, snapshot, emit, limit_s, t_start, max_iter=MAX_ITER, max_calls=None, patience=PATIENCE):
    """The calls of `torch.optim.LBFGS` on `params`. `full_loss(grad)` is the full-batch dict(loss, D, LE), `validate_fn()` the validation error, `snapshot()` a
    copy of the state and `emit(row)` writes a row of the trace. Returns the counts, the best call, the start and best validation errors, the stop reason and
    the best state."""
    opt = torch.optim.LBFGS(params, max_iter=max_iter, **LBFGS)
    state = opt.state[params[0]]
    evals = [0]

    def closure():
        r = full_loss(True)
        evals[0] += 1
        if not math.isfinite(r["loss"]):
            raise NonFinite(r["loss"])
        return r["loss"]

    r = full_loss(False)
    start_val = best_val = validate_fn()
    best_state = snapshot()
    stop = None if math.isfinite(r["loss"]) and math.isfinite(start_val) else "nonfinite"
    best_call = since_best = calls = iterations = evaluations = 0
    emit(dict(call=0, loss=r["loss"], D=r["D"], LE=r["LE"], val=start_val, iterations=0, evaluations=0, best=best_val, new_best=True, elapsed=time.time() - t_start))
    print(f"      call   0: loss {r['loss']:.8f}  D {r['D']:.6e}  val {start_val:.6e}  (the loaded checkpoint, the first best state)", flush=True)
    while stop is None:
        calls += 1
        evals0, iters0 = evals[0], state.get("n_iter", 0)
        bad = None
        try:
            opt.step(closure)
        except NonFinite as e:
            bad = float(e.args[0])
        it, ev = state.get("n_iter", 0) - iters0, evals[0] - evals0
        iterations, evaluations = iterations + it, evaluations + ev
        if bad is None:
            r = full_loss(False)
            if not math.isfinite(r["loss"]):
                bad = r["loss"]
        if bad is not None:
            stop, v, new_best = "nonfinite", None, False
            r = dict(loss=bad, D=None, LE=None)
        else:
            v = validate_fn()
            new_best = v < best_val
            if new_best:
                best_val, best_call, best_state, since_best = v, calls, snapshot(), 0
            else:
                since_best += 1
        emit(dict(call=calls, loss=r["loss"], D=r["D"], LE=r["LE"], val=v, iterations=it, evaluations=ev, best=best_val, new_best=new_best, elapsed=time.time() - t_start))
        print(f"      call {calls:3d}: loss {r['loss']:.8f}  val {'None' if v is None else format(v, '.6e')}  best {best_val:.6e} (call {best_call})  {it} iterations, {ev} evaluations", flush=True)
        if stop is None:
            if time.time() - t_start >= limit_s:
                stop = "time"
            elif since_best >= patience:
                stop = "patience"
            elif max_calls is not None and calls >= max_calls:
                stop = "max-calls"
    return dict(calls=calls, iterations=iterations, evaluations=evaluations, best_call=best_call, start_val=start_val, best_val=best_val, stop_reason=stop, best_state=best_state)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="the checkpoint of the trained draw")
    ap.add_argument("--weight", type=single_pass.positive, default=None, help="the frozen weight of L_E (default: from the source, see above)")
    single_pass.add_common(ap)
    ap.set_defaults(first_seed=None, n_train=None, n_val=None)
    ap.add_argument("--limit-s", type=single_pass.positive, default=LIMIT_S, help="stop after the first call that ends past this many seconds from the start of the job")
    ap.add_argument("--max-iter", type=single_pass.integer(1), default=MAX_ITER, help="iterations per call")
    ap.add_argument("--max-calls", type=single_pass.integer(1), default=None)
    return ap.parse_args(argv)


def run(a, corpus=None, evaluate_fn=None):
    """Runs the job and returns the summary dict. `corpus` is what `volumes.load_volumes` returns for the training volumes followed by the validation volumes;
    `evaluate_fn` replaces `single_pass.evaluate`; the tests use them."""
    t_start = time.time()
    set_precision()
    dev, dtype = torch.device(a.device), getattr(torch, a.dtype)
    model, ck = unet.load_checkpoint(a.source, dtype, dev)
    cfg, material = ck["configuration"], ck["material"]
    weight = kept_weight(ck, a.weight)
    tag = f"lbfgs-{material}-{cfg}-s{ck['seed']}"
    out_dir = Path(a.out) / tag
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty: a run never overwrites a file")
    out_dir.mkdir(parents=True, exist_ok=True)
    own = {k: ck["settings"].get(k, default) if getattr(a, k) is None else getattr(a, k)
           for k, default in (("first_seed", volumes.FIRST), ("n_train", volumes.N_TRAIN), ("n_val", volumes.N_VAL))}
    if own["n_train"] > volumes.N_TRAIN or own["n_val"] > volumes.N_VAL:
        raise SystemExit(f"there are {volumes.N_TRAIN} training volumes and {volumes.N_VAL} validation volumes")
    lists = volumes.seed_lists(own["first_seed"])
    train, val = lists["train"][:own["n_train"]], lists["val"][:own["n_val"]]
    n_tr, n_va = len(train), len(val)
    if corpus is None:
        corpus = volumes.load_volumes(Path(a.data_dir) / material, train + val, labels=True, scales=weight is not None)
    X, Y = corpus["C"].to(device=dev, dtype=dtype), corpus["A"].to(device=dev, dtype=dtype)
    S = None if weight is None else torch.as_tensor(corpus["S"], dtype=torch.float64).to(dev)
    tr, va = np.arange(n_tr), np.arange(n_tr, n_tr + n_va)
    pars = [p for p in model.parameters() if p.requires_grad]
    trace = out_dir / "trace.jsonl"

    def emit(row):
        with open(trace, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    print(f"      {n_tr} training volumes, {n_va} validation volumes, {a.dtype} on {dev}; source {a.source} (kept step {ck['best_step']}); energy weight {weight}", flush=True)
    result = lbfgs_loop(pars, lambda grad: full_batch(model, X, Y, S, tr, weight, grad=grad), lambda: (evaluate_fn or single_pass.evaluate)(model, X, Y, va, VAL_BATCH),
                        lambda: unet.state_copy(model), emit, a.limit_s, t_start, max_iter=a.max_iter, max_calls=a.max_calls)
    model.load_state_dict(result["best_state"])
    summary = dict(tag=tag, configuration=cfg, material=material, seed=ck["seed"], source_best_step=ck["best_step"], weight=weight, calls=result["calls"],
                   iterations=result["iterations"], evaluations=result["evaluations"], best_call=result["best_call"], start_val=result["start_val"],
                   best_val=result["best_val"], stop_reason=result["stop_reason"], seconds=time.time() - t_start, dtype=a.dtype, lbfgs=LBFGS, max_iter=a.max_iter)
    unet.save_checkpoint(out_dir / "checkpoint.pt", model, configuration=cfg, material=material, seed=ck["seed"], best_step=result["best_call"], weight=weight, settings=ck["settings"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    s = run(parse(argv))
    print(f"{s['tag']}: {s['calls']} calls, {s['iterations']} iterations, {s['evaluations']} evaluations in {s['seconds']:.1f} s; "
          f"validation {s['start_val']:.6e} to {s['best_val']:.6e} (best call {s['best_call']}); stop {s['stop_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
