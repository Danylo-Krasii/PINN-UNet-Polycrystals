"""The guard, the plateau rule, the stop rule and the stall rule of the training loop.

At every validation the guard may return the run to its best state, with the AdamW and plateau states of that state, and halve the base learning
rate (never below 1e-5). A configuration with labels rewinds when the validation error is non-finite or above `factor` times the best (the validation
test), and energy-only rewinds when the energy V_E is non-finite or above 0 (the energy test) or when the relative norm of the out-of-balance nodal forces of
the prediction is non-finite or above 3 times its lowest value (the residual test). The midpoint test adds to either: the run also rewinds when the
validation value is above the midpoint between its value at the start (the untrained network) and its best value so far. Nothing fires once
`max_rewinds` rewinds are used.

The plateau rule halves the base rate after more than `patience` validations without a gain. A run stops when its best state is `patience` steps
old (the count restarts at a rewind), but never before step `min_stop_step`. A draw is stalled if its best state lies at step 8,000 or earlier and
more than half of its steps had their gradient clipped.
"""
import copy
import math

import numpy as np

from src.training import energy

MIN_LR = 1e-5
STALL_STEP = 8000
STALL_SHARE = 0.5


class Plateau:
    """The rule of `torch.optim.lr_scheduler.ReduceLROnPlateau` (mode min, factor 0.5, cooldown 0, min_lr 1e-5, eps 1e-8) acting on a base rate that is not
    the rate in the optimizer; `better(v, best)` is the rule of gain."""

    def __init__(self, better, patience, factor=0.5, min_lr=MIN_LR, eps=1e-8):
        self.better, self.patience, self.factor, self.min_lr, self.eps = better, patience, factor, min_lr, eps
        self.best, self.bad = math.inf, 0

    def step(self, v, base):
        if self.better(v, self.best):
            self.best, self.bad = v, 0
        else:
            self.bad += 1
        if self.bad > self.patience:
            new = max(base * self.factor, self.min_lr)
            if base - new > self.eps:
                base = new
            self.bad = 0
        return base

    def state_dict(self):
        return dict(best=self.best, bad=self.bad)

    def load_state_dict(self, d):
        self.best, self.bad = d["best"], d["bad"]


def check(v, best, val_init, rewinds, max_rewinds, factor=3.0, midpoint=True, residual=None, lowest_residual=math.inf):
    """(fire, cause) at a validation with the value v. `residual` is the residual of energy-only (None for the configurations with labels)."""
    if rewinds >= max_rewinds:
        return False, ""
    if residual is not None:
        fire, cause = energy.e_guard(v, residual, lowest_residual)
    else:
        fire, cause = (not np.isfinite(v) or v > factor * best), "validation"
    if midpoint and not fire and v > (val_init + best) / 2:
        return True, "midpoint"
    return fire, cause


def halved_base(best_base, base):
    """The base rate after a rewind: half the lower of the saved and the current one, to 1e-5 at least."""
    return max(min(best_base, base) * 0.5, MIN_LR)


def rewind(model, opt, plateau, rule, s):
    """Returns the model, the optimizer, the plateau rule and the weight rule to the best state and halves the base rate. A rewind to the state of step 0
    (the optimizer state is empty) restarts the warm-up of the rate from this step. Returns the base rate."""
    model.load_state_dict(s.best_state)
    opt.load_state_dict(copy.deepcopy(s.best_opt))
    plateau.load_state_dict(copy.deepcopy(s.best_plateau))
    rule.restore(s.best_rule)
    s.base = halved_base(s.best_base, s.base)
    if s.best_step == 0:
        s.t0 = s.step
    return s.base


def should_stop(bad, step, patience, min_stop_step):
    """The stop rule: `bad` steps without a new best state or a rewind, at least `patience` of them, and not before step `min_stop_step`."""
    return patience is not None and bad >= patience and step >= min_stop_step


def is_stalled(steps, best_step, clipped):
    """The stall rule: the best state at step 8,000 or earlier and more than half of the steps clipped."""
    return best_step <= STALL_STEP and clipped > steps * STALL_SHARE
