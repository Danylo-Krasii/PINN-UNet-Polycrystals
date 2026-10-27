"""The weight of the energy term in the four combined configurations.

Every combined configuration follows the gradient of D + w(t) lambda(t) L_E, with D the label loss, L_E the energy loss and the warm-up
w(t) = min(1, t / warmup). The configurations differ in lambda(t):

  labels-energy-fixed        `Fixed`: lambda is a constant.
  labels-energy-adaptive     `Adaptive`: gradient-norm balancing, lambda = |grad D| / |grad L_E| measured every `every` steps, smoothed by a
                             moving average of factor alpha, capped.
  labels-energy-uncertainty  `Uncertainty`: two log-variances s = (s_d, s_p) are trained by their own AdamW on e^-s_d D + e^-s_p P + s_d + s_p, with
                             D and P the detached batch values and P = L_E - L_E* the energy gap, and lambda = min(cap, e^(s_d - s_p)).
  labels-energy-lagrangian   `Lagrangian`: the energy is minimized subject to D <= D_t by an augmented Lagrangian; the loss is that of
                             labels-energy-fixed until a gate opens, and lambda_F (lambda_D D + w(t) L_E) after it.

A rule object follows the training loop through `begin`, `loss`, `applied` and `after_step` at every step, and through `at_row` and `lam_at` at
every validation row; `snapshot` and `restore` return it to the state of the best weights at a rewind.
"""
import collections
import copy
import math

import numpy as np
import torch

from src.data import volumes
from src.training import energy

UNCERTAINTY_LR = 1.25e-3
P_FLOOR = 1e-12
GATE_LOSS = 1.5
GATE_FIRST = 1000
GATE_STEP = 6000
RAMP = 1000
UPDATE_EVERY = 500
WINDOW = 100
LAMBDA_MIN = 0.01
ALM_UPPER = 11.0
NU_UPPER = 10.0


def warmup(lam, step, steps):
    return lam * (min(1.0, step / steps) if steps else 1.0)


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.linalg.vector_norm(torch.cat([t.flatten() for t in g if t is not None]))


def gradnorm_update(lam, mse, le, params, alpha=0.9, cap=None):
    """The moving-average weight of gradient-norm balancing after one measurement; the first measurement is taken as it is."""
    lam_hat = (grad_norm(mse, params) / grad_norm(le, params).clamp_min(1e-30)).item()
    lam = lam_hat if lam is None else alpha * lam + (1 - alpha) * lam_hat
    return min(lam, cap) if cap is not None else lam


def uncertainty_loss(mse, gap, s):
    return torch.exp(-s[0]) * mse + torch.exp(-s[1]) * gap + s.sum()


def bounded(x, lo, hi):
    return min(max(x, lo), hi)


def star_ratio(C6, A):
    """(6,) float64: Pi*_j / S_j of one volume at the label, the Hill-Mandel value Pi*_j = -1/2 n^3 (<C>_jj - <A>_jj) over the scale S_j of the energy loss;
    C6 and A are the float64 (n, n, n, 6, 6) arrays of `data.read_volume`. Every value lies in [-1, 0]."""
    C6, A = np.asarray(C6, dtype=np.float64), np.asarray(A, dtype=np.float64)
    S = energy.scale_S(C6)[0].numpy()
    return -0.5 * C6.shape[0] ** 3 * (np.diag(C6.reshape(-1, 6, 6).mean(0)) - np.diag(A.reshape(-1, 6, 6).mean(0))) / S


def estar_of_volume(C6, A):
    """e*_v: the mean over the six load cases of Pi*_vj / S_vj, the energy loss L_E of the label of the volume; the cube operations and the shifts leave it as it is."""
    return float(star_ratio(C6, A).mean())


def estar_of_seeds(folder, seeds):
    """(len(seeds),) float64: e*_v of the volumes of the seeds, from the float64 labels."""
    return np.array([estar_of_volume(*volumes.read_volume(folder, s)) for s in seeds], dtype=np.float64)


def energy_gap(le, e_star):
    """P = L_E - e*, e* the mean of e*_v over the batch (a constant): P >= 0, P = 0 only at the label, and grad P = grad L_E."""
    return le - e_star


class Rule:
    """The base of the rules: a rule without state that adds nothing to the trace."""
    needs_gap = False

    def begin(self, d_hat, p_hat):
        pass

    def loss(self, mse, le, step):
        return mse + self.applied(step) * le

    def after_step(self, d_hat, p_hat):
        pass

    def at_row(self, step):
        pass

    def lam_at(self, step):
        return None

    def fields(self):
        return {}

    def snapshot(self):
        return None

    def restore(self, snap):
        pass


class Fixed(Rule):
    def __init__(self, lambda_f, warm):
        self.lambda_f, self.warm = lambda_f, warm

    def applied(self, step):
        return warmup(self.lambda_f, step, self.warm)


class Adaptive(Rule):
    """Gradient-norm balancing: lambda is measured at the first step and then every `every` steps."""

    def __init__(self, params, every, alpha, cap, warm):
        self.params, self.every, self.alpha, self.cap, self.warm = params, every, alpha, cap, warm
        self.lam = None

    def applied(self, step):
        return warmup(self.lam, step, self.warm)

    def loss(self, mse, le, step):
        if step % self.every == 0 or self.lam is None:
            self.lam = gradnorm_update(self.lam, mse, le, self.params, self.alpha, self.cap)
        return mse + self.applied(step) * le

    def lam_at(self, step):
        return self.lam

    def snapshot(self):
        return self.lam

    def restore(self, snap):
        self.lam = snap


def lam_of(s_d, s_p, cap):
    """lambda_U = min(cap, e^(s_d - s_p)) as a Python float (no cap when cap is None)."""
    lam = math.exp(min(s_d - s_p, 700.0))
    return lam if cap is None else min(cap, lam)


def valid(d_hat, p_hat):
    return math.isfinite(d_hat) and d_hat > 0 and math.isfinite(p_hat)


class Uncertainty(Rule):
    """Uncertainty weighting, normalized so that the label term keeps unit weight: s starts at (ln D, ln P) of the first batch. A batch value that is not finite
    (or a D that is not positive) leaves s and its moments as they are."""
    needs_gap = True

    def __init__(self, lr, cap, warm):
        self.lr, self.cap, self.warm = lr, cap, warm
        self.s = self.opt = None

    def optimizer(self):
        return torch.optim.AdamW([self.s], lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

    def begin(self, d_hat, p_hat):
        if self.s is not None or not valid(d_hat, p_hat):
            return
        self.s = torch.tensor([math.log(d_hat), math.log(max(p_hat, P_FLOOR))], dtype=torch.float64, requires_grad=True)
        self.opt = self.optimizer()

    def lam(self):
        return None if self.s is None else lam_of(float(self.s[0]), float(self.s[1]), self.cap)

    def applied(self, step):
        lam = self.lam()
        return warmup(0.0 if lam is None else lam, step, self.warm)

    def after_step(self, d_hat, p_hat):
        if self.s is None or not valid(d_hat, p_hat):
            return
        L_s = uncertainty_loss(torch.tensor(d_hat, dtype=torch.float64), torch.tensor(max(p_hat, P_FLOOR), dtype=torch.float64), self.s)
        self.opt.zero_grad(set_to_none=True)
        L_s.backward()
        self.opt.step()

    def lam_at(self, step):
        return self.lam()

    def fields(self):
        return dict(s_d=None if self.s is None else float(self.s[0]), s_p=None if self.s is None else float(self.s[1]))

    def snapshot(self):
        return dict(s=None if self.s is None else self.s.detach().clone(), opt=None if self.opt is None else copy.deepcopy(self.opt.state_dict()))

    def restore(self, snap):
        if snap["s"] is None:
            self.s = self.opt = None
            return
        self.s = snap["s"].detach().clone().requires_grad_(True)
        self.opt = self.optimizer()
        self.opt.load_state_dict(copy.deepcopy(snap["opt"]))


class Lagrangian(Rule):
    """The augmented Lagrangian: the objective is L_E and the constraint c = D / D_t - 1 <= 0. Until the gate the loss is that of labels-energy-fixed; after it
    lambda_F (lambda_D D + w(t) L_E) with lambda_D = nu_0 + rho(t) (lambda_ALM - nu_0), rho(t) = min(1, (t - t_gate) / RAMP), nu_0 = 1 / lambda_F,
    lambda_ALM = clip(nu + nu_0 clip(cbar_100, -1, 1), LAMBDA_MIN, ALM_UPPER nu_0), and the multiplier nu changed every UPDATE_EVERY steps by
    nu_0 clip(cbar_500, -1, 1) within [0, NU_UPPER nu_0]. cbar_100 and cbar_500 are the means of c over the last WINDOW steps and over the steps since the last
    update. The gate opens at the first validation row at step GATE_FIRST or later at which the mean of D over the last WINDOW steps is below
    GATE_LOSS D_t (by "loss"), or at the first validation row at or after GATE_STEP (by "step"). A D that is not finite does not enter the means."""

    def __init__(self, lambda_f, label_target, warm):
        self.lambda_f, self.d_t, self.warm = float(lambda_f), float(label_target), warm
        self.nu0 = 1.0 / self.lambda_f
        self.nu, self.gate_step, self.gate_by = self.nu0, None, None
        self.window, self.since, self.age, self.updates = collections.deque(maxlen=WINDOW), [], 0, 0

    def c_bar_100(self):
        return None if not self.window else sum(self.window) / len(self.window) / self.d_t - 1.0

    def lam_alm(self):
        c = self.c_bar_100()
        return bounded(self.nu + self.nu0 * (0.0 if c is None else bounded(c, -1.0, 1.0)), LAMBDA_MIN, ALM_UPPER * self.nu0)

    def lam_d(self, step):
        if self.gate_step is None:
            return self.nu0
        rho = bounded((step - self.gate_step) / RAMP, 0.0, 1.0)
        return (1.0 - rho) * self.nu0 + rho * self.lam_alm()

    def lam_at(self, step):
        return 1.0 / self.lam_d(step)

    def applied(self, step):
        if self.gate_step is None:
            return warmup(self.lambda_f, step, self.warm)
        return warmup(1.0, step, self.warm) / self.lam_d(step)

    def loss(self, mse, le, step):
        if self.gate_step is None:
            return mse + warmup(self.lambda_f, step, self.warm) * le
        return self.lambda_f * (self.lam_d(step) * mse.double() + warmup(1.0, step, self.warm) * le)

    def after_step(self, d_hat, p_hat):
        if math.isfinite(d_hat):
            self.window.append(d_hat)
            if self.gate_step is not None:
                self.since.append(d_hat)
        if self.gate_step is None:
            return
        self.age += 1
        if self.age == UPDATE_EVERY:
            if self.since:
                c = sum(self.since) / len(self.since) / self.d_t - 1.0
                self.nu = bounded(self.nu + self.nu0 * bounded(c, -1.0, 1.0), 0.0, NU_UPPER * self.nu0)
            self.since, self.age, self.updates = [], 0, self.updates + 1

    def at_row(self, step):
        if self.gate_step is not None:
            return
        if step >= GATE_FIRST and self.window and sum(self.window) / len(self.window) < GATE_LOSS * self.d_t:
            by = "loss"
        elif step >= GATE_STEP:
            by = "step"
        else:
            return
        self.gate_step, self.gate_by, self.nu, self.since, self.age = step, by, self.nu0, [], 0

    def fields(self):
        return dict(nu=self.nu / self.nu0, cbar=self.c_bar_100(), gate_step=self.gate_step, gate_by=self.gate_by)

    def snapshot(self):
        return dict(nu=self.nu, gate_step=self.gate_step, gate_by=self.gate_by, updates=self.updates)

    def restore(self, snap):
        self.nu, self.gate_step, self.gate_by, self.updates = snap["nu"], snap["gate_step"], snap["gate_by"], snap["updates"]
        self.window.clear()
        self.since, self.age = [], 0


def frozen_weight(configuration, lam, applied_log, every):
    """The weight of L_E that the L-BFGS phase freezes at a kept validation row (the `lam` and the `lam_applied` of the trace row): lam for
    labels-energy-adaptive, and for labels-energy-uncertainty and labels-energy-lagrangian the mean of the applied weight over the last `every` steps.
    None for the other configurations; labels-energy-fixed freezes lambda_F."""
    if configuration == "labels-energy-adaptive":
        return None if lam is None else float(lam)
    if configuration in ("labels-energy-uncertainty", "labels-energy-lagrangian"):
        return float(np.mean(applied_log[-every:]))
    return None


def make_rule(configuration, params, a):
    """The rule of a combined configuration; `a` holds lambda_f, label_target, uncertainty_lr, weight_cap, weight_warmup, balance_every and balance_alpha."""
    if configuration == "labels-energy-fixed":
        return Fixed(a.lambda_f, a.weight_warmup)
    if configuration == "labels-energy-adaptive":
        return Adaptive(params, a.balance_every, a.balance_alpha, a.weight_cap, a.weight_warmup)
    if configuration == "labels-energy-uncertainty":
        return Uncertainty(a.uncertainty_lr, a.weight_cap, a.weight_warmup)
    return Lagrangian(a.lambda_f, a.label_target, a.weight_warmup)
