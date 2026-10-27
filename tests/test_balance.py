"""The rules that set the weight of the energy: warm-up, gradient-norm balancing, uncertainty weighting and the augmented Lagrangian."""
import math

import pytest
import torch

from src.training import balance


def test_the_warm_up_rises_linearly_and_stays():
    assert balance.warmup(3.0, 0, 1000) == 0.0
    assert balance.warmup(3.0, 500, 1000) == 1.5
    assert balance.warmup(3.0, 1000, 1000) == 3.0 == balance.warmup(3.0, 5000, 1000)
    assert balance.warmup(3.0, 0, 0) == 3.0


def test_gradient_norm_balancing_is_the_ratio_of_the_gradient_norms_smoothed_and_capped():
    w = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    d = (w ** 2).sum()
    e = 0.5 * (w ** 2).sum()
    assert balance.gradnorm_update(None, d, e, [w]) == pytest.approx(2.0)
    assert balance.gradnorm_update(4.0, d, e, [w], alpha=0.9) == pytest.approx(0.9 * 4.0 + 0.1 * 2.0)
    assert balance.gradnorm_update(None, d, e, [w], cap=1.5) == 1.5


def test_adaptive_measures_at_the_first_step_and_then_every_given_number_of_steps():
    w = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    rule = balance.Adaptive([w], every=3, alpha=0.9, cap=100.0, warm=10)
    d, e = (w ** 2).sum(), 0.5 * (w ** 2).sum()
    loss = rule.loss(d, e, 1)
    assert rule.lam == pytest.approx(2.0) and float(loss) == pytest.approx(float(d) + 0.1 * 2.0 * float(e))
    rule.lam = 5.0
    rule.loss(d, e, 2)
    assert rule.lam == 5.0
    rule.loss(d, e, 3)
    assert rule.lam == pytest.approx(0.9 * 5.0 + 0.1 * 2.0)
    snap = rule.snapshot()
    rule.lam = 1.0
    rule.restore(snap)
    assert rule.lam == snap and rule.lam_at(3) == snap


def test_uncertainty_starts_at_the_logarithms_of_the_first_batch_and_the_weight_is_their_ratio():
    rule = balance.Uncertainty(1.25e-3, 100.0, 10)
    assert rule.needs_gap and rule.lam() is None and rule.applied(5) == 0.0
    rule.begin(0.04, 0.02)
    assert rule.lam() == pytest.approx(2.0) and rule.applied(5) == pytest.approx(1.0)
    rule.begin(1.0, 1.0)
    assert rule.lam() == pytest.approx(2.0)
    capped = balance.Uncertainty(1.25e-3, 1.5, 10)
    capped.begin(0.04, 0.02)
    assert capped.lam() == 1.5
    s0 = rule.s.detach().clone()
    rule.after_step(0.5, 0.01)
    assert not torch.equal(rule.s, s0)
    snap = rule.snapshot()
    rule.after_step(0.5, 0.01)
    rule.restore(snap)
    assert torch.equal(rule.s, snap["s"])
    rule.begin(math.nan, 1.0)
    rule.after_step(math.nan, 1.0)
    assert torch.equal(rule.s, snap["s"])


def test_the_energy_gap_is_the_energy_loss_minus_the_label_value():
    assert balance.energy_gap(-0.34, -0.36) == pytest.approx(0.02)


class Shrunk:
    """The constants of the Lagrangian rule, small enough to see the gate and the updates in a few steps."""

    def __enter__(self):
        self.saved = {k: getattr(balance, k) for k in ("GATE_LOSS", "GATE_FIRST", "GATE_STEP", "RAMP", "UPDATE_EVERY", "WINDOW")}
        balance.GATE_LOSS, balance.GATE_FIRST, balance.GATE_STEP, balance.RAMP, balance.UPDATE_EVERY, balance.WINDOW = 1.5, 4, 8, 4, 3, 3

    def __exit__(self, *args):
        for k, v in self.saved.items():
            setattr(balance, k, v)


def test_the_lagrangian_equals_labels_energy_fixed_until_its_gate_and_opens_by_the_loss():
    with Shrunk():
        rule = balance.Lagrangian(lambda_f=2.0, label_target=0.04, warm=4)
        fixed = balance.Fixed(2.0, 4)
        d, e = torch.tensor(0.09), torch.tensor(-0.3, dtype=torch.float64)
        for step in range(6):
            assert float(rule.loss(d, e, step)) == float(fixed.loss(d, e, step)) and rule.applied(step) == fixed.applied(step)
            rule.after_step(0.09, 0.0)
            rule.at_row(step)
            assert rule.gate_step is None
        for _ in range(3):
            rule.after_step(0.03, 0.0)
        rule.at_row(6)
        assert (rule.gate_step, rule.gate_by) == (6, "loss") and rule.nu == rule.nu0
        assert rule.lam_d(6) == rule.nu0 and rule.lam_d(10) == rule.lam_alm()


def test_the_lagrangian_gate_opens_by_the_step_and_the_multiplier_follows_the_constraint():
    with Shrunk():
        rule = balance.Lagrangian(lambda_f=2.0, label_target=0.04, warm=4)
        rule.at_row(7)
        assert rule.gate_step is None
        rule.at_row(8)
        assert (rule.gate_step, rule.gate_by) == (8, "step")
        for _ in range(3):
            rule.after_step(0.08, 0.0)
        assert rule.updates == 1 and rule.nu == pytest.approx(rule.nu0 * 2.0)
        for _ in range(3):
            rule.after_step(0.0, 0.0)
        assert rule.nu == pytest.approx(rule.nu0 * 1.0)
        for _ in range(3 * 20):
            rule.after_step(0.0, 0.0)
        assert rule.nu == 0.0
        snap = rule.snapshot()
        rule.after_step(0.1, 0.0)
        rule.restore(snap)
        assert rule.gate_step == 8 and not rule.window and rule.since == [] and rule.age == 0


def test_the_frozen_weight():
    assert balance.frozen_weight("labels-energy-adaptive", 3.5, [1.0, 2.0], 2) == 3.5
    assert balance.frozen_weight("labels-energy-uncertainty", 1.7, [1.0, 2.0, 3.0], 2) == 2.5
    assert balance.frozen_weight("labels-energy-lagrangian", 0.3, [1.0, 2.0, 3.0], 3) == 2.0
    assert balance.frozen_weight("labels-energy-fixed", None, [1.0], 1) is None and balance.frozen_weight("labels-only", None, [], 1) is None
