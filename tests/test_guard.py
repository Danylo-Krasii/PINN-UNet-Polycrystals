"""The guard, the plateau rule, the rewind, the stop rule and the stall rule."""
import copy
import math
import types

import pytest
import torch

from src.training import guard


def fire(v, best=0.05, val_init=0.2, rewinds=0, max_rewinds=20, factor=3.0, midpoint=True, residual=None, lowest=math.inf):
    return guard.check(v, best, val_init, rewinds, max_rewinds, factor, midpoint, residual, lowest)


def test_the_validation_test():
    assert fire(0.05) == (False, "validation")
    assert fire(0.1) == (False, "validation")
    assert fire(0.1500001, midpoint=False) == (True, "validation")
    assert fire(0.15, midpoint=False) == (False, "validation")
    assert fire(math.nan) == (True, "validation") and fire(math.inf) == (True, "validation")
    assert fire(0.11, factor=2.0, midpoint=False) == (True, "validation")


def test_the_midpoint_test():
    best, v0 = 0.1, 0.2
    assert fire(0.15, best, v0) == (False, "validation")
    assert fire(0.1500001, best, v0) == (True, "midpoint")
    assert fire(0.1500001, best, v0, midpoint=False) == (False, "validation")
    assert fire(0.2, 0.03, 0.18) == (True, "validation")


def test_the_midpoint_test_for_energy_only_uses_the_energy_at_the_start_zero():
    assert fire(-0.1, best=-0.2, val_init=0.0, residual=0.5, lowest=0.5) == (False, "")
    assert fire(-0.0999, best=-0.2, val_init=0.0, residual=0.5, lowest=0.5) == (True, "midpoint")
    assert fire(0.1, best=-0.2, val_init=0.0, residual=0.5, lowest=0.5) == (True, "energy")
    assert fire(-0.2, best=-0.2, val_init=0.0, residual=1.6, lowest=0.5) == (True, "residual")


def test_nothing_fires_once_the_rewinds_are_used_up():
    assert fire(math.nan, rewinds=20) == (False, "")
    assert fire(math.nan, rewinds=19) == (True, "validation")
    assert fire(0.5, rewinds=3, max_rewinds=3, residual=1.0, lowest=0.1) == (False, "")


def test_the_base_rate_is_halved_to_the_floor_of_1e_minus_5():
    assert guard.halved_base(1.25e-4, 1.25e-4) == 6.25e-5
    assert guard.halved_base(1.25e-4, 3e-5) == 1.5e-5
    assert guard.halved_base(1.9e-5, 1.9e-5) == 1e-5
    assert guard.halved_base(1e-5, 1e-5) == 1e-5


def test_the_plateau_rule_halves_after_more_than_patience_validations_without_gain():
    rule = guard.Plateau(lambda v, best: v < best * (1 - 1e-4), 2)
    base = 1.25e-4
    bases = []
    for v in (1.0, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95):
        base = rule.step(v, base)
        bases.append(base)
    assert bases == [1.25e-4] * 4 + [6.25e-5] * 3 + [3.125e-5]
    for _ in range(40):
        base = rule.step(1.0, base)
    assert base == 1e-5


def test_the_rewind_restores_the_best_state_and_the_rate():
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    plateau = guard.Plateau(lambda v, best: v < best, 3)
    for _ in range(3):
        opt.zero_grad()
        net(torch.ones(2, 4)).sum().backward()
        opt.step()
    s = types.SimpleNamespace(best_state={k: v.detach().clone() for k, v in net.state_dict().items()}, best_opt=copy.deepcopy(opt.state_dict()), best_plateau=plateau.state_dict(),
                              best_base=1.25e-4, base=1.25e-4, best_step=300, step=500, t0=7, best_rule="snapshot")
    restored = []
    rule = types.SimpleNamespace(restore=restored.append)
    for _ in range(2):
        opt.zero_grad()
        (net(torch.ones(2, 4)) ** 2).sum().backward()
        opt.step()
    assert guard.rewind(net, opt, plateau, rule, s) == 6.25e-5 == s.base
    assert all(torch.equal(net.state_dict()[k], v) for k, v in s.best_state.items()) and restored == ["snapshot"] and s.t0 == 7
    s.best_step = 0
    guard.rewind(net, opt, plateau, rule, s)
    assert s.t0 == 500


@pytest.mark.parametrize("bad,step,patience,min_stop,expected", [(6000, 12000, 6000, 12000, True), (5999, 12000, 6000, 12000, False), (6000, 11999, 6000, 12000, False),
                                                                  (9000, 20000, None, 0, False), (0, 100, 0, 0, True)])
def test_the_stop_rule(bad, step, patience, min_stop, expected):
    assert guard.should_stop(bad, step, patience, min_stop) is expected


@pytest.mark.parametrize("steps,best_step,clipped,expected", [(96000, 8000, 48001, True), (96000, 8000, 48000, False), (96000, 8001, 96000, False), (100, 0, 100, True), (100, 8000, 0, False)])
def test_the_stall_rule(steps, best_step, clipped, expected):
    assert guard.is_stalled(steps, best_step, clipped) is expected
