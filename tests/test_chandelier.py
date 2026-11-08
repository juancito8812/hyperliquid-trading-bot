"""Chandelier stop: ratchet, contraction guard, tiered lock — no overshoot."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.chandelier import (  # noqa: E402
    ChandelierConfig, ChandelierStop, trail_series,
)


def test_ratchet_never_loosens():
    cfg = ChandelierConfig()
    s = ChandelierStop(entry=100.0, atr=1.0, is_long=True, cfg=cfg)
    s.update(102.0, 1.0)
    high = s.level
    s.update(103.0, 5.0)  # atr blow-out: raw stop would drop hard
    assert s.level >= high - 1e-12


def test_contraction_cap_limits_single_update():
    cfg = ChandelierConfig(max_step_pct=0.01)
    s = ChandelierStop(entry=100.0, atr=10.0, is_long=True, cfg=cfg)  # level 90, risk 30
    l0 = s.level
    s.update(100.0, 10.0)  # peak=100, raw = 100-30 = 70 -> cap bounds tightening
    assert s.level <= l0 + 100.0 * 0.01 + 1e-9
    assert s.level >= l0  # still tightened


def test_tiered_profit_lock_after_1r_and_2r():
    cfg = ChandelierConfig(atr_mult=3.0, tiers=((1.0, 0.25), (2.0, 1.0)))
    s = ChandelierStop(entry=100.0, atr=1.0, is_long=True, cfg=cfg)  # risk=3
    s.update(103.0, 1.0)  # +1R -> lock 0.25R = 100.75
    assert s.level >= 100.75 - 1e-9
    s.update(107.5, 1.0)  # +2.5R -> lock 1R = 101 (already above)
    assert s.level >= 101.0 - 1e-9


def test_lock_never_overshoots_current_price():
    cfg = ChandelierConfig(atr_mult=3.0, tiers=((1.0, 0.25),))
    s = ChandelierStop(entry=100.0, atr=1.0, is_long=True, cfg=cfg)  # risk=3
    s.update(103.5, 1.0)  # +1.17R -> locked 100.75
    assert s.level >= 100.75 - 1e-9
    assert s.level <= 103.5 + 1e-9


def test_short_side_mirror():
    cfg = ChandelierConfig(atr_mult=3.0, tiers=((1.0, 0.25),))
    s = ChandelierStop(entry=100.0, atr=1.0, is_long=False, cfg=cfg)  # risk=3, level=103
    s.update(97.0, 1.0)  # -1R -> locked at 99.25
    assert s.level <= 99.25 + 1e-9
    assert not s.hit(98.0)
    assert s.hit(99.5)  # price back above the stop -> exit short


def test_short_ratchet_never_loosens():
    cfg = ChandelierConfig()
    s = ChandelierStop(entry=100.0, atr=1.0, is_long=False, cfg=cfg)  # level=103
    s.update(98.0, 1.0)
    low = s.level
    s.update(97.0, 5.0)  # atr blowout pushes raw level up
    assert s.level <= low + 1e-12


def test_hit_and_r_multiple():
    s = ChandelierStop(entry=100.0, atr=1.0, is_long=True)
    s.update(102.0, 1.0)
    assert s.hit(s.level - 0.01)
    assert abs(s.r_multiple(103.0) - 1.0) < 1e-9


def test_trail_series_replay():
    stops = trail_series([100.0, 101.0, 102.0, 101.5, 103.0], atr=1.0, is_long=True)
    assert stops[0] == 97.0
    assert all(stops[i] <= stops[i + 1] + 1e-12 for i in range(len(stops) - 1))


def test_validation_errors():
    try:
        ChandelierStop(entry=0.0, atr=1.0, is_long=True)
        assert False
    except ValueError:
        pass
    bad = ChandelierConfig(tiers=((2.0, 0.5), (1.0, 0.25)))
    try:
        ChandelierStop(100.0, 1.0, True, bad)
        assert False
    except ValueError:
        pass
    try:
        ChandelierStop(100.0, 1.0, True).update(0.0, 1.0)
        assert False
    except ValueError:
        pass
