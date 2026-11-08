from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import random

import pytest

from analysis.pbo import cscv_pbo


def test_perfectly_skilled_config_gives_low_pbo():
    """Config 0 has a real positive edge; the rest bleed slowly."""
    rng = random.Random(1)
    rows = []
    for t in range(2000):
        base = [rng.gauss(0.001, 0.01), rng.gauss(-0.0005, 0.02),
                rng.gauss(-0.0004, 0.02), rng.gauss(-0.0006, 0.02)]
        rows.append(base)
    res = cscv_pbo(rows, n_blocks=10)
    assert res.pbo < 0.1
    assert res.n_combos == 252  # C(10,5)

def test_overfit_noise_gives_pbo_near_half():
    """Pure noise: PBO ≈ 0.5 *across matrices*; one matrix has sd ~0.28."""
    pbos = [cscv_pbo([[rng.gauss(0.0, 0.01) for _ in range(4)]
                      for _ in range(2000)], n_blocks=10).pbo
            for rng in (random.Random(s) for s in range(8))]
    mean = sum(pbos) / len(pbos)
    assert abs(mean - 0.5) < 0.2   # no selection skill on average
    assert all(0.0 < p < 1.0 for p in pbos)

def test_oos_rank_pct_agrees_with_pbo_direction():
    rows = []
    rng = random.Random(4)
    for t in range(1200):
        one = rng.gauss(0.001, 0.005)          # genuinely better everywhere
        others = [rng.gauss(-0.0005, 0.02) for _ in range(3)]
        rows.append([one] + others)
    res = cscv_pbo(rows, n_blocks=10)
    assert res.mean_oos_rank_pct > 0.7
    assert res.pbo < 0.2

def test_shape_guards():
    with pytest.raises(ValueError):
        cscv_pbo([[0.1], [0.1]], n_blocks=10)
    with pytest.raises(ValueError):
        cscv_pbo([[0.1, 0.2]] * 5, n_blocks=10)
    with pytest.raises(ValueError):
        cscv_pbo([[0.1, 0.2]] * 2000, n_blocks=9)  # odd block count
    with pytest.raises(ValueError):
        cscv_pbo([[0.1, 0.2, 0.3], [0.1, 0.2]], n_blocks=10)  # ragged
