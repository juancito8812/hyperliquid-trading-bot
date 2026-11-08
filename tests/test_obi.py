"""OBI tracker: size-based imbalance, decay in wall-clock time, per-level weighting."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.obi import ObiConfig, ObiTracker  # noqa: E402


def _book(depth=1.0):
    bids = [(100.0 - i * 0.1, depth) for i in range(10)]
    asks = [(100.1 + i * 0.1, depth) for i in range(10)]
    return bids, asks


def test_symmetric_book_is_zero():
    t = ObiTracker()
    bids, asks = _book()
    snap = t.update(bids, asks, ts=1_000_000)
    assert snap["obi_1s"] == 0.0 and snap["obi_5s"] == 0.0
    assert t.value(1.0) == 0.0
    assert t.depth_usd == round(100.0 * 1.0 + 100.1 * 1.0, 2)


def test_bid_heavy_book_positive():
    t = ObiTracker()
    bids, asks = _book(depth=1.0)
    bids[0] = (99.9, 50.0)  # fat bid stack
    snap = t.update(bids, asks, ts=1_000_000)
    assert snap["obi_1s"] > 0.5 and t.gate_allows("long")


def test_ask_heavy_book_negative_and_gate_veto():
    t = ObiTracker()
    bids, asks = _book(depth=1.0)
    asks[0] = (100.1, 50.0)
    snap = t.update(bids, asks, ts=1_000_000)
    assert snap["obi_1s"] < -0.5
    assert not t.gate_allows("long", veto_under=-0.6)
    assert t.gate_allows("short", veto_under=-0.6)


def test_accumulation_then_decay_to_flat():
    """New snapshots accumulate; without new flow the decayed sum shrinks."""
    t = ObiTracker()
    bids, asks = _book()
    bids[0] = (99.9, 50.0)
    t.update(bids, asks, ts=0.0)
    for s in (0.5, 1.0, 1.5):
        t.update(bids, asks, ts=s)
    kept = t.bids[0]
    assert kept > 0
    # 6 s of silence with an empty book: half-life 1s -> ~1/64 of the sum
    t.update([], [], ts=6.0)
    assert abs(t.bids[0]) < kept / 20


def test_feed_outage_freezes_state_then_stale_resets():
    t = ObiTracker(ObiConfig(stale_s=5.0))
    bids, asks = _book()
    bids[0] = (99.9, 50.0)
    t.update(bids, asks, ts=0.0)
    v_before = t.value(1.0)
    # short outage (1 s): book state freezes (no filler volume invented)
    t.update([], [], ts=1.0)
    assert t.value(1.0) == v_before
    # long outage past stale_s: aggregates zeroed -> neutral
    t2 = ObiTracker(ObiConfig(stale_s=5.0))
    t2.update(bids, asks, ts=0.0)
    t2.update([], [], ts=100.0)
    assert t2.value(1.0) == 0.0


def test_top_of_book_dominates_full_depth():
    t = ObiTracker(ObiConfig(level_decay=100.0))  # effectively top level only
    bids, asks = _book()
    bids[5] = (99.5, 9999.0)  # huge passive size deep in the book
    snap = t.update(bids, asks, ts=0.0)
    assert abs(snap["obi_1s"]) < 0.2  # must not distort the top-of-book read
    t2 = ObiTracker(ObiConfig(level_decay=0.0))  # full depth, no weighting
    bids2, asks2 = _book()
    bids2[5] = (99.5, 9999.0)
    snap2 = t2.update(bids2, asks2, ts=0.0)
    assert abs(snap2["obi_1s"]) > 0.9  # naive full-depth averaging swallows it


def test_snapshot_state_and_errors():
    t = ObiTracker()
    st = t.snapshot_state()
    assert st["snapshots"] == 0 and st["obi_1s"] == 0.0
    try:
        ObiTracker(ObiConfig(half_lives=(0.0,)))
        assert False
    except ValueError:
        pass
    try:
        t.update([(100.0, -1.0)], [(100.1, 1.0)], ts=0.0)
        assert False
    except ValueError:
        pass
    try:
        t.update([(100.0,)], [(100.1, 1.0)], ts=0.0)
        assert False
    except ValueError:
        pass
