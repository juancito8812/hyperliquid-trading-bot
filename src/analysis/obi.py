"""Order-book imbalance (OBI) tracker with physical-time decay.

Design
------
Hyperliquid broadcast L2 snapshots (ws ``l2Book``). This module keeps *decayed
volume aggregates* at 2 half-lives (default 1 s and 5 s). Aggregates decay in
**wall-clock seconds**, not per update, so a burst of snapshots followed by a
feed outage reads exactly as flat book — physically.

    obi(win) = (Vbid_decayed - Vask_decayed) / (Vbid_decayed + Vask_decayed) ∈ [-1, 1]

Per-level weight ``exp(-idx * 0.35)`` keeps best-of-book dominant (the Rust
review flagged full-depth books averaging away the actionable top).

The live bot only has this if a recorder is running next to it — the REST
walker ``scripts/record_l2.py`` captures snapshots to ``.jsonl.gz`` for
backtests and for the optional strategy gate (``obi_gate: true``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

__all__ = ["ObiConfig", "ObiTracker", "obi_from_snapshot"]


@dataclass
class ObiConfig:
    """Half-lives in *seconds*; level weight decay per depth index."""

    half_lives: Sequence[float] = field(default_factory=lambda: (1.0, 5.0))
    level_decay: float = 0.35
    max_levels: int = 12
    stale_s: float = 30.0  # aggregates older than this are zeroed on update

    def validate(self) -> None:
        if any(h <= 0 for h in self.half_lives):
            raise ValueError("half_lives must be > 0")
        if self.max_levels < 1:
            raise ValueError("max_levels must be >= 1")


class ObiTracker:
    """Decayed best-of-book size imbalance with per-level weighting."""

    def __init__(self, cfg: ObiConfig | None = None) -> None:
        self.cfg = cfg or ObiConfig()
        self.cfg.validate()
        self.decays: List[float] = [math.log(2.0) / float(h) for h in self.cfg.half_lives]
        self.bids: List[float] = [0.0] * len(self.decays)  # decayed *sizes*
        self.asks: List[float] = [0.0] * len(self.decays)
        self.last_ts: float | None = None
        self.mid: float | None = None
        self.spread_bps: float = 0.0
        self.depth_usd: float = 0.0
        self.snapshots = 0

    # ── ingest ──────────────────────────────────────────────────────────
    def update(self, bids: Sequence[Sequence[float]], asks: Sequence[Sequence[float]],
               ts: float) -> Dict[str, float]:
        """Feed one L2 snapshot; ``bids``/``asks`` = [(price, size), best first]."""
        for lvl in tuple(bids) + tuple(asks):
            if len(lvl) < 2 or float(lvl[1]) < 0:
                raise ValueError("level must be (price, size>=0)")
        ts = float(ts)
        if self.last_ts is not None:
            dt = ts - self.last_ts
            if dt > 0:
                for i, d in enumerate(self.decays):
                    self.bids[i] *= math.exp(-d * dt)
                    self.asks[i] *= math.exp(-d * dt)
            if dt > self.cfg.stale_s:
                self.bids = [0.0] * len(self.decays)
                self.asks = [0.0] * len(self.decays)
        best_bid = best_ask = 0.0
        for i in range(len(self.decays)):
            vb = va = 0.0
            for idx, (price, size) in enumerate(bids[: self.cfg.max_levels]):
                vb += float(size) * math.exp(-idx * self.cfg.level_decay)
            for idx, (price, size) in enumerate(asks[: self.cfg.max_levels]):
                va += float(size) * math.exp(-idx * self.cfg.level_decay)
            self.bids[i] += vb
            self.asks[i] += va
        if bids and asks:
            bb, ba = float(bids[0][0]), float(asks[0][0])
            if bb > 0 and ba > 0:
                self.mid = (bb + ba) / 2
                self.spread_bps = round((ba - bb) / self.mid * 10000, 3) if self.mid else 0.0
        best_bid = float(bids[0][0]) * float(bids[0][1]) if bids else 0.0
        best_ask = float(asks[0][0]) * float(asks[0][1]) if asks else 0.0
        self.depth_usd = round(best_bid + best_ask, 2)
        self.last_ts = ts
        self.snapshots += 1
        out = {"ts": ts}
        for i, h in enumerate(self.cfg.half_lives):
            out[f"obi_{h:g}s"] = round(self.value(h), 4)
        out["spread_bps"] = self.spread_bps
        out["depth_usd"] = self.depth_usd
        return out

    # ── queries ─────────────────────────────────────────────────────────
    def value(self, half_life: float) -> float:
        i = self.decays.index(math.log(2.0) / float(half_life))
        b, a = self.bids[i], self.asks[i]
        if b + a <= 0:
            return 0.0
        return (b - a) / (b + a)

    def gate_allows(self, side: str, veto_under: float = -0.6) -> bool:
        """Long entry vetoed while bid/ask flow is sharply against it."""
        obi = self.value(self.cfg.half_lives[0])
        if side == "long" and obi < veto_under:
            return False
        if side == "short" and obi > -veto_under:
            return False
        return True

    def snapshot_state(self) -> dict:
        return {
            "snapshots": self.snapshots,
            "last_ts": self.last_ts,
            "obi_1s": round(self.value(self.cfg.half_lives[0]), 4)
            if self.snapshots else 0.0,
            "spread_bps": self.spread_bps,
        }


def obi_from_snapshot(rec: dict, half_lives: Sequence[float] = (1.0, 5.0)) -> float:
    """One-shot imbalance from a raw ``l2Book`` record (no state)."""
    tracker = ObiTracker(ObiConfig(half_lives=half_lives))
    bids = [(float(l["px"]), float(l["sz"])) for l in rec.get("bids", [])]
    asks = [(float(l["px"]), float(l["sz"])) for l in rec.get("asks", [])]
    ts = float(rec.get("ts", 0.0))
    if ts > 1e12:
        ts /= 1000.0
    tracker.update(bids, asks, ts)
    return tracker.value(half_lives[0])


def load_recording(path: str, limit: int | None = None) -> List[dict]:
    """Read a ``record_l2.py`` output (.jsonl or .jsonl.gz)."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    out: List[dict] = []
    with opener(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out
