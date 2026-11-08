"""Chandelier trailing stop with a profit-lock ladder and a contraction guard.

Ported from the reviewed Rust engine (``hyperliquid-bot/src/trend.rs``). The
rules that matter and are easy to get wrong:

* **Ratchet only.** The stop never moves against the position, not even when
  ATR expands and the raw chandelier level drops back.
* **Contraction guard.** One update may move the stop at most
  ``max_step_pct`` of price, so a single fat-finger print cannot yank the stop
  to the entry. The guard only limits tightening; it never loosens.
* **Tiered profit lock.** Once the trade has run ``r`` risk units, the stop is
  floored at ``entry + locked_r * risk`` for longs (mirrored for shorts).
* **No overshoot.** The locked floor is capped at the current price, so the
  stop can never sit beyond the market (which would read as an instant exit).

The stop is pure state: feed it prices/ATR, read ``level`` and ``hit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

__all__ = ["ChandelierConfig", "ChandelierStop"]


@dataclass
class ChandelierConfig:
    """Tuning for :class:`ChandelierStop`."""

    atr_mult: float = 3.0
    # Tighter multiple applied once the trade is well in profit.
    tighten_mult: float = 2.0
    tighten_after_r: float = 2.0
    # Max distance one update may tighten, as a fraction of price.
    max_step_pct: float = 0.02
    # (r_multiple_reached, r_multiple_locked) pairs, ascending.
    tiers: Sequence[Tuple[float, float]] = field(
        default_factory=lambda: ((1.0, 0.25), (2.0, 1.0))
    )

    def validate(self) -> None:
        if self.atr_mult <= 0 or self.tighten_mult <= 0:
            raise ValueError("atr_mult and tighten_mult must be > 0")
        if not 0 < self.max_step_pct < 1:
            raise ValueError("max_step_pct must be within (0, 1)")
        rs = [float(r) for r, _ in self.tiers]
        if rs != sorted(rs):
            raise ValueError("tiers must be ordered by r_multiple")
        if any(float(lock) < 0 for _, lock in self.tiers):
            raise ValueError("tier locked_r must be >= 0")


class ChandelierStop:
    """Stateful trailing stop for one position."""

    def __init__(
        self,
        entry: float,
        atr: float,
        is_long: bool,
        cfg: ChandelierConfig | None = None,
        risk_distance: float | None = None,
    ) -> None:
        if entry <= 0:
            raise ValueError("entry must be > 0")
        cfg = cfg or ChandelierConfig()
        cfg.validate()
        if atr <= 0:
            raise ValueError("atr must be > 0")
        self.cfg = cfg
        self.entry = float(entry)
        self.is_long = bool(is_long)
        # R unit: the initial stop distance unless the caller pins one.
        self.risk = float(risk_distance) if risk_distance else atr * cfg.atr_mult
        self.atr = float(atr)
        self.peak = self.entry
        self.level = self.entry - self.risk if self.is_long else self.entry + self.risk
        self.updates = 0

    # ── core ────────────────────────────────────────────────────────────
    def _raw_level(self, atr: float) -> float:
        mult = self.cfg.atr_mult
        if self.is_long:
            progress = (self.peak - self.entry) / self.risk if self.risk else 0.0
            if progress >= self.cfg.tighten_after_r:
                mult = self.cfg.tighten_mult
            return self.peak - atr * mult
        progress = (self.entry - self.peak) / self.risk if self.risk else 0.0
        if progress >= self.cfg.tighten_after_r:
            mult = self.cfg.tighten_mult
        return self.peak + atr * mult

    def _locked_level(self) -> float:
        progress_r = (
            (self.peak - self.entry) if self.is_long else (self.entry - self.peak)
        ) / self.risk if self.risk else 0.0
        locked_r = 0.0
        for reached, lock in self.cfg.tiers:
            if progress_r >= float(reached):
                locked_r = max(locked_r, float(lock))
        if locked_r <= 0:
            return self.level
        offset = locked_r * self.risk
        return self.entry + offset if self.is_long else self.entry - offset

    def update(self, price: float, atr: float) -> float:
        """Feed the latest price; returns the (ratcheted) stop level."""
        if price <= 0:
            raise ValueError("price must be > 0")
        if atr <= 0:
            raise ValueError("atr must be > 0")
        self.atr = float(atr)
        self.peak = max(self.peak, price) if self.is_long else min(self.peak, price)

        if self.is_long:
            # contraction cap bounds tightening from the raw chandelier; the
            # tier lock is a promise and may step past the cap.
            raw = min(self._raw_level(atr),
                      self.level + price * self.cfg.max_step_pct, price)
            candidate = max(raw, min(self._locked_level(), price))
            self.level = max(self.level, candidate)
        else:
            raw = max(self._raw_level(atr),
                      self.level - price * self.cfg.max_step_pct, price)
            candidate = min(raw, max(self._locked_level(), price))
            self.level = min(self.level, candidate)
        self.updates += 1
        return self.level

    # ── queries ─────────────────────────────────────────────────────────
    def hit(self, price: float) -> bool:
        return price <= self.level if self.is_long else price >= self.level

    def r_multiple(self, price: float) -> float:
        if self.risk <= 0:
            return 0.0
        move = price - self.entry if self.is_long else self.entry - price
        return move / self.risk

    def snapshot(self, price: float | None = None) -> dict:
        out = {
            "level": self.level,
            "peak": self.peak,
            "risk": self.risk,
            "updates": self.updates,
            "is_long": self.is_long,
        }
        if price is not None:
            out["r_multiple"] = round(self.r_multiple(price), 3)
        return out


def trail_series(
    prices: Sequence[float], atr: float, is_long: bool, cfg: ChandelierConfig | None = None
) -> List[float]:
    """Convenience for tests/analysis: replay prices, return the stop series."""
    prices = list(prices)
    track = ChandelierStop(prices[0], atr, is_long, cfg)
    return [track.update(p, atr) for p in prices]
