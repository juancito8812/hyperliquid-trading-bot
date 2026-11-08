"""Candle buffering for strategies that need OHLCV history.

The engine hands strategies a single ``MarketData`` price tick — no candles.
This module rebuilds candles from those ticks and can seed the buffer with
historical candles from the exchange, which is what the EMA/ADX strategy needs
to compute EMAs / ATR / ADX without an external data feed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

__all__ = ["Candle", "CandleBuffer", "INTERVAL_MS"]


@dataclass
class Candle:
    """A single OHLCV bar. ``ts`` is the bar's open time in milliseconds."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _to_ms(timestamp: float) -> int:
    """Normalise a timestamp in seconds or milliseconds to milliseconds."""
    ts = float(timestamp)
    return int(ts if ts > 1e11 else ts * 1000.0)


class CandleBuffer:
    """Rolling buffer of closed candles, built from ticks and/or history."""

    def __init__(self, interval: str = "15m", max_candles: int = 500):
        if interval not in INTERVAL_MS:
            raise ValueError(
                f"unsupported interval {interval!r}. Options: {sorted(INTERVAL_MS)}"
            )
        if max_candles < 10:
            raise ValueError("max_candles must be >= 10")
        self.interval = interval
        self.interval_ms = INTERVAL_MS[interval]
        self.max_candles = max_candles
        self.candles: List[Candle] = []
        self._current: Optional[Candle] = None

    # ── helpers ─────────────────────────────────────────────────────────
    def bucket_start(self, timestamp: float) -> int:
        """Open time (ms) of the bar that ``timestamp`` belongs to."""
        ts_ms = _to_ms(timestamp)
        return ts_ms - (ts_ms % self.interval_ms)

    def _trim(self) -> None:
        if len(self.candles) > self.max_candles:
            del self.candles[: len(self.candles) - self.max_candles]

    # ── ingestion ───────────────────────────────────────────────────────
    def seed(self, rows: Optional[Iterable[Any]]) -> int:
        """Load historical candles.

        Accepts Hyperliquid ``candles_snapshot`` dicts (``t/o/h/l/c/v``),
        ccxt-style lists (``[t, o, h, l, c, v]``) or ``Candle`` objects.
        Returns the number of candles loaded, oldest first.
        """
        parsed: List[Candle] = []
        for row in rows or []:
            candle = self._parse_row(row)
            if candle is not None:
                parsed.append(candle)

        parsed.sort(key=lambda c: c.ts)
        deduped: Dict[int, Candle] = {c.ts: c for c in parsed}
        ordered = [deduped[ts] for ts in sorted(deduped)]

        # Drop a trailing still-open bar: the last history entry is usually the
        # bar currently forming, and feeding it as closed would double count.
        now_bucket = self.bucket_start(time.time())
        if ordered and ordered[-1].ts >= now_bucket:
            ordered.pop()

        self.candles = ordered[-self.max_candles :]
        self._current = None
        logger.info(
            "📚 CandleBuffer seeded: %s candles @ %s (last close=%s)",
            len(self.candles),
            self.interval,
            self.candles[-1].close if self.candles else "n/a",
        )
        return len(self.candles)

    @staticmethod
    def _parse_row(row: Any) -> Optional[Candle]:
        try:
            if isinstance(row, Candle):
                return row
            if isinstance(row, dict):
                ts = row.get("t", row.get("timestamp", row.get("time")))
                o = row.get("o", row.get("open"))
                h = row.get("h", row.get("high"))
                l = row.get("l", row.get("low"))
                c = row.get("c", row.get("close"))
                v = row.get("v", row.get("volume", 0.0))
            elif isinstance(row, (list, tuple)) and len(row) >= 5:
                ts, o, h, l, c = row[:5]
                v = row[5] if len(row) > 5 else 0.0
            else:
                return None
            if ts is None or c is None:
                return None
            return Candle(
                ts=_to_ms(ts),
                open=float(o if o is not None else c),
                high=float(h if h is not None else c),
                low=float(l if l is not None else c),
                close=float(c),
                volume=float(v or 0.0),
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.warning("Skipping malformed candle row %r: %s", row, exc)
            return None

    def add_price(self, price: float, timestamp: Optional[float] = None) -> bool:
        """Feed a tick. Returns ``True`` when that tick opened a new bar.

        The returned ``True`` means the previous bar is now closed and appended
        to :attr:`candles` — the moment a bar-based strategy should recompute
        indicators and look for entries.
        """
        price = float(price)
        if price <= 0:
            return False
        ts_ms = _to_ms(timestamp if timestamp is not None else time.time())
        bucket = ts_ms - (ts_ms % self.interval_ms)

        if self._current is None:
            self._current = Candle(bucket, price, price, price, price, 0.0)
            return False

        if bucket == self._current.ts:
            self._current.high = max(self._current.high, price)
            self._current.low = min(self._current.low, price)
            self._current.close = price
            return False

        if bucket < self._current.ts:  # out-of-order tick: keep the bar alive
            self._current.high = max(self._current.high, price)
            self._current.low = min(self._current.low, price)
            self._current.close = price
            return False

        self.candles.append(self._current)
        self._trim()
        self._current = Candle(bucket, price, price, price, price, 0.0)
        return True

    # ── accessors ───────────────────────────────────────────────────────
    @property
    def current(self) -> Optional[Candle]:
        """The still-forming bar (not used for signals)."""
        return self._current

    def __len__(self) -> int:
        return len(self.candles)

    def ohlcv(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """(highs, lows, closes, opens) of the closed candles."""
        highs = [c.high for c in self.candles]
        lows = [c.low for c in self.candles]
        closes = [c.close for c in self.candles]
        opens = [c.open for c in self.candles]
        return highs, lows, closes, opens
