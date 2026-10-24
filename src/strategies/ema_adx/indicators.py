"""Pure-numpy indicator maths for the EMA/ADX strategy.

Mirrors the Wilder smoothing of the original pandas implementation
(``ewm(alpha=1/period, adjust=False)`` for ATR/ADX, ``ewm(span=period)`` for
the EMAs) so live signals can be computed from a plain list of candles and
match a pandas backtest bar-for-bar.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Anything array-like: list, tuple or ndarray.
Values = Union[Sequence[float], "np.ndarray"]

__all__ = [
    "ema",
    "true_range",
    "wilder_smooth",
    "atr",
    "adx",
    "compute",
]


def _arr(values: Values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D sequence of floats")
    return arr


def ema(values: Values, period: int) -> np.ndarray:
    """Exponential moving average, ``adjust=False`` (seed = first value)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    v = _arr(values)
    if v.size == 0:
        raise ValueError("ema needs at least one value")
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, v.size):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_smooth(values: Values, period: int) -> np.ndarray:
    """Wilder's smoothing (``alpha = 1 / period``)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    v = _arr(values)
    if v.size == 0:
        raise ValueError("wilder_smooth needs at least one value")
    alpha = 1.0 / period
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, v.size):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def true_range(high: Values, low: Values, close: Values) -> np.ndarray:
    """True range series (first bar falls back to high - low)."""
    h, l, c = _arr(high), _arr(low), _arr(close)
    if not (h.size == l.size == c.size):
        raise ValueError("high/low/close must have the same length")
    if c.size == 0:
        raise ValueError("true_range needs at least one bar")
    tr = np.empty_like(c)
    tr[0] = h[0] - l[0]
    if c.size > 1:
        prev_close = c[:-1]
        tr[1:] = np.maximum.reduce(
            [h[1:] - l[1:], np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close)]
        )
    return tr


def atr(high: Values, low: Values, close: Values, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder)."""
    return wilder_smooth(true_range(high, low, close), period)


def adx(high: Values, low: Values, close: Values, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index plus +DI / -DI (Wilder)."""
    h, l, c = _arr(high), _arr(low), _arr(close)
    if c.size < 2:
        zeros = np.zeros_like(c)
        return zeros, zeros, zeros

    tr = true_range(h, l, c)
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate(([0.0], plus_dm))
    minus_dm = np.concatenate(([0.0], minus_dm))

    atr_smooth = wilder_smooth(tr, period)
    plus_smooth = wilder_smooth(plus_dm, period)
    minus_smooth = wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr_smooth > 0, 100.0 * plus_smooth / atr_smooth, 0.0)
        minus_di = np.where(atr_smooth > 0, 100.0 * minus_smooth / atr_smooth, 0.0)
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)

    return wilder_smooth(dx, period), plus_di, minus_di


def compute(
    highs: Values,
    lows: Values,
    closes: Values,
    *,
    ema_fast: int = 3,
    ema_slow: int = 9,
    atr_period: int = 14,
    adx_period: int = 14,
    min_bars: int = 30,
) -> Optional[Dict[str, float]]:
    """Compute the indicator snapshot consumed by the strategy.

    Returns ``None`` while there is not enough history, otherwise a dict with
    the same keys the original (pandas) implementation produced.
    """
    closes_arr = _arr(closes)
    need = max(min_bars, ema_slow + 2, adx_period * 2)
    if closes_arr.size < need:
        return None

    fast = ema(closes_arr, ema_fast)
    slow = ema(closes_arr, ema_slow)
    atr_series = atr(highs, lows, closes_arr, atr_period)
    adx_series, _plus_di, _minus_di = adx(highs, lows, closes_arr, adx_period)

    latest_fast = float(fast[-1])
    latest_slow = float(slow[-1])
    prev_fast = float(fast[-2])
    prev_slow = float(slow[-2])

    return {
        "ema_fast": latest_fast,
        "ema_slow": latest_slow,
        "ema_fast_prev": prev_fast,
        "ema_slow_prev": prev_slow,
        "atr": float(atr_series[-1]),
        "adx": float(adx_series[-1]),
        "close": float(closes_arr[-1]),
        "crossover_bull": (prev_fast <= prev_slow) and (latest_fast > latest_slow),
        "crossover_bear": (prev_fast >= prev_slow) and (latest_fast < latest_slow),
    }
