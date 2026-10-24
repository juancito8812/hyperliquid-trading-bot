"""EMA cross + ADX filter strategy.

    from strategies import create_strategy
    strategy = create_strategy("ema_adx", {"symbol": "SOL", "interval": "15m"})
"""

from .candles import Candle, CandleBuffer, INTERVAL_MS
from .indicators import adx, atr, compute, ema, true_range, wilder_smooth
from .strategy import EmaAdxConfig, EmaAdxStrategy, hyperliquid_history_provider

__all__ = [
    "Candle",
    "CandleBuffer",
    "INTERVAL_MS",
    "EmaAdxConfig",
    "EmaAdxStrategy",
    "hyperliquid_history_provider",
    "adx",
    "atr",
    "compute",
    "ema",
    "true_range",
    "wilder_smooth",
]
