"""EMA cross + ADX-filter strategy for the Hyperliquid engine.

Ported from the standalone SOL/ETH/BTC bot so it runs on this engine's
execution path (real fills via userFills, grouped TPSL, risk manager).

Rules — identical to the original bot:

- Entry: EMA(fast) crosses EMA(slow) on a *closed* candle while ADX >= threshold.
- Size: ``balance * risk_per_trade / (ATR * atr_sl_multiplier)``.
- Stop loss: ``entry -/+ ATR * atr_sl_multiplier``.
- Take profit: ``entry +/- ATR * atr_sl_multiplier * risk_reward_ratio``.
- SAR: price moves ``ATR * sar_atr_multiplier`` against the position while
  ADX >= threshold -> exit and reverse.

Because the engine only hands strategies a price tick, candles are rebuilt
from ticks and seeded from the exchange's own history (see ``candles.py``).
Exits are evaluated on every tick (intrabar, like the original 15s loop),
entries only on candle close.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from interfaces.strategy import (
    MarketData,
    Position,
    SignalType,
    TradingSignal,
    TradingStrategy,
)

from .candles import INTERVAL_MS, CandleBuffer
from .indicators import compute as compute_indicators

logger = logging.getLogger(__name__)

__all__ = ["EmaAdxConfig", "EmaAdxStrategy", "hyperliquid_history_provider"]

# HL coin names have no quote suffix: "SOL/USDC" -> "SOL".
_QUOTE_SUFFIXES = ("/USDC", "/USD", "-PERP", "/USDT")

_INFO_CACHE: Dict[str, Any] = {}

HistoryProvider = Callable[[str, str, int], List[Any]]


def normalize_asset(symbol: str) -> str:
    """Normalise a config symbol to the exchange's asset name."""
    if not symbol:
        raise ValueError("symbol cannot be empty")
    name = str(symbol).strip().upper()
    for suffix in _QUOTE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def hyperliquid_history_provider(testnet: bool = True) -> HistoryProvider:
    """Return a provider that fetches candles straight from Hyperliquid's /info.

    Uses httpx directly (not the SDK) so it ignores ambient SOCKS proxy settings
    that would otherwise need PySocks installed.
    """

    base_url = (
        "https://api.hyperliquid-testnet.xyz/info"
        if testnet
        else "https://api.hyperliquid.xyz/info"
    )

    def provider(symbol: str, interval: str, limit: int) -> List[Any]:
        import httpx

        interval_ms = INTERVAL_MS[interval]
        end = int(time.time() * 1000)
        start = end - interval_ms * (limit + 2)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start,
                "endTime": end,
            },
        }
        with httpx.Client(timeout=20.0, trust_env=False) as client:
            response = client.post(base_url, json=payload)
            response.raise_for_status()
            data = response.json()
        return data if isinstance(data, list) else []

    return provider


@dataclass
class EmaAdxConfig:
    """Strategy parameters (same meaning as the original bot's config.py)."""

    symbol: str = "SOL"
    interval: str = "15m"

    ema_fast: int = 3
    ema_slow: int = 9
    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 23.0

    atr_sl_multiplier: float = 2.0
    risk_reward_ratio: float = 3.0
    sar_atr_multiplier: float = 1.2

    risk_per_trade: float = 0.01
    max_position_pct: float = 0.9  # notional cap vs available balance
    min_notional: float = 10.0  # exchange minimum order value (USD)

    warmup_candles: int = 200
    order_cooldown_s: float = 30.0
    pending_reverse_ttl_s: float = 180.0

    testnet: bool = True
    dex: Optional[str] = None

    def validate(self) -> None:
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be < ema_slow")
        if self.adx_threshold < 0 or self.adx_threshold > 100:
            raise ValueError("adx_threshold must be within 0-100")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.2:
            raise ValueError("risk_per_trade must be within (0, 0.2]")
        if self.atr_sl_multiplier <= 0:
            raise ValueError("atr_sl_multiplier must be > 0")
        if self.risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio must be > 0")
        if self.interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval {self.interval!r}")
        if self.warmup_candles < 40:
            raise ValueError("warmup_candles must be >= 40")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EmaAdxConfig":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known - {"type"}
        if unknown:
            logger.debug("Ignoring unknown EMA/ADX config keys: %s", sorted(unknown))
        clean = {k: v for k, v in data.items() if k in known}
        cfg = cls(**clean)
        cfg.symbol = normalize_asset(cfg.symbol)
        cfg.validate()
        return cfg


class EmaAdxStrategy(TradingStrategy):
    """See module docstring for the rules."""

    def __init__(
        self,
        config: Dict[str, Any],
        history_provider: Optional[HistoryProvider] = None,
    ):
        super().__init__("ema_adx", dict(config or {}))
        self.cfg = EmaAdxConfig.from_dict(config)
        self.asset = self.cfg.symbol
        self.buffer = CandleBuffer(
            interval=self.cfg.interval,
            max_candles=max(self.cfg.warmup_candles + 50, 300),
        )
        self._history_provider = history_provider or hyperliquid_history_provider(
            self.cfg.testnet
        )
        self._seeded = False
        self._seed_attempts = 0
        self._indicators: Optional[Dict[str, float]] = None
        self._inflight_until = 0.0
        self._pending_reverse: Optional[Dict[str, Any]] = None
        self._last_bar_ts: Optional[int] = None
        self.stats: Dict[str, Any] = {
            "signals": 0,
            "entries": 0,
            "exits": 0,
            "warmup_ok": False,
            "last_reason": "",
        }

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        super().start()
        self._seed_history()

    def stop(self) -> None:
        super().stop()

    def _seed_history(self) -> None:
        if self._seeded:
            return
        self._seed_attempts += 1
        try:
            rows = self._history_provider(
                self.cfg.symbol, self.cfg.interval, self.cfg.warmup_candles
            )
            loaded = self.buffer.seed(rows)
            self._seeded = loaded >= 40
            self.stats["warmup_ok"] = self._seeded
            if self._seeded:
                self._refresh_indicators()
            else:
                logger.warning(
                    "⚠️ EMA/ADX warm-up incomplete: %s/%s candles",
                    loaded,
                    self.cfg.warmup_candles,
                )
        except Exception as exc:
            logger.warning("⚠️ EMA/ADX history warm-up failed: %s", exc)

    def _refresh_indicators(self) -> None:
        highs, lows, closes, _opens = self.buffer.ohlcv()
        self._indicators = compute_indicators(
            highs,
            lows,
            closes,
            ema_fast=self.cfg.ema_fast,
            ema_slow=self.cfg.ema_slow,
            atr_period=self.cfg.atr_period,
            adx_period=self.cfg.adx_period,
        )

    # ── main entry point ────────────────────────────────────────────────
    def generate_signals(
        self, market_data: MarketData, positions: List[Position], balance: float
    ) -> List[TradingSignal]:
        if not self.is_active or market_data is None:
            return []

        price = float(getattr(market_data, "price", 0) or 0)
        if price <= 0:
            return []

        if not self._seeded:
            if self._seed_attempts < 5:
                self._seed_history()
            if not self._seeded:
                return []

        bar_closed = self.buffer.add_price(price, getattr(market_data, "timestamp", None))
        if bar_closed:
            self._refresh_indicators()

        now = time.time()
        position = self._find_position(positions)
        if position is not None:
            return self._exit_signals(position, price, now)
        return self._entry_signals(price, float(balance or 0.0), now, bar_closed)

    # ── signals ─────────────────────────────────────────────────────────
    def _entry_signals(
        self, price: float, balance: float, now: float, bar_closed: bool
    ) -> List[TradingSignal]:
        if now < self._inflight_until:
            return []
        ind = self._indicators
        if ind is None or ind["atr"] <= 0 or balance <= 0:
            return []

        side: Optional[str] = None
        reason = ""

        pending = self._pending_reverse
        if pending is not None:
            self._pending_reverse = None
            if now - float(pending["ts"]) <= self.cfg.pending_reverse_ttl_s:
                side = pending["side"]
                reason = f"sar_reverse_from_{pending['reason']}"

        if side is None and bar_closed and ind["adx"] >= self.cfg.adx_threshold:
            if ind["crossover_bull"]:
                side, reason = "buy", "ema_cross_up"
            elif ind["crossover_bear"]:
                side, reason = "sell", "ema_cross_down"

        if side is None:
            return []

        size = self._position_size(balance, ind["atr"], price)
        if size <= 0:
            logger.info(
                "⏭️ EMA/ADX skipped %s entry: size below exchange minimum "
                "(balance=%.2f atr=%.4f price=%.4f)",
                self.asset,
                balance,
                ind["atr"],
                price,
            )
            return []

        signal = TradingSignal(
            signal_type=SignalType.BUY if side == "buy" else SignalType.SELL,
            asset=self.asset,
            size=size,
            price=None,  # market order
            reason=reason,
            metadata={
                "adx": round(ind["adx"], 2),
                "atr": round(ind["atr"], 6),
                "ema_fast": round(ind["ema_fast"], 6),
                "ema_slow": round(ind["ema_slow"], 6),
                "strategy": "ema_adx",
            },
            dex=self.cfg.dex,
        )
        self._inflight_until = now + self.cfg.order_cooldown_s
        self.stats["signals"] += 1
        self.stats["entries"] += 1
        self.stats["last_reason"] = reason
        logger.info(
            "📈 EMA/ADX %s %s size=%.6f @ ~%.4f (adx=%.1f atr=%.4f reason=%s)",
            "BUY" if side == "buy" else "SELL",
            self.asset,
            size,
            price,
            ind["adx"],
            ind["atr"],
            reason,
        )
        return [signal]

    def _exit_signals(
        self, position: Position, price: float, now: float
    ) -> List[TradingSignal]:
        ind = self._indicators
        if ind is None or ind["atr"] <= 0 or now < self._inflight_until:
            return []

        entry = float(position.entry_price or 0)
        if entry <= 0:
            return []

        atr_value = ind["atr"]
        sl_distance = atr_value * self.cfg.atr_sl_multiplier
        tp_distance = sl_distance * self.cfg.risk_reward_ratio
        sar_distance = atr_value * self.cfg.sar_atr_multiplier
        is_long = float(position.size) > 0

        if is_long:
            if price <= entry - sl_distance:
                return self._close(position, price, now, "stop_loss")
            if price >= entry + tp_distance:
                return self._close(position, price, now, "take_profit")
            if price <= entry - sar_distance and ind["adx"] >= self.cfg.adx_threshold:
                return self._close(position, price, now, "sar", reverse_side="sell")
        else:
            if price >= entry + sl_distance:
                return self._close(position, price, now, "stop_loss")
            if price <= entry - tp_distance:
                return self._close(position, price, now, "take_profit")
            if price >= entry + sar_distance and ind["adx"] >= self.cfg.adx_threshold:
                return self._close(position, price, now, "sar", reverse_side="buy")
        return []

    def _close(
        self,
        position: Position,
        price: float,
        now: float,
        reason: str,
        reverse_side: Optional[str] = None,
    ) -> List[TradingSignal]:
        if reverse_side:
            self._pending_reverse = {"side": reverse_side, "ts": now, "reason": reason}
        self._inflight_until = now + self.cfg.order_cooldown_s
        self.stats["signals"] += 1
        self.stats["exits"] += 1
        self.stats["last_reason"] = reason
        logger.info(
            "📉 EMA/ADX CLOSE %s size=%.6f @ ~%.4f (reason=%s)",
            self.asset,
            abs(float(position.size)),
            price,
            reason,
        )
        return [
            TradingSignal(
                signal_type=SignalType.CLOSE,
                asset=position.asset or self.asset,
                size=abs(float(position.size)),
                price=None,
                reason=reason,
                metadata={"strategy": "ema_adx", "reason": reason},
                dex=getattr(position, "dex", None) or self.cfg.dex,
            )
        ]

    # ── helpers ─────────────────────────────────────────────────────────
    def _position_size(self, balance: float, atr_value: float, price: float) -> float:
        stop_distance = atr_value * self.cfg.atr_sl_multiplier
        if stop_distance <= 0 or price <= 0:
            return 0.0
        risk_amount = balance * self.cfg.risk_per_trade
        size = risk_amount / stop_distance
        max_size = (balance * self.cfg.max_position_pct) / price
        size = min(size, max_size)
        if size * price < self.cfg.min_notional:
            return 0.0
        return round(size, 4)

    def _find_position(self, positions: Optional[List[Position]]) -> Optional[Position]:
        for pos in positions or []:
            if not pos or abs(float(getattr(pos, "size", 0) or 0)) <= 0:
                continue
            try:
                if normalize_asset(pos.asset) == self.asset:
                    return pos
            except ValueError:
                continue
        return None

    # ── hooks ───────────────────────────────────────────────────────────
    def on_trade_executed(
        self, signal: TradingSignal, executed_price: float, executed_size: float
    ) -> None:
        logger.info(
            "✅ EMA/ADX fill %s %s size=%.6f @ %.4f",
            signal.signal_type.value,
            signal.asset,
            executed_size,
            executed_price,
        )

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        logger.error("❌ EMA/ADX error: %s (context=%s)", error, context)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": self.is_active,
            "symbol": self.asset,
            "interval": self.cfg.interval,
            "candles": len(self.buffer),
            "warmup_ok": self._seeded,
            "indicators": self._indicators,
            "pending_reverse": self._pending_reverse,
            "stats": dict(self.stats),
        }
