"""End-to-end simulation: real candles -> engine -> orders (no network, no keys).

Drives TradingEngine._handle_price_update with a fake exchange so the whole
chain (strategy signals -> _execute_signal -> _place_order/_close_positions ->
immediate fill -> position update) is exercised.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.engine import TradingEngine  # noqa: E402
from interfaces.exchange import Order  # noqa: E402
from interfaces.strategy import MarketData, Position  # noqa: E402
from strategies.ema_adx import EmaAdxStrategy  # noqa: E402

INTERVAL_MS = 900_000


def synth_closes(n: int = 320) -> list[float]:
    """Deterministic price path with clear trends and reversals."""
    closes = []
    price = 100.0
    for i in range(n):
        drift = 0.55 * math.sin(i / 18.0) - 0.25 * math.sin(i / 5.0)
        price = max(5.0, price + drift)
        closes.append(price)
    return closes


def candles_from(closes: list[float]) -> list[dict]:
    base_ms = int(time.time() * 1000) - INTERVAL_MS * (len(closes) + 2)
    rows = []
    for i, close in enumerate(closes):
        rows.append(
            {
                "t": base_ms + i * INTERVAL_MS,
                "o": close - 0.1,
                "h": close + 0.3,
                "l": close - 0.3,
                "c": close,
                "v": 100.0,
            }
        )
    return rows


class InstantFillExchange:
    """Fake adapter that fills every order immediately at the last price."""

    def __init__(self, balance: float = 1000.0):
        self.balance = balance
        self.position: Position | None = None
        self.entries: list[Order] = []
        self.closes: list[Order] = []
        self.price = 100.0

    async def get_positions(self, dex=None):  # noqa: ANN001
        if self.position is None or self.position.size == 0:
            return []
        # Refresh mark-to-market so the strategy sees a live entry price.
        self.position.current_value = abs(self.position.size) * self.price
        return [self.position]

    async def get_balance(self, asset: str = "USD"):  # noqa: ANN001
        return SimpleNamespace(available=self.balance, total=self.balance)

    async def place_order(self, order: Order) -> str:
        if order.reduce_only:
            self.closes.append(order)
            if self.position is not None:
                self.position.size += order.size if order.side.value == "buy" else -order.size
                if abs(self.position.size) < 1e-9:
                    self.position = None
        else:
            self.entries.append(order)
            signed = order.size if order.side.value == "buy" else -order.size
            if self.position is None:
                self.position = Position(
                    asset=order.asset,
                    size=signed,
                    entry_price=self.price,
                    current_value=order.size * self.price,
                    unrealized_pnl=0.0,
                    timestamp=time.time(),
                )
            else:
                self.position.size += signed
        return f"0xoid{len(self.entries) + len(self.closes)}"

    async def cancel_all_orders(self) -> int:
        return 0


def make_engine(exchange: InstantFillExchange, strategy: EmaAdxStrategy) -> TradingEngine:
    engine = object.__new__(TradingEngine)
    engine.config = {"risk_management": {}}
    engine.running = True
    engine.exchange = exchange
    engine.strategy = strategy
    engine.market_data = None
    engine.risk_manager = None
    engine.current_positions = []
    engine.pending_orders = {}
    engine._signals_by_oid = {}
    engine._tpsl_oids = {}
    engine._last_fill_ts = {}
    engine.executed_trades = 0
    engine.logger = logging.getLogger("test_sim")
    return engine


def test_simulation_produces_entries_and_exits():
    closes = synth_closes()
    strategy = EmaAdxStrategy(
        {"symbol": "SOL", "interval": "15m", "order_cooldown_s": 0, "pending_reverse_ttl_s": 300},
        history_provider=lambda *a, **k: candles_from(closes[:200]),
    )
    strategy.start()
    assert strategy.get_status()["warmup_ok"] is True

    exchange = InstantFillExchange()
    engine = make_engine(exchange, strategy)

    async def run() -> None:
        seeded = candles_from(closes[:200])
        # One tick per interval: each tick closes the previous candle.
        ts = seeded[-1]["t"] / 1000.0 + 900.0
        for close in closes[200:]:
            exchange.price = close
            await engine._handle_price_update(
                MarketData(
                    asset="SOL",
                    price=close,
                    volume_24h=1_000_000,
                    timestamp=ts,
                )
            )
            ts += 900.0

    asyncio.run(run())

    assert exchange.entries, "strategy never opened a position"
    assert exchange.closes, "strategy never closed a position"
    for order in exchange.entries:
        assert order.size > 0
        assert order.price is None  # market orders
    for order in exchange.closes:
        assert order.reduce_only is True
    assert strategy.get_status()["stats"]["signals"] >= 2
