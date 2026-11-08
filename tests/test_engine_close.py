"""Engine-level test: a CLOSE signal must actually exit with a reduce-only order."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.engine import TradingEngine  # noqa: E402
from interfaces.exchange import Order, OrderSide, OrderType  # noqa: E402
from interfaces.strategy import (  # noqa: E402
    Position,
    SignalType,
    TradingSignal,
)


class FakeExchange:
    """Minimal adapter: records orders, reports a single open position."""

    def __init__(self, size: float = 1.5, entry: float = 100.0):
        self.position = Position(
            asset="SOL",
            size=size,
            entry_price=entry,
            current_value=abs(size) * entry,
            unrealized_pnl=0.0,
            timestamp=time.time(),
        )
        self.orders: list[Order] = []
        self.cancelled = 0

    async def get_positions(self, dex=None):  # noqa: ANN001
        return [self.position] if self.position.size else []

    async def place_order(self, order: Order) -> str:
        self.orders.append(order)
        return f"0xabc{len(self.orders)}"

    async def cancel_all_orders(self) -> int:
        self.cancelled += 1
        return 3


def make_engine(exchange: FakeExchange) -> TradingEngine:
    engine = object.__new__(TradingEngine)
    engine.config = {}
    engine.exchange = exchange
    engine.strategy = None
    engine.pending_orders = {}
    engine._signals_by_oid = {}
    engine.logger = logging.getLogger("test_engine")
    return engine


def close_signal(size: float = 1.5, dex=None) -> TradingSignal:  # noqa: ANN001
    return TradingSignal(
        signal_type=SignalType.CLOSE,
        asset="SOL",
        size=size,
        price=None,
        reason="stop_loss",
        metadata={"strategy": "ema_adx"},
        dex=dex,
    )


def test_close_signal_places_reduce_only_market_order():
    exchange = FakeExchange(size=1.5)
    engine = make_engine(exchange)

    asyncio.run(engine._close_positions(close_signal()))

    assert len(exchange.orders) == 1
    order = exchange.orders[0]
    assert order.side is OrderSide.SELL  # long -> sell to close
    assert order.order_type is OrderType.MARKET
    assert order.reduce_only is True
    assert order.size == 1.5
    assert engine.pending_orders  # tracked
    assert len(engine._signals_by_oid) == 1  # correlatable to a fill


def test_close_signal_reverses_side_for_short():
    exchange = FakeExchange(size=-2.0)
    engine = make_engine(exchange)

    asyncio.run(engine._close_positions(close_signal(size=2.0)))

    order = exchange.orders[0]
    assert order.side is OrderSide.BUY
    assert order.size == 2.0
    assert order.reduce_only is True


def test_close_signal_without_position_is_noop():
    exchange = FakeExchange(size=0.0)
    engine = make_engine(exchange)

    asyncio.run(engine._close_positions(close_signal()))

    assert exchange.orders == []


def test_partial_close_never_exceeds_position():
    exchange = FakeExchange(size=1.0)
    engine = make_engine(exchange)

    asyncio.run(engine._close_positions(close_signal(size=5.0)))

    assert exchange.orders[0].size == 1.0


def test_cancel_all_action_still_only_cancels():
    exchange = FakeExchange()
    engine = make_engine(exchange)
    signal = close_signal()
    signal.metadata = {"action": "cancel_all"}

    asyncio.run(engine._close_positions(signal))

    assert exchange.cancelled == 1
    assert exchange.orders == []
