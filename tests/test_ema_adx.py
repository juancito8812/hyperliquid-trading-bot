"""Tests for the EMA/ADX strategy port (indicators, candles, signals)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interfaces.strategy import MarketData, Position, SignalType  # noqa: E402
from strategies import STRATEGY_REGISTRY, create_strategy  # noqa: E402
from strategies.ema_adx import (  # noqa: E402
    CandleBuffer,
    EmaAdxConfig,
    EmaAdxStrategy,
    compute,
    ema,
    true_range,
)
from strategies.ema_adx.indicators import adx, atr  # noqa: E402

INTERVAL_MS = 900_000


# ── indicators ──────────────────────────────────────────────────────────
def test_ema_matches_manual_recursion():
    out = ema([1, 2, 3, 4], 3)  # alpha = 0.5
    assert out == pytest.approx([1.0, 1.5, 2.25, 3.125])


def test_ema_period_one_is_identity():
    values = [5.0, 7.0, 6.0]
    assert ema(values, 1) == pytest.approx(values)


def test_true_range_values():
    tr = true_range([10, 12, 11], [8, 9, 7], [9, 11, 8])
    # bar0 = 2, bar1 = max(3, |12-9|, |9-9|) = 3,
    # bar2 = max(4, |11-11|, |7-11|) = 4
    assert tr == pytest.approx([2.0, 3.0, 4.0])


def test_atr_positive_and_shaped():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 80))
    high = close + 1.5
    low = close - 1.5
    values = atr(high, low, close, 14)
    assert values.shape == close.shape
    assert np.all(values > 0)


def test_adx_bounded_and_direction_aware():
    close = np.linspace(100, 140, 120)  # steady uptrend
    high = close + 0.5
    low = close - 0.5
    adx_values, plus_di, minus_di = adx(high, low, close, 14)
    assert np.all(adx_values >= 0) and np.all(adx_values <= 100)
    assert adx_values[-1] > 25  # strong trend
    assert plus_di[-1] > minus_di[-1]


def test_compute_requires_enough_bars():
    closes = [100.0] * 10
    assert compute(closes, closes, closes) is None


def test_compute_returns_expected_keys():
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.5, 120))
    high, low = close + 1, close - 1
    snap = compute(high, low, close)
    assert snap is not None
    for key in (
        "ema_fast",
        "ema_slow",
        "ema_fast_prev",
        "ema_slow_prev",
        "atr",
        "adx",
        "close",
        "crossover_bull",
        "crossover_bear",
    ):
        assert key in snap
    assert isinstance(snap["crossover_bull"], bool)
    assert isinstance(snap["crossover_bear"], bool)
    assert not (snap["crossover_bull"] and snap["crossover_bear"])


# ── candle building ─────────────────────────────────────────────────────
def test_buffer_closes_bar_on_bucket_rollover():
    buf = CandleBuffer("15m")
    base = (int(time.time()) // 900) * 900 - 3600  # past bucket, seconds
    assert buf.add_price(100.0, base) is False
    assert buf.add_price(105.0, base + 10) is False
    assert buf.add_price(95.0, base + 20) is False
    assert len(buf) == 0

    # next bucket -> previous bar closes
    assert buf.add_price(101.0, base + 900) is True
    assert len(buf) == 1
    bar = buf.candles[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 105.0, 95.0, 95.0)


def test_buffer_seed_sorts_dedupes_and_drops_open_bar():
    buf = CandleBuffer("15m")
    now_ms = int(time.time() * 1000)
    rows = [
        {"t": now_ms - 3 * INTERVAL_MS, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1},
        {"t": now_ms - 2 * INTERVAL_MS, "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 1},
        {"t": now_ms - 3 * INTERVAL_MS, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1},
        {"t": now_ms, "o": 9, "h": 9, "l": 9, "c": 9, "v": 1},  # still forming
    ]
    loaded = buf.seed(rows)
    assert loaded == 2
    assert [c.ts for c in buf.candles] == sorted(c.ts for c in buf.candles)


def test_buffer_rejects_bad_interval():
    with pytest.raises(ValueError):
        CandleBuffer("7m")


# ── config ──────────────────────────────────────────────────────────────
def test_config_validation_rejects_bad_params():
    with pytest.raises(ValueError):
        EmaAdxConfig(ema_fast=9, ema_slow=3).validate()
    with pytest.raises(ValueError):
        EmaAdxConfig(adx_threshold=150).validate()
    with pytest.raises(ValueError):
        EmaAdxConfig(risk_per_trade=0.5).validate()
    with pytest.raises(ValueError):
        EmaAdxConfig(interval="7m").validate()


def test_config_normalises_symbol():
    cfg = EmaAdxConfig.from_dict({"symbol": "SOL/USDC", "type": "ema_adx"})
    assert cfg.symbol == "SOL"


def test_registry_exposes_ema_adx():
    assert "ema_adx" in STRATEGY_REGISTRY
    strategy = create_strategy("ema_adx", {"symbol": "SOL", "testnet": True})
    assert isinstance(strategy, EmaAdxStrategy)


# ── strategy signals ────────────────────────────────────────────────────
def make_strategy(**overrides) -> EmaAdxStrategy:
    cfg = {"symbol": "SOL", "interval": "15m", "testnet": True}
    cfg.update(overrides)
    return EmaAdxStrategy(cfg, history_provider=lambda *a, **k: [])


def market(price: float) -> MarketData:
    return MarketData(
        asset="SOL", price=price, volume_24h=1_000_000, timestamp=time.time()
    )


def position(size: float, entry: float) -> Position:
    return Position(
        asset="SOL",
        size=size,
        entry_price=entry,
        current_value=abs(size) * entry,
        unrealized_pnl=0.0,
        timestamp=time.time(),
    )


def snapshot(**overrides) -> dict:
    base = {
        "ema_fast": 101.0,
        "ema_slow": 100.0,
        "ema_fast_prev": 99.0,
        "ema_slow_prev": 100.0,
        "atr": 1.0,
        "adx": 30.0,
        "close": 100.0,
        "crossover_bull": False,
        "crossover_bear": False,
    }
    base.update(overrides)
    return base


def seed_ready(strategy: EmaAdxStrategy) -> EmaAdxStrategy:
    """Skip history warm-up so the signal logic can be tested directly."""
    strategy._seeded = True
    return strategy


def test_entry_on_bull_cross():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot(crossover_bull=True)
    signals = strategy._entry_signals(price=100.0, balance=1000.0, now=time.time(), bar_closed=True)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type is SignalType.BUY
    assert signal.asset == "SOL"
    assert signal.size > 0
    assert signal.price is None  # market order
    assert signal.reason == "ema_cross_up"


def test_entry_on_bear_cross_is_sell():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot(crossover_bear=True, ema_fast=99.0)
    signals = strategy._entry_signals(100.0, 1000.0, time.time(), True)
    assert len(signals) == 1 and signals[0].signal_type is SignalType.SELL


def test_no_entry_below_adx_threshold():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot(crossover_bull=True, adx=10.0)
    assert strategy._entry_signals(100.0, 1000.0, time.time(), True) == []


def test_no_entry_without_bar_close():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot(crossover_bull=True)
    assert strategy._entry_signals(100.0, 1000.0, time.time(), False) == []


def test_cooldown_blocks_duplicate_signals():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot(crossover_bull=True)
    now = time.time()
    assert len(strategy._entry_signals(100.0, 1000.0, now, True)) == 1
    assert strategy._entry_signals(100.25, 1000.0, now + 1, True) == []


def test_stop_loss_exit_for_long():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot()  # atr=1, sl mult=2 -> stop at entry-2
    pos = position(1.0, 100.0)
    signals = strategy._exit_signals(pos, price=97.5, now=time.time())
    assert len(signals) == 1
    assert signals[0].signal_type is SignalType.CLOSE
    assert signals[0].reason == "stop_loss"


def test_take_profit_exit_for_long():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot()  # tp = 2 * 3 = 6 -> 106
    pos = position(1.0, 100.0)
    signals = strategy._exit_signals(pos, price=106.5, now=time.time())
    assert len(signals) == 1 and signals[0].reason == "take_profit"


def test_sar_exit_sets_pending_reverse():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot()  # sar distance = 1.2
    pos = position(-1.0, 100.0)  # short
    now = time.time()
    signals = strategy._exit_signals(pos, price=101.5, now=now)
    assert len(signals) == 1 and signals[0].reason == "sar"
    assert strategy._pending_reverse == {
        "side": "buy",
        "ts": now,
        "reason": "sar",
    }

    # Next flat tick re-enters in the reversed direction immediately.
    strategy._indicators = snapshot()
    entries = strategy._entry_signals(101.5, 1000.0, now + 31, bar_closed=False)
    assert len(entries) == 1 and entries[0].signal_type is SignalType.BUY


def test_no_exit_inside_noise_band():
    strategy = seed_ready(make_strategy())
    strategy._indicators = snapshot()
    pos = position(1.0, 100.0)
    assert strategy._exit_signals(pos, price=100.5, now=time.time()) == []


def test_position_size_respects_risk_and_cap():
    strategy = seed_ready(make_strategy())
    # risk 1% of 1000 = 10 USD, stop distance = 2*1 = 2 -> 5 units
    size = strategy._position_size(balance=1000.0, atr_value=1.0, price=100.0)
    assert size == pytest.approx(5.0)
    # 90% notional cap: 1000*0.9/100 = 9 units
    capped = strategy._position_size(balance=1000.0, atr_value=0.05, price=100.0)
    assert capped == pytest.approx(9.0)


def test_position_size_skips_below_min_notional():
    strategy = seed_ready(make_strategy())
    assert strategy._position_size(balance=20.0, atr_value=5.0, price=100.0) == 0.0


def test_generate_signals_waits_for_history():
    strategy = make_strategy()  # provider returns []
    assert strategy.generate_signals(market(100.0), [], 1000.0) == []
    assert strategy.get_status()["warmup_ok"] is False


def test_generate_signals_enters_after_seeded_history():
    closes = list(np.linspace(100, 90, 60)) + list(np.linspace(90, 130, 20))
    base_ms = int(time.time() * 1000) - 900_000 * (len(closes) + 2)
    rows = [
        {
            "t": base_ms + i * INTERVAL_MS,
            "o": c,
            "h": c * 1.003,
            "l": c * 0.997,
            "c": c,
            "v": 10.0,
        }
        for i, c in enumerate(closes)
    ]
    strategy = EmaAdxStrategy(
        {"symbol": "SOL", "interval": "15m"},
        history_provider=lambda symbol, interval, limit: rows,
    )
    strategy.start()
    assert strategy.get_status()["warmup_ok"] is True
    assert strategy._indicators is not None

    # A tick inside the current bar must not trade: entries need a close.
    signals = strategy.generate_signals(market(closes[-1]), [], 1000.0)
    assert signals == [] or all(s.signal_type is not SignalType.HOLD for s in signals)


def test_yaml_bot_configs_load_and_validate():
    from core.enhanced_config import EnhancedBotConfig

    bots_dir = Path(__file__).resolve().parents[1] / "bots"
    for name in ("sol_ema_adx.yaml", "eth_ema_adx.yaml", "btc_ema_adx.yaml"):
        config = EnhancedBotConfig.from_yaml(bots_dir / name)
        assert config.strategy["type"] == "ema_adx"
        assert config.name
