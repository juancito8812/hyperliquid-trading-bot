"""Tests for ladder entries, partial exits and the simulated broker accounting."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interfaces.strategy import Position, SignalType  # noqa: E402
from strategies.ema_adx import EmaAdxConfig, EmaAdxStrategy  # noqa: E402

INTERVAL_MS = 900_000


# ── helpers (mirror tests/test_ema_adx.py) ──────────────────────────────
def make_strategy(**overrides) -> EmaAdxStrategy:
    cfg = {"symbol": "SOL", "interval": "15m", "testnet": True}
    cfg.update(overrides)
    strategy = EmaAdxStrategy(cfg, history_provider=lambda *a, **k: [])
    strategy._seeded = True
    return strategy


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


def position(size: float, entry: float) -> Position:
    return Position(
        asset="SOL",
        size=size,
        entry_price=entry,
        current_value=abs(size) * entry,
        unrealized_pnl=0.0,
        timestamp=time.time(),
    )


# ── config validation ───────────────────────────────────────────────────
def test_ladder_weights_must_not_exceed_one():
    with pytest.raises(ValueError):
        EmaAdxConfig(entry_ladder=[0.6, 0.6]).validate()


def test_ladder_weights_must_be_positive():
    with pytest.raises(ValueError):
        EmaAdxConfig(entry_ladder=[0.0, 1.0]).validate()


def test_partial_exits_must_be_ordered_and_capped():
    with pytest.raises(ValueError):
        EmaAdxConfig(partial_exits=[[2.0, 0.5], [1.0, 0.5]]).validate()
    with pytest.raises(ValueError):
        EmaAdxConfig(partial_exits=[[1.0, 0.7], [2.0, 0.7]]).validate()
    with pytest.raises(ValueError):
        EmaAdxConfig(partial_exits=[[1.0, 1.5]]).validate()


def test_default_config_keeps_single_entry_and_single_exit():
    cfg = EmaAdxConfig()
    assert cfg.entry_ladder == [1.0]
    assert cfg.partial_exits == []
    cfg.validate()


# ── ladder entry ────────────────────────────────────────────────────────
def test_first_leg_is_weighted_fraction_of_full_risk_size():
    strategy = make_strategy(entry_ladder=[0.5, 0.25, 0.25])
    strategy._indicators = snapshot(crossover_bull=True)
    signals = strategy._entry_signals(100.0, 1000.0, time.time(), True)
    assert len(signals) == 1
    # 1% of 1000 = 10 USD risk over a 2 USD stop -> 5 units full size
    assert strategy._plan["full_size"] == pytest.approx(5.0)
    assert signals[0].size == pytest.approx(2.5)


def test_ladder_adds_only_after_the_configured_adverse_step():
    strategy = make_strategy(entry_ladder=[0.5, 0.25, 0.25], ladder_step_atr=0.5)
    strategy._indicators = snapshot(crossover_bull=True)
    now = time.time()
    strategy._entry_signals(100.0, 1000.0, now, True)
    strat_pos = position(2.5, 100.0)

    strategy._indicators = snapshot()
    assert strategy._exit_signals(strat_pos, price=99.6, now=now + 31) == []

    add = strategy._exit_signals(strat_pos, price=99.4, now=now + 62)
    assert len(add) == 1
    assert add[0].signal_type is SignalType.BUY
    assert add[0].reason == "ladder_add_1"
    assert add[0].size == pytest.approx(1.25)


def test_second_ladder_step_waits_for_a_full_extra_step():
    strategy = make_strategy(entry_ladder=[0.5, 0.25, 0.25], ladder_step_atr=0.5)
    strategy._indicators = snapshot(crossover_bull=True)
    now = time.time()
    strategy._entry_signals(100.0, 1000.0, now, True)
    strat_pos = position(2.5, 100.0)
    strategy._indicators = snapshot()
    assert strategy._exit_signals(strat_pos, price=99.4, now=now + 31)[0].reason == "ladder_add_1"
    # step 2 sits 1.0 ATR below the anchor
    assert strategy._exit_signals(strat_pos, price=99.2, now=now + 62) == []
    second = strategy._exit_signals(strat_pos, price=99.0, now=now + 93)
    assert second[0].reason == "ladder_add_2"
    assert second[0].size == pytest.approx(1.25)


def test_ladder_exhausts_after_all_legs():
    strategy = make_strategy(entry_ladder=[0.5, 0.5], ladder_step_atr=0.5)
    strategy._indicators = snapshot(crossover_bull=True)
    now = time.time()
    strategy._entry_signals(100.0, 1000.0, now, True)
    strat_pos = position(2.5, 100.0)
    strategy._indicators = snapshot()
    assert len(strategy._exit_signals(strat_pos, price=99.4, now=now + 31)) == 1
    deeper = strategy._exit_signals(strat_pos, price=98.0, now=now + 62)
    assert [s for s in deeper if s.signal_type is SignalType.BUY] == []


def test_stop_loss_wins_over_ladder_add():
    strategy = make_strategy(entry_ladder=[0.5, 0.5], ladder_step_atr=0.5)
    strategy._indicators = snapshot(crossover_bull=True)
    now = time.time()
    strategy._entry_signals(100.0, 1000.0, now, True)
    strat_pos = position(2.5, 100.0)
    strategy._indicators = snapshot()
    signals = strategy._exit_signals(strat_pos, price=97.5, now=now + 31)
    assert signals[0].reason == "stop_loss"


# ── partial exits ───────────────────────────────────────────────────────
def test_partial_exit_sells_fraction_at_first_r_level():
    strategy = make_strategy(partial_exits=[[1.0, 0.5], [2.0, 0.3]])
    strategy._indicators = snapshot()
    now = time.time()
    strat_pos = position(5.0, 100.0)  # 1R = 2 * ATR = 2 USD
    assert strategy._exit_signals(strat_pos, price=101.9, now=now) == []
    first = strategy._exit_signals(strat_pos, price=102.2, now=now + 31)
    assert len(first) == 1
    assert first[0].signal_type is SignalType.CLOSE
    assert first[0].reason == "partial_1r"
    assert first[0].size == pytest.approx(2.5)


def test_partial_exits_are_not_repeated_and_scale_to_second_level():
    strategy = make_strategy(partial_exits=[[1.0, 0.5], [2.0, 0.3]])
    strategy._indicators = snapshot()
    now = time.time()
    strat_pos = position(5.0, 100.0)
    assert strategy._exit_signals(strat_pos, price=102.2, now=now)[0].reason == "partial_1r"
    second = strategy._exit_signals(strat_pos, price=104.2, now=now + 31)
    assert second[0].reason == "partial_2r"
    assert second[0].size == pytest.approx(1.5)
    # nothing left to scale out; price back inside the band -> no signal
    assert strategy._exit_signals(strat_pos, price=103.0, now=now + 62) == []


def test_partial_exit_works_for_shorts():
    strategy = make_strategy(partial_exits=[[1.0, 0.5]])
    strategy._indicators = snapshot()
    now = time.time()
    strat_pos = position(-5.0, 100.0)
    signals = strategy._exit_signals(strat_pos, price=97.8, now=now)
    assert signals[0].reason == "partial_1r"
    assert signals[0].size == pytest.approx(2.5)


def test_plan_resets_when_flat():
    from interfaces.strategy import MarketData

    strategy = make_strategy(entry_ladder=[0.5, 0.5])
    strategy._indicators = snapshot(crossover_bull=True)
    strategy._entry_signals(100.0, 1000.0, time.time(), True)
    assert strategy._plan is not None
    strategy._indicators = snapshot()
    strategy.generate_signals(
        MarketData(asset="SOL", price=100.0, volume_24h=1e6, timestamp=time.time()),
        [],
        1000.0,
    )
    assert strategy._plan is None


# ── simulated broker ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bt():
    path = ROOT / "scripts" / "backtest_ema_adx.py"
    spec = importlib.util.spec_from_file_location("bt_ema_adx", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses needs the module importable
    spec.loader.exec_module(module)
    return module


def test_broker_average_entry_on_adds(bt):
    broker = bt.Broker(coin="SOL", balance=1000.0)
    broker.apply("buy", 2.0, 100.0, 1, reduce_only=False)
    broker.apply("buy", 2.0, 98.0, 2, reduce_only=False)
    assert broker.position.size == pytest.approx(4.0)
    slip_buy = 1 + bt.SLIPPAGE
    expected = (2.0 * 100.0 * slip_buy + 2.0 * 98.0 * slip_buy) / 4.0
    assert broker.position.entry_price == pytest.approx(expected)
    assert broker.open_trade.size == pytest.approx(4.0)
    assert broker.open_trade.entry_price == pytest.approx(expected)


def test_broker_partial_close_keeps_the_rest_open(bt):
    broker = bt.Broker(coin="SOL", balance=1000.0)
    broker.apply("buy", 4.0, 100.0, 1, reduce_only=False)
    broker.apply("sell", 2.0, 102.0, 2, reduce_only=True)
    assert len(broker.trades) == 1
    assert broker.trades[0].size == pytest.approx(2.0)
    assert broker.position is not None and broker.position.size == pytest.approx(2.0)
    assert broker.open_trade is not None and broker.open_trade.size == pytest.approx(2.0)

    broker.apply("sell", 2.0, 96.0, 3, reduce_only=True)
    assert len(broker.trades) == 2
    assert broker.position is None and broker.open_trade is None
    # every fee charged is attributed to some trade
    assert sum(t.fees for t in broker.trades) == pytest.approx(broker.fees_paid)


def test_broker_partial_pnl_uses_average_entry(bt):
    broker = bt.Broker(coin="SOL", balance=1000.0)
    broker.apply("buy", 2.0, 100.0, 1, reduce_only=False)
    broker.apply("buy", 2.0, 100.0, 2, reduce_only=False)  # same price -> avg stays 100
    broker.apply("sell", 4.0, 110.0, 3, reduce_only=True)
    assert broker.trades[0].pnl > 0
    assert broker.trades[0].entry_price == pytest.approx(100.0 * (1 + bt.SLIPPAGE))


def test_broker_stats_expose_risk_metrics(bt):
    broker = bt.Broker(coin="SOL", balance=1000.0)
    broker.apply("buy", 1.0, 100.0, 1, reduce_only=False)
    broker.mark(101.0, 1)
    broker.mark(99.0, 2)
    stats = broker.stats()
    for key in ("expectancy_usd", "exposure_pct", "avg_leverage", "max_drawdown_pct"):
        assert key in stats
    assert stats["exposure_pct"] == pytest.approx(100.0)
    assert stats["avg_leverage"] > 0
