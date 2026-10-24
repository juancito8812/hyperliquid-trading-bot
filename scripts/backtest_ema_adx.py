"""Backtest the EMA/ADX strategy on one year of real market data.

Design notes
------------
* The **live strategy code** is replayed unchanged (``strategies.ema_adx``), so
  what is measured is what the bot would have done.
* Execution is simulated by :class:`Broker`: market fills with 0.045% taker fee
  and 3 bps slippage; stops/targets are checked on every tick.
* Ticks are 1-minute candles and the strategy's own resampler builds the
  ``interval`` bars from them, so intrabar stop/target hits behave like live.
* Data defaults to Binance USDⓈ-M futures klines (full history, tracks
  Hyperliquid perp prices). ``--source hyperliquid`` uses the venue's own API,
  but it only serves the newest ~5000 candles per interval (52 days at 15m),
  so a real 1-year venue test needs ``--interval 2h`` or ``4h``.

Usage
-----
    uv run python scripts/backtest_ema_adx.py --coins SOL,ETH,BTC --interval 15m --days 365
    uv run python scripts/backtest_ema_adx.py --source hyperliquid --interval 4h --days 365
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interfaces.strategy import MarketData, Position  # noqa: E402
from strategies.ema_adx import EmaAdxStrategy  # noqa: E402

HL_API = "https://api.hyperliquid.xyz/info"
BN_FUT = "https://data.binance.vision/data/futures/um"
CACHE_DIR = ROOT / "backtest_results" / "cache"

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}
FEE = 0.00045         # Hyperliquid perp taker fee (overridable with --fee)
SLIPPAGE = 0.0003     # 3 bps per fill
START_BALANCE = 1000.0

Bar = Dict[str, float]

# ── data ────────────────────────────────────────────────────────────────
def _bar(ts: int, o: float, h: float, l: float, c: float, v: float = 0.0) -> Bar:
    return {"t": int(ts), "o": o, "h": h, "l": l, "c": c, "v": v}


def _download(url: str, tries: int = 4) -> Optional[bytes]:
    with httpx.Client(timeout=120.0, follow_redirects=True, trust_env=False) as client:
        for attempt in range(tries):
            try:
                resp = client.get(url)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPError:
                if attempt == tries - 1:
                    return None
                time.sleep(1.5 * (attempt + 1))
    return None


def _parse_kline_zip(blob: bytes) -> List[Bar]:
    """Binance vision klines CSV (monthly files ship without a header)."""
    out: List[Bar] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            text = zf.read(name).decode("utf-8", "ignore")
            for row in csv.reader(io.StringIO(text)):
                if not row or not row[0].strip():
                    continue
                if not row[0].isdigit():
                    continue  # header line
                ts = int(row[0])
                if ts > 10 ** 14:      # microseconds (Binance since 2025-01-01)
                    ts //= 1000
                out.append(_bar(ts, float(row[1]), float(row[2]),
                                float(row[3]), float(row[4]), float(row[5])))
    return out


def fetch_binance_1m(coin: str, days: int) -> List[Bar]:
    """Full 1-minute history from Binance USDⓈ-M futures (monthly + daily zips)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"bn_{coin}_1m_{days}d.json.gz"
    import gzip
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86_400:
        with gzip.open(cache, "rt") as fh:
            return json.load(fh)

    symbol = f"{coin}USDT"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    out: Dict[int, Bar] = {}

    month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while month <= end:
        key = month.strftime("%Y-%m")
        url = f"{BN_FUT}/monthly/klines/{symbol}/1m/{symbol}-1m-{key}.zip"
        blob = _download(url)
        if blob:
            for b in _parse_kline_zip(blob):
                out[b["t"]] = b
        else:  # current (incomplete) month → daily files
            day = max(month, start.replace(hour=0, minute=0, second=0, microsecond=0))
            while day <= end:
                dk = day.strftime("%Y-%m-%d")
                durl = f"{BN_FUT}/daily/klines/{symbol}/1m/{symbol}-1m-{dk}.zip"
                dblob = _download(durl)
                if dblob:
                    for b in _parse_kline_zip(dblob):
                        out[b["t"]] = b
                day += timedelta(days=1)
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        time.sleep(0.15)

    rows = [out[k] for k in sorted(out)]
    with gzip.open(cache, "wt") as fh:
        json.dump(rows, fh)
    return rows


def fetch_hyperliquid(coin: str, interval: str, days: int) -> List[Bar]:
    """Venue candles, walking backwards. Depth is capped (~5000 candles)."""
    step = INTERVAL_MS[interval]
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    out: Dict[int, Bar] = {}
    cursor_end = end
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        while cursor_end > start:
            chunk_start = max(start, cursor_end - step * 5000)
            payload = {"type": "candleSnapshot",
                       "req": {"coin": coin, "interval": interval,
                               "startTime": chunk_start, "endTime": cursor_end}}
            rows = None
            for attempt in range(6):
                try:
                    resp = client.post(HL_API, json=payload)
                    if resp.status_code == 429:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    rows = resp.json() or []
                    break
                except httpx.HTTPError:
                    if attempt == 5:
                        raise
                    time.sleep(1.0 * (attempt + 1))
            if rows is None:
                raise SystemExit(f"{coin} {interval}: rate limited, retry later")
            time.sleep(0.2)
            if not rows:
                break
            for r in rows:
                out[int(r["t"])] = _bar(int(r["t"]), float(r["o"]), float(r["h"]),
                                        float(r["l"]), float(r["c"]), float(r.get("v", 0)))
            oldest = min(int(r["t"]) for r in rows)
            if oldest > chunk_start + step * 2:
                break  # API clamped to the newest rows: no older history
            nxt = oldest - step
            if nxt >= cursor_end:
                break
            cursor_end = nxt
    ordered = [out[k] for k in sorted(out)]
    return ordered[:-1]


def aggregate(bars: List[Bar], interval: str) -> List[Bar]:
    """Bucket finer bars into ``interval`` OHLCV bars."""
    step = INTERVAL_MS[interval]
    buckets: Dict[int, Bar] = {}
    for b in bars:
        key = b["t"] - (b["t"] % step)
        cur = buckets.get(key)
        if cur is None:
            buckets[key] = _bar(key, b["o"], b["h"], b["l"], b["c"], b["v"])
        else:
            cur["h"] = max(cur["h"], b["h"])
            cur["l"] = min(cur["l"], b["l"])
            cur["c"] = b["c"]
            cur["v"] += b["v"]
    return [buckets[k] for k in sorted(buckets)]


# ── simulated broker ────────────────────────────────────────────────────
@dataclass
class Trade:
    coin: str
    side: str
    size: float
    entry_ts: int
    entry_price: float
    exit_ts: int = 0
    exit_price: float = 0.0
    reason: str = ""
    pnl: float = 0.0
    fees: float = 0.0
    equity_after: float = 0.0


@dataclass
class Broker:
    coin: str
    balance: float
    position: Optional[Position] = None
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)
    open_trade: Optional[Trade] = None
    fees_paid: float = 0.0

    def positions(self) -> List[Position]:
        return [self.position] if self.position else []

    def apply(self, side: str, size: float, price: float, ts: int, reduce_only: bool) -> None:
        if size <= 0:
            return
        slip = price * (1 + SLIPPAGE) if side == "buy" else price * (1 - SLIPPAGE)
        fee = size * slip * FEE
        self.fees_paid += fee

        if reduce_only:
            if not self.position or not self.open_trade:
                return
            direction = 1 if self.position.size > 0 else -1
            closed = min(size, abs(self.position.size))
            pnl = (slip - self.open_trade.entry_price) * closed * direction
            self.balance += pnl - fee
            trade = self.open_trade
            trade.exit_ts, trade.exit_price = ts, slip
            trade.pnl = pnl - fee
            trade.fees += fee
            trade.equity_after = self.balance
            self.trades.append(trade)
            self.open_trade = None
            remaining = abs(self.position.size) - closed
            self.position = None if remaining <= 1e-9 else Position(
                asset=self.coin, size=remaining * direction,
                entry_price=self.position.entry_price,
                current_value=remaining * slip, unrealized_pnl=0.0,
                timestamp=ts / 1000.0)
            return

        signed = size if side == "buy" else -size
        self.balance -= fee
        if self.position is None:
            self.position = Position(asset=self.coin, size=signed, entry_price=slip,
                                     current_value=abs(signed) * slip, unrealized_pnl=0.0,
                                     timestamp=ts / 1000.0)
            self.open_trade = Trade(coin=self.coin, side="long" if signed > 0 else "short",
                                    size=abs(signed), entry_ts=ts, entry_price=slip, fees=fee)
        else:
            self.position.size += signed

    def mark(self, price: float, ts: int) -> None:
        equity = self.balance
        if self.position:
            direction = 1 if self.position.size > 0 else -1
            equity += (price - self.position.entry_price) * abs(self.position.size) * direction
        self.equity_curve.append((ts, equity))

    def close_open_at(self, price: float, ts: int, reason: str) -> None:
        if not self.position or not self.open_trade:
            return
        side = "sell" if self.position.size > 0 else "buy"
        self.open_trade.reason = reason
        self.apply(side, abs(self.position.size), price, ts, reduce_only=True)

    def stats(self) -> Dict[str, Any]:
        closed = [t for t in self.trades if t.exit_ts]
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        peak, max_dd = -1e18, 0.0
        for _, eq in self.equity_curve:
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak * 100)
        net = self.balance - START_BALANCE
        return {
            "trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 2) if closed else 0.0,
            "net_pnl_usd": round(net, 2),
            "net_pnl_pct": round(net / START_BALANCE * 100, 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "avg_trade_usd": round(statistics.mean([t.pnl for t in closed]), 2) if closed else 0.0,
            "best_trade_usd": round(max((t.pnl for t in closed), default=0.0), 2),
            "worst_trade_usd": round(min((t.pnl for t in closed), default=0.0), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "fees_usd": round(self.fees_paid, 2),
            "exit_reasons": {r: sum(1 for t in closed if t.reason == r)
                             for r in sorted({t.reason for t in closed})},
        }


# ── replay ──────────────────────────────────────────────────────────────
def load_strategy_cfg(coin: str, interval: str) -> Dict[str, Any]:
    path = ROOT / "bots" / f"{coin.lower()}_ema_adx.yaml"
    cfg: Dict[str, Any] = {}
    if path.exists():
        cfg = (yaml.safe_load(path.read_text()) or {}).get("strategy") or {}
        cfg.pop("type", None)
    cfg.setdefault("symbol", coin)
    cfg["interval"] = interval
    return cfg


def run_backtest(coin: str, interval: str, days: int, warmup: int = 200,
                 source: str = "binance",                 overrides: Optional[Dict[str, Any]] = None,
                 trades_out: Optional[Path] = None,
                 label_note: str = "") -> Dict[str, Any]:
    if source == "binance":
        ticks_all = fetch_binance_1m(coin, days)
    else:
        if interval != "1m" and 5000 * INTERVAL_MS[interval] < days * 86_400_000:
            print(f"⚠️  {coin}: Hyperliquid serves only ~5000 {interval} candles "
                  f"({5000 * INTERVAL_MS[interval] / 86_400_000:.0f}d); use 4h for a full year")
        ticks_all = fetch_hyperliquid(coin, interval, days)
    if not ticks_all:
        raise SystemExit(f"{coin}: no data")

    bars = aggregate(ticks_all, interval) if INTERVAL_MS[interval] > INTERVAL_MS["1m"] else ticks_all
    if len(bars) < warmup + 50:
        raise SystemExit(f"{coin}: only {len(bars)} {interval} bars — need {warmup + 50}+")

    cfg = load_strategy_cfg(coin, interval)
    cfg.update(overrides or {})
    cfg["warmup_candles"] = warmup
    seed = bars[:warmup]
    replay_from = bars[warmup]["t"]

    strategy = EmaAdxStrategy(cfg, history_provider=lambda *a, **k: seed)
    strategy.start()
    if not strategy.get_status()["warmup_ok"]:
        raise SystemExit(f"{coin}: warm-up failed")

    # 1m ticks give intrabar stop/target fidelity; the venue path feeds the bars.
    if source == "binance" and INTERVAL_MS[interval] >= 60_000:
        ticks = [t for t in ticks_all if t["t"] >= replay_from]
    else:
        ticks = [{"t": b["t"] + 1, "o": b["o"], "h": b["h"], "l": b["l"],
                  "c": b["c"], "v": b["v"]} for b in bars[warmup:]]

    broker = Broker(coin=coin, balance=START_BALANCE)
    for tick in ticks:
        price = tick["c"]
        market = MarketData(asset=coin, price=price, volume_24h=tick.get("v", 0.0),
                            timestamp=(tick["t"] + 1) / 1000.0)
        signals = strategy.generate_signals(market, broker.positions(), broker.balance)
        broker.mark(price, tick["t"])
        for sig in signals:
            size = float(sig.size or 0)
            if size <= 0:
                continue
            is_close = sig.signal_type.value == "close"
            side = "buy" if sig.signal_type.value == "buy" else "sell"
            before = len(broker.trades)
            broker.apply(side, size, price, tick["t"], reduce_only=is_close)
            if is_close and len(broker.trades) > before:
                broker.trades[-1].reason = sig.reason or "strategy_exit"
            strategy.on_trade_executed(sig, price, size)

    if broker.position:
        last = ticks[-1]
        broker.close_open_at(last["c"], last["t"], "end_of_test")
        broker.mark(last["c"], last["t"])

    if trades_out:
        with trades_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["coin", "side", "size", "entry_time", "entry_price",
                        "exit_time", "exit_price", "reason", "pnl", "equity_after"])
            for t in broker.trades:
                w.writerow([t.coin, t.side, round(t.size, 6),
                            datetime.fromtimestamp(t.entry_ts / 1000, timezone.utc).isoformat(),
                            round(t.entry_price, 6),
                            datetime.fromtimestamp(t.exit_ts / 1000, timezone.utc).isoformat() if t.exit_ts else "",
                            round(t.exit_price, 6), t.reason, round(t.pnl, 4),
                            round(t.equity_after, 2)])

    out = broker.stats()
    out.update({
        "coin": coin, "interval": interval, "days": days, "source": source,
        "bars": len(bars), "ticks": len(ticks),
        "from": bars[0]["t"], "to": bars[-1]["t"],
        "buy_and_hold_pct": round((bars[-1]["c"] / bars[warmup]["o"] - 1) * 100, 2),
        "strategy_signals": strategy.get_status()["stats"],
        "tag": label_note or "",
        "overrides": overrides or {},
        "config": {k: cfg.get(k) for k in
                   ("ema_fast", "ema_slow", "adx_period", "adx_threshold",
                    "atr_period", "atr_sl_multiplier", "risk_reward_ratio",
                    "risk_per_trade", "max_leverage")},
    })
    return out


def main() -> None:
    global FEE
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default="SOL,ETH,BTC")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--source", default="binance", choices=["binance", "hyperliquid"])
    ap.add_argument("--out", default="backtest_results")
    ap.add_argument("--fee", type=float, default=0.00045, help="taker fee rate per fill")
    ap.add_argument("--set", default="", help="strategy overrides, e.g. adx_threshold=25,sar_atr_multiplier=2")
    ap.add_argument("--tag", default="", help="label for this run; written into file names and the JSON")
    args = ap.parse_args()

    FEE = args.fee
    overrides: Dict[str, Any] = {}
    for pair in filter(None, (p.strip() for p in args.set.split(","))):
        k, _, v = pair.partition("=")
        val = v.strip()
        if val.lower() in ("true", "false"):
            overrides[k.strip()] = val.lower() == "true"
        elif val.replace(".", "", 1).replace("-", "", 1).isdigit():
            overrides[k.strip()] = float(val)
        else:
            overrides[k.strip()] = val

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    outdir_early = ROOT / args.out
    outdir_early.mkdir(exist_ok=True)
    results = []
    for coin in coins:
        t0 = time.time()
        print(f"\n⏳ {coin} {args.interval} — {args.days}d ({args.source}) …", flush=True)
        res = run_backtest(coin, args.interval, args.days, source=args.source,
                           overrides=overrides, label_note=args.tag,
                           trades_out=outdir_early / (
                               f"{coin}_{args.interval}"
                               f"{('_' + args.tag) if args.tag else ''}_trades.csv"))
        results.append(res)
        print(f"✅ {coin}: {res['trades']} trades | win {res['win_rate_pct']}% | "
              f"net {res['net_pnl_pct']}% | maxDD {res['max_drawdown_pct']}% | "
              f"PF {res['profit_factor']} | B&H {res['buy_and_hold_pct']}% "
              f"| {time.time() - t0:.0f}s", flush=True)

    outdir = ROOT / args.out
    outdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = args.tag or ("overrides" if overrides else "base")
    tag = f"ema_adx_{args.interval}_{args.days}d_{args.source}_{label}_{stamp}"
    (outdir / f"{tag}.json").write_text(json.dumps(results, indent=2, default=str))

    cols = ["coin", "interval", "trades", "wins", "losses", "win_rate_pct", "net_pnl_pct",
            "max_drawdown_pct", "profit_factor", "avg_trade_usd", "best_trade_usd",
            "worst_trade_usd", "fees_usd", "buy_and_hold_pct"]
    with (outdir / f"{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\n📄 {outdir / (tag + '.json')}\n📄 {outdir / (tag + '.csv')}")


if __name__ == "__main__":
    main()
