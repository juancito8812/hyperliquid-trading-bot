"""Run the EMA+ADX variant matrix for one coin and append results to JSON.

Usage: uv run python scripts/run_matrix.py SOL [days]
Resumable: variants already present in the output file are skipped.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bt_ema_adx", ROOT / "scripts" / "backtest_ema_adx.py")
assert spec is not None and spec.loader is not None
bt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bt
spec.loader.exec_module(bt)

LADDER = {"entry_ladder": [0.5, 0.25, 0.25], "ladder_step_atr": 0.5}
PARTIAL = {"partial_exits": [[1.0, 0.5], [2.0, 0.3]]}
WIDE = {"entry_ladder": [0.4, 0.3, 0.3], "ladder_step_atr": 1.0,
        "partial_exits": [[1.0, 0.5], [2.0, 0.5]]}

VARIANTS = [
    {"name": "base", "overrides": {}},
    {"name": "ladder", "overrides": LADDER},
    {"name": "partial", "overrides": PARTIAL},
    {"name": "both", "overrides": {**LADDER, **PARTIAL}},
    {"name": "wide", "overrides": WIDE},
    {"name": "both_nofee", "overrides": {**LADDER, **PARTIAL}, "fee": 0.0},
    # maker fills (limit orders): 0.015% instead of 0.045% taker
    {"name": "both_maker", "overrides": {**LADDER, **PARTIAL}, "fee": 0.00015},
    {"name": "wide_maker", "overrides": WIDE, "fee": 0.00015},
]


def main() -> None:
    coin = sys.argv[1].upper()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    interval = sys.argv[3] if len(sys.argv) > 3 else "15m"
    out = ROOT / "backtest_results" / f"matrix_{coin}_{interval}_{days}d.json"
    out.parent.mkdir(exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {r["variant"] for r in rows}

    for v in VARIANTS:
        if v["name"] in done:
            print(f"skip {coin} {v['name']}", flush=True)
            continue
        bt.FEE = v.get("fee", 0.00045)
        t0 = time.time()
        res = bt.run_backtest(coin, interval, days, overrides=v.get("overrides") or {},
                              label_note=f"{v['name']}")
        res["variant"] = v["name"]
        res["runtime_s"] = round(time.time() - t0, 1)
        rows.append(res)
        out.write_text(json.dumps(rows, indent=2))
        print(f"{coin} {v['name']}: {res['net_pnl_pct']:+.2f}% "
              f"pf={res['profit_factor']} trades={res['trades']} "
              f"dd={res['max_drawdown_pct']}% fees={res['fees_usd']} "
              f"({res['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
