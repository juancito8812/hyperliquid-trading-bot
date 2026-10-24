# Backtesting 1 año — EMA/ADX (SOL, ETH, BTC)

Motor: `scripts/backtest_ema_adx.py` — replay del MISMO código de la estrategia
(`src/strategies/ema_adx`) sobre velas de 1 minuto reales, con broker simulado.

- Datos: Binance USDⓈ-M futuros 1m (misma fuente de precio que el perp de Hyperliquid) para 15m/1h.
  Para 4h se usó el API nativo de Hyperliquid (`--source hyperliquid`, único intervalo con 1 año completo allí).
- Capital inicial: $1000 · Fee taker 0.045% por fill · Slippage 3 bps · Riesgo 1% por trade · RR 3.
- Periodo: 365 días hasta 2026-10-24.

## Resultados (config actual de los YAML)

| Moneda | TF | Trades | Win% | Neto | MaxDD | PF | Fees |
|---|---|---|---|---|---|---|---|
| SOL | 15m | 1467 | 18.5 | -76.5% | 84.9% | 0.95 | $949 |
| ETH | 15m | 1506 | 18.4 | -88.5% | 89.3% | 0.74 | $445 |
| BTC | 15m | 1545 | 17.4 | -84.8% | 85.1% | 0.76 | $568 |
| SOL | 1h | 306 | 20.6 | -17.5% | 28.3% | 0.95 | $162 |
| ETH | 1h | 357 | 19.9 | -25.0% | 43.5% | 0.91 | $178 |
| BTC | 1h | 382 | 17.0 | -42.6% | 45.6% | 0.81 | $222 |
| SOL | 4h | 28 | 17.9 | -14.5% | 19.1% | 0.51 | $12 |
| ETH | 4h | 26 | 26.9 | -4.1% | 10.6% | 0.85 | $14 |
| BTC | 4h | 24 | 12.5 | -16.5% | 20.2% | 0.34 | $15 |

Buy & hold en el mismo año: SOL -55%, ETH -46%, BTC -32% (año bajista).

## Diagnóstico

1. **15m es inviable**: ~1500 trades/año ⇒ ~$450-950 de fees sobre una cuenta de $1000.
   La estrategia queda cerca de break-even bruto (PF 0.74-0.95) y las comisiones la hunden.
2. **Sin fees tampoco gana**: 1h con `--fee 0` da SOL -2.7%, ETH -7.0%, BTC -24.6%.
   O sea, el edge bruto es ≈ 0 y el coste operativo lo vuelve negativo.
3. **4h es el menos malo** (-4% a -16%), pero sigue perdiendo: 24-28 trades es muestra estadísticamente insuficiente.
4. **Win rate 17-27% con RR 3** exige >25% solo para empatar; el SAR reverse reabre posición
   inmediatamente tras cada salida, lo que genera churn (889 salidas por `sar` de 1467 en SOL).

## Variantes probadas en 1h (las 3 monedas siguen negativas)

- RR 2.0 → SOL -24.7% / ETH -43.0% / BTC -41.2% (peor)
- RR 1.5 → -33.6% / -30.6% / -43.1% (peor)
- ADX 30 → -8.2% / -14.9% / -41.6% (mejor en SOL/ETH: menos trades)
- SAR 3.0 → -27.9% / -12.9% / -35.5%

## Conclusión

Con el código y la configuración actuales **no hay edge demostrable en 1 año**. El mejor
resultado absoluto es 4h en ETH (-4.1%) y ni siquiera bate a estar fuera del mercado en un año bajista.
No poner capital real sin cambios de lógica, no solo de parámetros.

## Reproducir

    uv run python scripts/backtest_ema_adx.py --coins SOL,ETH,BTC --interval 15m --days 365
    uv run python scripts/backtest_ema_adx.py --coins SOL,ETH,BTC --interval 4h --days 365 --source hyperliquid
    uv run python scripts/backtest_ema_adx.py --coins SOL --interval 1h --days 365 --fee 0 --set adx_threshold=30

Artefactos: `backtest_results/ema_adx_*.json|csv` + `<COIN>_<tf>_trades.csv` (trade a trade).
