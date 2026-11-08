# Medición 1 año — EMA+ADX (SOL/ETH/BTC)

Generado con `scripts/run_matrix.py` sobre `scripts/backtest_ema_adx.py`.
Ventana: 365 días de datos Binance 1m, replay bar-a-bar con ticks de 1m para
fidelidad intrabar. Balance inicial $1,000, riesgo 1% por trade, sin apalancamiento
adicional (`avg_leverage` < 1 en todas las corridas). Comisión taker 0.045%/fill
(salvo variantes `nofee` y `maker`).

## Qué se midió

6 estructuras de gestión sobre la misma señal (cruce EMA 3/9 + ADX > 23, SL 2·ATR, TP 3R):

| variante | entrada | salidas |
|---|---|---|
| `base` | 1 entrada, 100% del riesgo | SL / TP / reversión SAR |
| `ladder` | escalera 50/25/25, paso 0.5·ATR | SL / TP / SAR |
| `partial` | 1 entrada | parciales 50% en 1R y 30% en 2R |
| `both` | escalera | parciales |
| `wide` | escalera 40/30/30, paso 1·ATR | parciales 50% en 1R y 50% en 2R |
| `both_nofee` | = `both` | = `both`, comisión 0 |
| `both_maker` / `wide_maker` | = `both` / `wide` | comisión maker 0.015% |

## Resultado 15m (el timeframe que veníamos usando)

| coin | variante | neto | PF | legs | maxDD | fees |
|---|---|---|---|---|---|---|
| SOL | base | **-76.5%** | 0.95 | 1467 | 84.9% | $949 |
| SOL | ladder | -50.0% | 0.99 | 1156 | 69.5% | $860 |
| SOL | wide | -59.8% | 0.84 | 1898 | 60.2% | $444 |
| SOL | both_nofee | -23.1% | 0.95 | 1945 | 47.9% | $0 |
| SOL | wide_maker | -40.0% | 0.88 | 1880 | 43.1% | $174 |
| ETH | base | **-88.5%** | 0.74 | 1506 | 89.3% | $445 |
| ETH | ladder | -71.4% | 0.81 | 1139 | 72.8% | $459 |
| ETH | wide | -56.7% | 0.83 | 1909 | 57.1% | $433 |
| ETH | both_nofee | -30.0% | 0.90 | 1978 | 34.4% | $0 |
| ETH | wide_maker | -34.8% | 0.89 | 1912 | 35.6% | $173 |
| BTC | base | **-84.8%** | 0.76 | 1545 | 85.1% | $568 |
| BTC | ladder | -75.7% | 0.74 | 1219 | 76.1% | $475 |
| BTC | wide | -61.6% | 0.76 | 2004 | 62.4% | $439 |
| BTC | both_nofee | -38.2% | 0.84 | 2039 | 40.8% | $0 |
| BTC | wide_maker | -39.3% | 0.83 | 1990 | 41.2% | $177 |

## Resultado 4h (misma señal, 16x menos operaciones)

| coin | base | ladder | partial | both | wide | both_nofee |
|---|---|---|---|---|---|---|
| SOL | -18.0% | -4.8% | -9.1% | -7.9% | -7.3% | -6.5% |
| ETH | -7.2% | -5.9% | -4.4% | -1.6% | **+6.2%** (PF 1.31, 106 legs) | +0.1% |
| BTC | -19.5% | -14.7% | -17.7% | -12.4% | -6.5% | -9.5% |

Buy & hold del mismo periodo: BTC -32.3%, ETH -46.6%, SOL -55.5%.

## Lectura

1. **No hay edge bruto.** Con comisión 0 el 15m sigue perdiendo (-23% a -38%).
   PF sin comisiones 0.84–0.95 en todas las monedas y variantes.
2. **El 15m muere por costo de fricción.** 1,100–2,400 legs/año × 2 fills × 0.045%
   sobre ~$1,000 de balance = $444–949 de comisiones, 44–95% del capital.
   Pasar a maker recorta el drag a la mitad, pero no cambia el signo.
3. **La escalera y los parciales no crean edge**: bajan el drawdown y el tamaño
   promedio, pero con PF < 1 solo reducen la velocidad de la pérdida.
4. **El único resultado positivo del año (ETH 4h `wide`, +6.2%) tiene 106 legs**:
   estadísticamente indistinguible de cero. No es base para desplegar capital.
5. **Ninguna variante superó a buy & hold** en 15m; en 4h varias perdieron menos
   que hold en un año bajista, que es lo esperable de una estrategia de tendencia.

## Veredicto

Esta configuración (EMA 3/9 + ADX 23 en 15m) **no es desplegable**. La estructura
de salidas no es el problema: la señal no tiene expectativa positiva antes de costos.
Antes de volver a capital real hacen falta: (a) filtro de tendencia en marco superior
(1d/4h) para no operar contra el sesgo, (b) operar en 4h como mínimo, (c) ejecución
maker obligatoria, y (d) re-medir con walk-forward en vez de un solo año.

## Reproducir

```bash
uv run python scripts/run_matrix.py SOL 365 15m   # escribe backtest_results/matrix_SOL_15m_365d.json
uv run pytest tests/ -q                            # 54 tests, incluye escalera/parciales/broker
```
