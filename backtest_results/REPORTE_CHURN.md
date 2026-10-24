# Resultado: control de churn (SAR reverse + cooldown) — 1 año, 1h, Binance

Rama: `feat/ema-adx-churn` · Baseline = `allow_sar_reverse=true`, sin cooldown (comportamiento live actual).

| Variante | SOL | ETH | BTC |
|---|---|---|---|
| Base (neto, con fee 0.045%) | -17.48% | -24.99% | -42.59% |
| Sin reverse (neto) | -36.84% | -28.17% | -41.37% |
| Sin reverse + cooldown 3 velas (neto) | -24.17% | -12.23% | -33.69% |
| Base **bruto** (fee 0) | -2.69% | -7.04% | -24.59% |
| Sin reverse + cd3 **bruto** (fee 0) | -14.86% | +0.68% | -20.35% |

## Veredicto
- **Quitar el reverse empeora** el resultado (menos trades, mismo win rate).
- Cooldown 3 velas ayuda (ETH pasa de -25% a -12%) pero **sigue perdiendo**.
- Lo decisivo: **a fee cero el edge bruto es ~0 o negativo** (SOL -2.7%, ETH -7%, BTC -24.6%). El strategy no tiene ventaja estadística; el fee solo amplifica la pérdida.
- Ninguna combinación de parámetros llega a PF > 1.2 de forma consistente → no es cuestión de tuning.

## Conclusión
No poner en live. El problema es la señal de entrada (cruce EMA + ADX), no la gestión de salida.

## Qué sí queda hecho
- `allow_sar_reverse` y `cooldown_bars_after_loss` en `EmaAdxConfig` (defaults = comportamiento anterior, no cambia nada en live).
- `--tag` y `--set` con tipos (bool/float) en el script de backtest.
- 37 tests verdes.
