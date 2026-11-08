## Extensible grid trading bot for [Hyperliquid DEX](https://hyperliquid.xyz)

> ⚠️ This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Never trade with funds you cannot afford to lose. Always thoroughly test strategies on testnet before live deployment.

This project is under active development. Feel free to submit questions, suggestions, and issues through GitHub.

You're welcome to use the best docs on Hyperliquid API via [Chainstack Developer Portal MCP server](https://docs.chainstack.com/docs/developer-portal-mcp-server).

## 🚀 Quick start

### **Prerequisites**
- [uv package manager](https://github.com/astral-sh/uv)
- Hyperliquid testnet account with testnet funds (see [Chainstack Hyperliquid faucet](https://faucet.chainstack.com/hyperliquid-testnet-faucet))

### **Installation**

```bash
# Clone the repository
git clone https://github.com/chainstacklabs/hyperliquid-trading-bot
cd hyperliquid-trading-bot

# Install dependencies using uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your Hyperliquid testnet private key
```

### **Configuration**

Create your environment file:
```bash
# .env
HYPERLIQUID_TESTNET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
HYPERLIQUID_TESTNET=true
```

The bot comes with a pre-configured conservative BTC grid strategy in `bots/btc_conservative.yaml`. Review and adjust parameters as needed.

### **Running the bot**

```bash
# Auto-discover and run the first active configuration
uv run src/run_bot.py

# Validate configuration before running
uv run src/run_bot.py --validate

# Run specific configuration
uv run src/run_bot.py bots/btc_conservative.yaml
```

## ⚙️ Configuration

Bot configurations use YAML format with comprehensive parameter documentation:

```yaml
# Conservative BTC Grid Strategy
name: "btc_conservative_clean"
active: true  # Enable/disable this strategy

account:
  max_allocation_pct: 10.0  # Use only 10% of account balance

grid:
  symbol: "BTC"
  levels: 10               # Number of grid levels
  price_range:
    mode: "auto"           # Auto-calculate from current price
    auto:
      range_pct: 5.0      # ±5% price range (conservative)

risk_management:
  # Exit Strategies
  stop_loss_enabled: false      # Auto-close positions on loss threshold
  stop_loss_pct: 8.0           # Loss % before closing (1-20%)
  take_profit_enabled: false   # Auto-close positions on profit threshold
  take_profit_pct: 25.0        # Profit % before closing (5-100%)
  
  # Account Protection
  max_drawdown_pct: 15.0       # Stop trading on account drawdown % (5-50%)
  max_position_size_pct: 40.0  # Max position as % of account (10-100%)
  
  # Grid Rebalancing
  rebalance:
    price_move_threshold_pct: 12.0  # Rebalance trigger

monitoring:
  log_level: "INFO"       # DEBUG/INFO/WARNING/ERROR
```

## 📚 Learning examples

Master the Hyperliquid API with standalone educational scripts:

```bash
# Market data and pricing
uv run learning_examples/02_market_data/get_all_prices.py
uv run learning_examples/02_market_data/get_market_metadata.py

# Account information
uv run learning_examples/03_account_info/get_user_state.py
uv run learning_examples/03_account_info/get_open_orders.py

# Trading operations
uv run learning_examples/04_trading/place_limit_order.py
uv run learning_examples/04_trading/cancel_orders.py

# Real-time data
uv run learning_examples/01_websockets/realtime_prices.py
```

## 🛡️ Exit strategies

The bot includes automated risk management and position exit features:

**Position-level exits:**
- **Stop loss**: Automatically close positions when loss exceeds configured percentage (1-20%)
- **Take profit**: Automatically close positions when profit exceeds configured percentage (5-100%)

**Account-level protection:**
- **Max drawdown**: Stop all trading when account-level losses exceed threshold (5-50%)
- **Position size limits**: Prevent individual positions from exceeding percentage of account (10-100%)

**Operational exits:**
- **Grid rebalancing**: Cancel orders and recreate grid when price moves outside range
- **Graceful shutdown**: Cancel pending orders on bot termination (positions preserved by default)

All exit strategies are configurable per bot and disabled by default for safety.

## 📈 EMA/ADX trend strategy

Besides the grid strategy, `src/strategies/ema_adx/` ships a trend-following
strategy ported from the standalone `hyperliquid-tradingbot` project.

How it works:

- Resamples live trades into candles for the configured `interval` (default `15m`).
- Seeds ~500 historical candles from Hyperliquid `/info` on start, so it can trade
  without waiting for warmup.
- Entry: EMA fast/slow crossover **confirmed on a closed candle** with
  `ADX >= adx_threshold` (default 25).
- Exit (checked on every tick against the real position): ATR stop loss,
  ATR take profit, or a SAR-style reversal trail that also flips the side.
- Sizing: risk `risk_per_trade_pct` of balance over `atr * atr_sl_multiplier`,
  capped by `max_position_pct` of available balance.

Config lives in the same bot YAML (`bots/sol_ema_adx.yaml`, `eth_ema_adx.yaml`,
`btc_ema_adx.yaml`):

```yaml
strategy:
  type: ema_adx        # selects EmaAdxStrategy
  symbol: SOL
  interval: 15m
  ema_fast: 3
  ema_slow: 9
  adx_threshold: 23
  risk_per_trade: 0.01
```

```bash
uv run src/run_bot.py bots/sol_ema_adx.yaml --validate   # check config
uv run src/run_bot.py bots/sol_ema_adx.yaml              # testnet run
```

Tests (offline, no keys needed): `uv run --extra test pytest tests -q`.

Note: a `CLOSE` signal now exits with a **reduce-only market order** in
`TradingEngine._close_positions` (previously close signals were ignored).

## 🐳 Docker

The image ships Python 3.13, every dependency from `uv.lock`, `src/` and
`bots/`, runs as non-root (`uid 1000`) and defaults to `bots/sol_ema_adx.yaml`.

```bash
docker build -t hl-engine:latest .

# validate a config without keys
docker run --rm hl-engine:latest --validate bots/sol_ema_adx.yaml

# run one bot; keys come from .env (never baked into the image)
docker run -d --name hl-sol --restart unless-stopped \
  --env-file .env -v "$PWD/bots:/app/bots:ro" \
  hl-engine:latest bots/sol_ema_adx.yaml
```

Or with compose (one profile per pair, any number can run side by side):

```bash
cp .env.example .env          # set HYPERLIQUID_TESTNET_PRIVATE_KEY / HYPERLIQUID_TESTNET
docker compose --profile sol --profile eth up -d --build
docker compose logs -f sol
docker compose --profile tools run --rm validate   # config check only
```

Credential options inside the container: `HYPERLIQUID_TESTNET_PRIVATE_KEY`,
`HYPERLIQUID_MAINNET_PRIVATE_KEY`, the `*_KEY_FILE` variants (mount keys under
`./keys:/keys:ro`), or a per-bot key in the YAML (not recommended).

Deploy on a small VPS without building there (2 GB boxes choke on the build):

```bash
docker save hl-engine:latest | gzip -1 > hl-engine.tgz   # on the build machine
scp hl-engine.tgz user@vps:/tmp/ && ssh user@vps \
  'gunzip -c /tmp/hl-engine.tgz | docker load && cd ~/hl-engine && docker compose --profile sol up -d'
```

`restart: unless-stopped` plus the container `HEALTHCHECK` (checks the runner
process is alive) keeps the bot up across reboots and crashes.

## 🔧 Development

### **Package management**
This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management:

```bash
uv sync              # Install/sync dependencies
uv add <package>     # Add new dependencies
uv run <command>     # Run commands in virtual environment
```

### **Testing**
All components are tested against Hyperliquid testnet:

```bash
# Test learning examples
uv run learning_examples/04_trading/place_limit_order.py

# Validate bot configuration
uv run src/run_bot.py --validate

# Run bot in testnet mode (default)
uv run src/run_bot.py
```
