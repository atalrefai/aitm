# Engine 5 — Live Trading (Paper / Demo)

**Risk management is mandatory.** No order without SL/TP.
Real `live` mode is refused until explicit user approval.

```bash
# Paper inference (no orders)
python -m atis.engines.engine5_live_trading.run --symbols EURUSD --timeframe H1

# Demo account orders
python -m atis.engines.engine5_live_trading.run --symbols EURUSD --timeframe H1 --execute-demo

# Loop
python -m atis.engines.engine5_live_trading.run --symbols EURUSD --timeframe H1 --loop --max-iterations 5
```

Kill switch: set `engine5_live.kill_switch: true` in `config/engine_config.yaml`.
