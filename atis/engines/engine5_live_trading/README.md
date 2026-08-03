# Engine 5 — Live Trading (Paper / Demo)

**Risk management is mandatory.** No order without SL/TP.
SL/TP are **near-entry and model-driven** by default (`engine5_live.dynamic_exits`):
expected return, confidence, risk score, *local* support/resistance (within ~2×ATR),
and live ATR — not distant historical levels or fixed pip offsets.
Set `dynamic_exits.enabled: false` to fall back to fixed ATR multipliers.
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

## Multi-timeframe (autotrader)

When several timeframes are selected, **each TF runs independently** by default
(`multi_tf_independent: true`, `multi_tf_fusion: false`):

1. Load that TF’s trained model
2. Analyze that TF’s bars/features
3. Decide and place (or skip) without consulting other TFs

Set `multi_tf_fusion: true` to restore legacy vote-fusion into one order.
`max_open_positions_per_tf` caps open ATIS positions tagged with that TF’s comment (`ATIS|H1|…`).
