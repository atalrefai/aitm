# Engine 3 — Feature & Pattern Engine

Uses `shared/feature_engine` (same code path as live trading).

```bash
python -m atis.engines.engine3_features.run --symbols EURUSD --timeframes H1 --force-rebuild
```

Output: `data/features/{symbol}/{timeframe}/features.parquet`
