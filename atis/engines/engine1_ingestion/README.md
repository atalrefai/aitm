# Engine 1 — Data Ingestion

## Purpose
Connect to MetaTrader 5 and fetch OHLCV bars with **incremental updates**.

## Inputs
- `config/symbols.yaml`, `config/timeframes.yaml`, `config/engine_config.yaml`
- `config/secrets.env` (MT5 credentials)
- `data/registry/data_state_registry.db` (`last_updated_ts`)

## Outputs
- `data/raw/{symbol}/{timeframe}/{symbol}_{timeframe}.parquet`
- Registry updates + `logs/ingestion/ingestion_run_report.json`

## Run
```bash
# Smoke: EURUSD H1 (defaults)
python -m atis.engines.engine1_ingestion.run

# Explicit
python -m atis.engines.engine1_ingestion.run --symbols EURUSD --timeframes H1

# Full rebuild of selected series
python -m atis.engines.engine1_ingestion.run --symbols EURUSD --timeframes H1 --force-rebuild
```
