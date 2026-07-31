# Engine 2 — Data Cleaning

## Purpose
Convert `data/raw` → `data/clean` with missing-data handling, outlier flags, UTC normalization, and incremental processing.

## Run
```bash
python -m atis.engines.engine2_cleaning.run --symbols EURUSD --timeframes H1
python -m atis.engines.engine2_cleaning.run --symbols EURUSD --timeframes H1 --force-rebuild
```

## Outputs
- `data/clean/{symbol}/{timeframe}/{symbol}_{timeframe}.parquet` (includes `is_imputed`, `is_outlier`)
- `logs/cleaning/data_quality_report.json`
