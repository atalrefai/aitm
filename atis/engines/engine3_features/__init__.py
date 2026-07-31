"""Engine 3 — Feature & Pattern Engine (per symbol/timeframe, incremental)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atis.config import (
    PROJECT_ROOT,
    ensure_project_dirs,
    get_path,
    load_engine_config,
    load_timeframes,
    set_global_seed,
)
from atis.engines.engine2_cleaning import clean_parquet_path
from atis.shared.data_json import export_timeframe_json
from atis.shared.data_registry import DataStateRegistry, compute_checksum
from atis.shared.feature_engine import compute_features, load_indicators_config
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.engine3")

LAYER = "features"


@dataclass
class FeatureResult:
    symbol: str
    timeframe: str
    rows_processed: int = 0
    rows_total: int = 0
    feature_count: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    error: str | None = None


@dataclass
class FeatureRunReport:
    started_at: str
    finished_at: str | None = None
    results: list[FeatureResult] = field(default_factory=list)
    status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "results": [asdict(r) for r in self.results],
            "summary": {
                "symbols": len(self.results),
                "errors": sum(1 for r in self.results if r.error),
                "rows_processed": sum(r.rows_processed for r in self.results),
            },
        }


def features_parquet_path(symbol: str, timeframe: str) -> Path:
    base = get_path("data_features")
    return base / symbol / timeframe / f"features.parquet"


def features_json_path(symbol: str, timeframe: str) -> Path:
    base = get_path("data_features")
    return base / symbol / timeframe / "features.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _merge_features(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read(path)
    if existing.empty:
        merged = new_rows.copy()
    elif new_rows.empty:
        merged = existing
    else:
        # Align columns
        all_cols = list(dict.fromkeys(list(existing.columns) + list(new_rows.columns)))
        existing = existing.reindex(columns=all_cols)
        new_rows = new_rows.reindex(columns=all_cols)
        merged = pd.concat([existing, new_rows], ignore_index=True)
    if merged.empty:
        return merged
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged = (
        merged.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    merged.to_parquet(path, index=False, engine="pyarrow")
    return merged


def compute_symbol_timeframe(
    registry: DataStateRegistry,
    symbol: str,
    timeframe: str,
    *,
    force_rebuild: bool = False,
) -> FeatureResult:
    result = FeatureResult(symbol=symbol, timeframe=timeframe)
    clean_path = clean_parquet_path(symbol, timeframe)
    feat_path = features_parquet_path(symbol, timeframe)
    cfg = load_engine_config().get("engine3_features", {})
    lookback = int(cfg.get("lookback_bars", 250))
    ind_cfg = load_indicators_config()

    try:
        clean = _read(clean_path)
        if clean.empty:
            result.error = "clean_data_missing"
            registry.upsert(symbol=symbol, timeframe=timeframe, layer=LAYER, last_run_status="empty")
            return result

        if force_rebuild and feat_path.exists():
            feat_path.unlink()

        last_feat = None if force_rebuild else registry.last_updated_ts(symbol, timeframe, LAYER)
        existing = _read(feat_path)
        if last_feat is None and not existing.empty and not force_rebuild:
            last_feat = existing["timestamp"].max().to_pydatetime()

        minutes = int(load_timeframes()[timeframe]["minutes"])
        if last_feat is not None:
            start = last_feat - pd.Timedelta(minutes=minutes * lookback)
            window = clean[clean["timestamp"] >= start].copy()
        else:
            window = clean.copy()

        featured = compute_features(window, ind_cfg)
        # Count feature columns (exclude OHLCV meta)
        base_cols = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
            "symbol",
            "timeframe",
            "is_imputed",
            "is_outlier",
            "label",
            "label_meta",
        }
        result.feature_count = len([c for c in featured.columns if c not in base_cols])

        if last_feat is not None and not force_rebuild:
            to_write = featured[featured["timestamp"] > last_feat].copy()
        else:
            to_write = featured

        result.rows_processed = len(to_write)
        merged = _merge_features(feat_path, to_write)
        export_timeframe_json(
            merged,
            features_json_path(symbol, timeframe),
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
        )
        result.rows_total = len(merged)

        if merged.empty:
            result.error = "features_empty"
            registry.upsert(symbol=symbol, timeframe=timeframe, layer=LAYER, last_run_status="empty")
            return result

        result.first_ts = merged["timestamp"].min().isoformat()
        result.last_ts = merged["timestamp"].max().isoformat()
        checksum = compute_checksum(
            [result.first_ts, result.last_ts, result.rows_total, result.feature_count]
        )
        registry.upsert(
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
            first_available_ts=merged["timestamp"].min().to_pydatetime(),
            last_updated_ts=merged["timestamp"].max().to_pydatetime(),
            last_run_status="success",
            row_count=result.rows_total,
            checksum=checksum,
            extra_json=json.dumps({"feature_count": result.feature_count}),
        )
        registry.audit(
            "engine3",
            "features_success",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps(asdict(result)),
        )
        logger.info(
            "features_done",
            symbol=symbol,
            timeframe=timeframe,
            rows_processed=result.rows_processed,
            rows_total=result.rows_total,
            feature_count=result.feature_count,
        )
        return result

    except Exception as exc:
        result.error = str(exc)
        logger.exception("features_failed", symbol=symbol, timeframe=timeframe, error=str(exc))
        registry.upsert(symbol=symbol, timeframe=timeframe, layer=LAYER, last_run_status="error")
        registry.audit(
            "engine3",
            "features_error",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps({"error": str(exc)}),
        )
        return result


def run_features(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    force_rebuild: bool = False,
) -> FeatureRunReport:
    ensure_project_dirs()
    set_global_seed()
    cfg_e1 = load_engine_config().get("engine1_ingestion", {})
    if symbols is None:
        symbols = list(cfg_e1.get("default_symbols", ["XAUUSD"]))
    if timeframes is None:
        timeframes = list(cfg_e1.get("default_timeframes", ["H1"]))

    report = FeatureRunReport(started_at=_utc_now().isoformat())
    registry = DataStateRegistry()
    tasks = [(symbol, tf) for symbol in symbols for tf in timeframes]
    if len(tasks) <= 1:
        for symbol, tf in tasks:
            report.results.append(
                compute_symbol_timeframe(
                    registry, symbol, tf, force_rebuild=force_rebuild
                )
            )
    else:
        ordered_results: list[FeatureResult | None] = [None] * len(tasks)
        max_workers = len(tasks)
        logger.info("features_parallel_start", tasks=len(tasks), workers=max_workers)
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="engine3_features",
        ) as executor:
            future_to_idx = {
                executor.submit(
                    compute_symbol_timeframe,
                    registry,
                    symbol,
                    tf,
                    force_rebuild=force_rebuild,
                ): idx
                for idx, (symbol, tf) in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered_results[idx] = future.result()
        report.results.extend([r for r in ordered_results if r is not None])
    report.finished_at = _utc_now().isoformat()
    report.status = "success" if not any(r.error for r in report.results) else "partial_error"

    out_dir = PROJECT_ROOT / "logs" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"features_run_report_{stamp}.json").write_text(payload, encoding="utf-8")
    (out_dir / "features_run_report.json").write_text(payload, encoding="utf-8")
    logger.info("features_report_written", status=report.status)
    return report
