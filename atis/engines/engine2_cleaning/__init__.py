"""Engine 2 — Data Cleaning (incremental, quality-aware)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atis.config import (
    PROJECT_ROOT,
    ensure_project_dirs,
    get_path,
    load_engine_config,
    load_timeframes,
    set_global_seed,
)
from atis.engines.engine1_ingestion import raw_parquet_path
from atis.shared.data_json import export_timeframe_json
from atis.shared.data_registry import DataStateRegistry, compute_checksum
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.engine2")

LAYER = "clean"


@dataclass
class CleanResult:
    symbol: str
    timeframe: str
    rows_processed: int = 0
    rows_total: int = 0
    missing_filled: int = 0
    outliers_flagged: int = 0
    gaps_dropped: int = 0
    quality_score: float = 0.0
    first_ts: str | None = None
    last_ts: str | None = None
    error: str | None = None


@dataclass
class CleaningRunReport:
    started_at: str
    finished_at: str | None = None
    results: list[CleanResult] = field(default_factory=list)
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
                "outliers_flagged": sum(r.outliers_flagged for r in self.results),
            },
        }


def clean_parquet_path(symbol: str, timeframe: str) -> Path:
    base = get_path("data_clean")
    return base / symbol / timeframe / f"{symbol}_{timeframe}.parquet"


def clean_json_path(symbol: str, timeframe: str) -> Path:
    base = get_path("data_clean")
    return base / symbol / timeframe / f"{symbol}_{timeframe}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cfg() -> dict[str, Any]:
    return load_engine_config().get("engine2_cleaning", {})


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def validate_ohlc(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of logically invalid OHLC rows."""
    invalid = (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | df[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    return invalid


def flag_outliers(df: pd.DataFrame) -> pd.Series:
    """Flag return spikes via IQR or Z-score — never silent-delete."""
    cfg = _cfg()
    method = str(cfg.get("outlier_method", "iqr")).lower()
    returns = df["close"].pct_change()
    mask = pd.Series(False, index=df.index)

    if method == "zscore":
        thr = float(cfg.get("outlier_zscore_threshold", 4.0))
        mu = returns.mean()
        sigma = returns.std(ddof=0)
        if sigma and not np.isnan(sigma) and sigma > 0:
            mask = (returns - mu).abs() > thr * sigma
    else:
        mult = float(cfg.get("outlier_iqr_multiplier", 3.0))
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lower = q1 - mult * iqr
            upper = q3 + mult * iqr
            mask = (returns < lower) | (returns > upper)

    # Also flag invalid OHLC as outliers
    mask = mask.fillna(False) | validate_ohlc(df)
    return mask.fillna(False)


def align_timeframe_grid(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Ensure chronological order, unique timestamps, and drop misaligned bars
    for coarse timeframes (H1 must start at :00, etc.).
    """
    if df.empty:
        return df
    out = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    minutes = int(load_timeframes()[timeframe]["minutes"])

    if minutes >= 60 and minutes < 10080:
        # Hourly+ intraday: require minute==0 and hour divisible by step
        hours = minutes // 60
        ts = out["timestamp"]
        aligned = (ts.dt.minute == 0) & (ts.dt.second == 0) & ((ts.dt.hour % hours) == 0)
        out = out.loc[aligned]
    elif minutes == 10080:
        # Weekly — keep as-is (broker week start may vary)
        pass
    else:
        # Intraday minutes: timestamp minute should be multiple of TF
        ts = out["timestamp"]
        aligned = (ts.dt.second == 0) & ((ts.dt.minute % minutes) == 0)
        out = out.loc[aligned]

    return out.reset_index(drop=True)


def fill_short_gaps(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, int, int]:
    """
    Reindex to expected grid within span; fill short gaps; drop long ones.
    Returns (df, missing_filled, gaps_dropped).
    """
    cfg = _cfg()
    max_fill = int(cfg.get("max_gap_fill_bars", 3))
    strategy = str(cfg.get("fill_strategy", "linear")).lower()

    if df.empty or len(df) < 2:
        out = df.copy()
        if "is_imputed" not in out.columns:
            out["is_imputed"] = False
        return out, 0, 0

    minutes = int(load_timeframes()[timeframe]["minutes"])
    # Skip dense grid rebuild for weekly — too sparse / broker-dependent
    if minutes >= 10080:
        out = df.copy()
        out["is_imputed"] = False
        return out, 0, 0

    out = df.set_index("timestamp").sort_index()
    # Only fill within continuous weekday segments to avoid fabricating weekend bars
    full_index = pd.date_range(out.index.min(), out.index.max(), freq=f"{minutes}min", tz="UTC")
    # Keep only periods that are near existing trading hours density:
    # mark expected slots that fall on Saturday/Sunday as non-required
    trading_index = full_index[~full_index.dayofweek.isin([5, 6])]

    reindexed = out.reindex(trading_index)
    missing = reindexed["close"].isna()

    # Identify contiguous missing runs
    run_id = (~missing).cumsum()
    missing_filled = 0
    gaps_dropped = 0
    drop_mask = pd.Series(False, index=reindexed.index)

    for _, group in reindexed[missing].groupby(run_id[missing]):
        run_len = len(group)
        idx = group.index
        if run_len <= max_fill:
            missing_filled += run_len
        else:
            gaps_dropped += run_len
            drop_mask.loc[idx] = True

    # Drop long gaps entirely
    reindexed = reindexed.loc[~drop_mask]

    value_cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    value_cols = [c for c in value_cols if c in reindexed.columns]

    if strategy == "forward_fill":
        filled = reindexed.copy()
        filled[value_cols] = filled[value_cols].ffill(limit=max_fill)
    elif strategy == "drop":
        filled = reindexed.dropna(subset=["close"])
        missing_filled = 0
    else:
        # linear interpolation for numeric OHLC; volumes ffill
        filled = reindexed.copy()
        ohlc = [c for c in ["open", "high", "low", "close"] if c in filled.columns]
        filled[ohlc] = filled[ohlc].interpolate(method="time", limit=max_fill)
        vol_cols = [c for c in ["tick_volume", "spread", "real_volume"] if c in filled.columns]
        filled[vol_cols] = filled[vol_cols].ffill(limit=max_fill)

    # Rows that were missing before fill and now have close
    was_missing = missing.reindex(filled.index).fillna(False)
    still_missing = filled["close"].isna()
    imputed = was_missing & ~still_missing
    filled["is_imputed"] = imputed.fillna(False)

    # Drop any remaining NaN closes
    before_drop = len(filled)
    filled = filled.dropna(subset=["close"])
    gaps_dropped += before_drop - len(filled)

    # Preserve symbol/timeframe constants
    for col in ("symbol", "timeframe"):
        if col in out.columns:
            filled[col] = out[col].dropna().iloc[0] if out[col].notna().any() else None

    filled = filled.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in filled.columns:
        # reset_index already named timestamp if index name was timestamp
        pass
    # Ensure timestamp column name
    if filled.columns[0] != "timestamp" and "timestamp" not in filled.columns:
        filled = filled.rename(columns={filled.columns[0]: "timestamp"})

    return filled, int(imputed.sum()), int(gaps_dropped)


def clean_dataframe(raw: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, CleanResult]:
    """Full clean pipeline on a raw OHLCV frame (may be a delta slice + lookback)."""
    meta = CleanResult(symbol="", timeframe=timeframe)
    if raw.empty:
        return raw, meta

    df = normalize_timezone(raw)
    df = align_timeframe_grid(df, timeframe)
    df, filled, dropped = fill_short_gaps(df, timeframe)
    meta.missing_filled = filled
    meta.gaps_dropped = dropped

    if "is_imputed" not in df.columns:
        df["is_imputed"] = False

    outlier_mask = flag_outliers(df) if not df.empty else pd.Series(dtype=bool)
    df["is_outlier"] = outlier_mask.reindex(df.index).fillna(False).astype(bool)
    meta.outliers_flagged = int(df["is_outlier"].sum())

    # Quality score: 1 - (flagged + dropped) / max(rows,1) clipped
    n = max(len(df), 1)
    penalty = (meta.outliers_flagged + meta.gaps_dropped) / n
    meta.quality_score = float(max(0.0, min(1.0, 1.0 - penalty)))
    meta.rows_processed = len(df)

    if not df.empty:
        meta.first_ts = df["timestamp"].min().isoformat()
        meta.last_ts = df["timestamp"].max().isoformat()
        meta.rows_total = len(df)

    # Ensure no duplicate timestamps
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.reset_index(drop=True)
    return df, meta


def _merge_clean(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_parquet(path)
    if existing.empty:
        merged = new_rows.copy()
    elif new_rows.empty:
        merged = existing
    else:
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


def clean_symbol_timeframe(
    registry: DataStateRegistry,
    symbol: str,
    timeframe: str,
    *,
    force_rebuild: bool = False,
) -> CleanResult:
    raw_path = raw_parquet_path(symbol, timeframe)
    clean_path = clean_parquet_path(symbol, timeframe)
    result = CleanResult(symbol=symbol, timeframe=timeframe)

    try:
        raw = _read_parquet(raw_path)
        if raw.empty:
            result.error = "raw_data_missing"
            registry.upsert(
                symbol=symbol,
                timeframe=timeframe,
                layer=LAYER,
                last_run_status="empty",
            )
            return result

        if force_rebuild and clean_path.exists():
            clean_path.unlink()

        last_clean = None if force_rebuild else registry.last_updated_ts(symbol, timeframe, LAYER)
        existing_clean = _read_parquet(clean_path)
        if last_clean is None and not existing_clean.empty and not force_rebuild:
            last_clean = existing_clean["timestamp"].max().to_pydatetime()

        # Incremental: process only new raw rows, but keep lookback for continuity
        lookback_bars = 50
        if last_clean is not None:
            minutes = int(load_timeframes()[timeframe]["minutes"])
            lookback_start = last_clean - pd.Timedelta(minutes=minutes * lookback_bars)
            slice_df = raw[raw["timestamp"] >= lookback_start].copy()
        else:
            slice_df = raw.copy()

        cleaned, meta = clean_dataframe(slice_df, timeframe)
        result.missing_filled = meta.missing_filled
        result.outliers_flagged = meta.outliers_flagged
        result.gaps_dropped = meta.gaps_dropped
        result.quality_score = meta.quality_score

        if last_clean is not None and not cleaned.empty:
            # Write only rows newer than last_clean (incremental write)
            to_write = cleaned[cleaned["timestamp"] > last_clean].copy()
        else:
            to_write = cleaned

        # On full rebuild write everything
        if force_rebuild or last_clean is None:
            to_write = cleaned

        result.rows_processed = len(to_write)
        merged = _merge_clean(clean_path, to_write)
        export_timeframe_json(
            merged,
            clean_json_path(symbol, timeframe),
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
        )
        result.rows_total = len(merged)

        if merged.empty:
            result.error = "clean_empty"
            registry.upsert(
                symbol=symbol,
                timeframe=timeframe,
                layer=LAYER,
                last_run_status="empty",
                row_count=0,
            )
            return result

        result.first_ts = merged["timestamp"].min().isoformat()
        result.last_ts = merged["timestamp"].max().isoformat()
        # Recompute quality on full clean set lightly
        result.quality_score = meta.quality_score

        checksum = compute_checksum(
            [result.first_ts, result.last_ts, result.rows_total, float(merged["close"].iloc[-1])]
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
            extra_json=json.dumps(
                {
                    "missing_filled": result.missing_filled,
                    "outliers_flagged": result.outliers_flagged,
                    "gaps_dropped": result.gaps_dropped,
                    "quality_score": result.quality_score,
                }
            ),
        )
        registry.audit(
            "engine2",
            "clean_success",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps(asdict(result)),
        )
        logger.info(
            "clean_done",
            symbol=symbol,
            timeframe=timeframe,
            rows_processed=result.rows_processed,
            rows_total=result.rows_total,
            quality_score=result.quality_score,
            outliers=result.outliers_flagged,
        )
        return result

    except Exception as exc:
        result.error = str(exc)
        logger.exception("clean_failed", symbol=symbol, timeframe=timeframe, error=str(exc))
        registry.upsert(
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
            last_run_status="error",
        )
        registry.audit(
            "engine2",
            "clean_error",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps({"error": str(exc)}),
        )
        return result


def run_cleaning(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    force_rebuild: bool = False,
) -> CleaningRunReport:
    ensure_project_dirs()
    set_global_seed()
    cfg_e1 = load_engine_config().get("engine1_ingestion", {})
    if symbols is None:
        symbols = list(cfg_e1.get("default_symbols", ["XAUUSD"]))
    if timeframes is None:
        timeframes = list(cfg_e1.get("default_timeframes", ["H1"]))

    report = CleaningRunReport(started_at=_utc_now().isoformat())
    registry = DataStateRegistry()

    for symbol in symbols:
        for tf in timeframes:
            report.results.append(
                clean_symbol_timeframe(
                    registry,
                    symbol,
                    tf,
                    force_rebuild=force_rebuild,
                )
            )

    report.finished_at = _utc_now().isoformat()
    report.status = "success" if not any(r.error for r in report.results) else "partial_error"

    out_dir = PROJECT_ROOT / "logs" / "cleaning"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(report.to_dict(), indent=2)
    (out_dir / f"cleaning_run_report_{stamp}.json").write_text(payload, encoding="utf-8")
    (out_dir / "cleaning_run_report.json").write_text(payload, encoding="utf-8")
    # Per-run quality table
    quality_rows = [asdict(r) for r in report.results]
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(quality_rows, indent=2),
        encoding="utf-8",
    )
    logger.info("cleaning_report_written", status=report.status)
    return report
