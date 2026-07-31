"""Engine 1 — Data Ingestion from MetaTrader 5 (incremental OHLCV)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
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
from atis.shared.data_json import export_timeframe_json
from atis.shared.data_registry import DataStateRegistry, compute_checksum
from atis.shared.logging_utils import get_logger
from atis.shared.mt5_client import MT5Client, mt5_session

logger = get_logger("atis.engine1")

LAYER = "raw"


@dataclass
class SymbolIngestResult:
    symbol: str
    timeframe: str
    mode: str  # backfill | incremental | skipped
    rows_fetched: int = 0
    rows_written: int = 0
    rows_total: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    gaps_detected: int = 0
    error: str | None = None


@dataclass
class IngestionRunReport:
    started_at: str
    finished_at: str | None = None
    force_rebuild: bool = False
    results: list[SymbolIngestResult] = field(default_factory=list)
    status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "force_rebuild": self.force_rebuild,
            "status": self.status,
            "results": [asdict(r) for r in self.results],
            "summary": {
                "symbols": len(self.results),
                "errors": sum(1 for r in self.results if r.error),
                "rows_written": sum(r.rows_written for r in self.results),
                "gaps_detected": sum(r.gaps_detected for r in self.results),
            },
        }


def raw_parquet_path(symbol: str, timeframe: str) -> Path:
    base = get_path("data_raw")
    return base / symbol / timeframe / f"{symbol}_{timeframe}.parquet"


def raw_json_path(symbol: str, timeframe: str) -> Path:
    base = get_path("data_raw")
    return base / symbol / timeframe / f"{symbol}_{timeframe}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _version_hash(symbol: str, timeframe: str, first_ts: str, last_ts: str, rows: int) -> str:
    payload = f"{symbol}|{timeframe}|{first_ts}|{last_ts}|{rows}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _expected_bar_delta(timeframe: str) -> timedelta:
    tfs = load_timeframes()
    minutes = int(tfs[timeframe]["minutes"])
    return timedelta(minutes=minutes)


def detect_gaps(df: pd.DataFrame, timeframe: str) -> list[dict[str, Any]]:
    """
    Detect suspicious gaps between consecutive bars.
    Weekend / multi-day gaps on FX are ignored (natural market close).
    """
    if df.empty or len(df) < 2:
        return []
    delta = _expected_bar_delta(timeframe)
    # Allow 1.5x expected spacing before flagging; skip weekend-sized holes
    max_ok = delta * 1.5
    weekend_threshold = timedelta(hours=36)

    gaps: list[dict[str, Any]] = []
    ts = df["timestamp"].sort_values().reset_index(drop=True)
    diffs = ts.diff().iloc[1:]
    for i, d in enumerate(diffs, start=1):
        if pd.isna(d):
            continue
        if d <= max_ok:
            continue
        if d >= weekend_threshold:
            # Likely weekend / holiday — not a data gap for FX
            continue
        gaps.append(
            {
                "from": ts.iloc[i - 1].isoformat(),
                "to": ts.iloc[i].isoformat(),
                "gap": str(d),
            }
        )
    return gaps


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _upsert_parquet(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent merge on timestamp — Principle 1.1 / Idempotency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(path)
    if existing.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = existing
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
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


def _chunked_history(
    client: MT5Client,
    symbol: str,
    timeframe: str,
    date_from: datetime,
    date_to: datetime,
    chunk_bars: int,
) -> pd.DataFrame:
    """
    Fetch history in chunks. MT5 often caps ~10k–100k bars per call;
    we walk forward by time windows derived from timeframe minutes.
    """
    tfs = load_timeframes()
    minutes = int(tfs[timeframe]["minutes"])
    # Approximate window covered by chunk_bars
    window = timedelta(minutes=minutes * chunk_bars)
    frames: list[pd.DataFrame] = []
    cursor = date_from
    safety = 0
    max_loops = 5000

    while cursor < date_to and safety < max_loops:
        safety += 1
        end = min(cursor + window, date_to)
        part = client.copy_rates_range(symbol, timeframe, cursor, end)
        if not part.empty:
            frames.append(part)
            last = part["timestamp"].max().to_pydatetime()
            # Advance past last bar to avoid infinite loop on same window
            next_cursor = last + timedelta(minutes=minutes)
            if next_cursor <= cursor:
                next_cursor = end
            cursor = next_cursor
        else:
            cursor = end

        if end >= date_to:
            break

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = (
        out.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    # Clip to requested range
    out = out[(out["timestamp"] >= date_from) & (out["timestamp"] <= date_to)]
    return out.reset_index(drop=True)


def ingest_symbol_timeframe(
    client: MT5Client,
    registry: DataStateRegistry,
    symbol: str,
    timeframe: str,
    *,
    force_rebuild: bool = False,
) -> SymbolIngestResult:
    cfg = load_engine_config().get("engine1_ingestion", {})
    backfill_years = int(cfg.get("backfill_years", 3))
    chunk_bars = int(cfg.get("max_bars_per_request", 10000))
    path = raw_parquet_path(symbol, timeframe)
    now = _utc_now()

    result = SymbolIngestResult(symbol=symbol, timeframe=timeframe, mode="incremental")

    try:
        resolved = client.resolve_symbol(symbol)
        logger.info("ingest_start", symbol=symbol, resolved=resolved, timeframe=timeframe)

        if force_rebuild and path.exists():
            path.unlink()
            logger.info("force_rebuild_deleted", path=str(path))

        last_ts = None if force_rebuild else registry.last_updated_ts(symbol, timeframe, LAYER)
        existing = _read_existing(path)
        if last_ts is None and not existing.empty and not force_rebuild:
            last_ts = existing["timestamp"].max().to_pydatetime()

        if last_ts is None:
            result.mode = "backfill"
            date_from = now - timedelta(days=365 * backfill_years)
            date_to = now
        else:
            result.mode = "incremental"
            # Start slightly before last bar to refresh incomplete candle
            minutes = int(load_timeframes()[timeframe]["minutes"])
            date_from = last_ts - timedelta(minutes=minutes)
            date_to = now

        fetched = _chunked_history(client, symbol, timeframe, date_from, date_to, chunk_bars)
        result.rows_fetched = len(fetched)

        if fetched.empty and existing.empty:
            result.error = "no_data_returned"
            registry.upsert(
                symbol=symbol,
                timeframe=timeframe,
                layer=LAYER,
                last_run_status="empty",
                row_count=0,
            )
            registry.audit(
                "engine1",
                "ingest_empty",
                symbol=symbol,
                timeframe=timeframe,
            )
            return result

        before_count = len(existing)
        merged = _upsert_parquet(path, fetched)
        export_timeframe_json(
            merged,
            raw_json_path(symbol, timeframe),
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
        )
        result.rows_total = len(merged)
        result.rows_written = max(0, result.rows_total - before_count)
        # On rebuild / refresh of last bars, rows_written may be 0 even if fetch > 0
        if force_rebuild:
            result.rows_written = result.rows_total

        gaps = detect_gaps(merged, timeframe)
        result.gaps_detected = len(gaps)

        first_ts = merged["timestamp"].min().to_pydatetime()
        last_updated = merged["timestamp"].max().to_pydatetime()
        result.first_ts = first_ts.isoformat()
        result.last_ts = last_updated.isoformat()

        checksum = compute_checksum(
            [
                result.first_ts,
                result.last_ts,
                result.rows_total,
                float(merged["close"].iloc[-1]),
            ]
        )
        vhash = _version_hash(symbol, timeframe, result.first_ts, result.last_ts, result.rows_total)

        registry.upsert(
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
            first_available_ts=first_ts,
            last_updated_ts=last_updated,
            last_run_status="success",
            row_count=result.rows_total,
            checksum=checksum,
            version_hash=vhash,
            extra_json=json.dumps({"gaps": gaps[:20], "resolved_symbol": resolved}),
        )
        registry.audit(
            "engine1",
            "ingest_success",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps(asdict(result)),
        )
        logger.info(
            "ingest_done",
            symbol=symbol,
            timeframe=timeframe,
            mode=result.mode,
            rows_fetched=result.rows_fetched,
            rows_total=result.rows_total,
            gaps=result.gaps_detected,
        )
        return result

    except Exception as exc:
        result.error = str(exc)
        logger.exception("ingest_failed", symbol=symbol, timeframe=timeframe, error=str(exc))
        registry.upsert(
            symbol=symbol,
            timeframe=timeframe,
            layer=LAYER,
            last_run_status="error",
            row_count=0,
        )
        registry.audit(
            "engine1",
            "ingest_error",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps({"error": str(exc)}),
        )
        return result


def run_ingestion(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    force_rebuild: bool = False,
) -> IngestionRunReport:
    """Run Engine 1 for the given symbol/timeframe matrix."""
    ensure_project_dirs()
    set_global_seed()
    cfg = load_engine_config().get("engine1_ingestion", {})
    if symbols is None:
        symbols = list(cfg.get("default_symbols", ["XAUUSD"]))
    if timeframes is None:
        timeframes = list(cfg.get("default_timeframes", ["H1"]))
    if cfg.get("force_rebuild"):
        force_rebuild = True

    report = IngestionRunReport(
        started_at=_utc_now().isoformat(),
        force_rebuild=force_rebuild,
    )
    registry = DataStateRegistry()

    with mt5_session() as client:
        for symbol in symbols:
            for tf in timeframes:
                result = ingest_symbol_timeframe(
                    client,
                    registry,
                    symbol,
                    tf,
                    force_rebuild=force_rebuild,
                )
                report.results.append(result)

    report.finished_at = _utc_now().isoformat()
    report.status = "success" if not any(r.error for r in report.results) else "partial_error"

    reports_dir = PROJECT_ROOT / "logs" / "ingestion"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_path = reports_dir / f"ingestion_run_report_{stamp}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    # Also write latest pointer
    (reports_dir / "ingestion_run_report.json").write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info("ingestion_report_written", path=str(out_path), status=report.status)
    return report
