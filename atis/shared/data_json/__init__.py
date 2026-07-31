"""Helpers to persist one timeframe DataFrame as a standalone JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def export_timeframe_json(
    df: pd.DataFrame,
    path: Path,
    *,
    symbol: str,
    timeframe: str,
    layer: str,
) -> Path:
    """Write one timeframe as a pretty JSON document with metadata + rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].astype(str)
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    if "timeframe" not in frame.columns:
        frame["timeframe"] = timeframe

    records: list[dict[str, Any]] = []
    cols = list(frame.columns)
    for row in frame.itertuples(index=False, name=None):
        records.append({cols[i]: _json_value(row[i]) for i in range(len(cols))})

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "layer": layer,
        "row_count": len(records),
        "columns": cols,
        "rows": records,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_timeframe_json(path: Path) -> pd.DataFrame:
    """Load a timeframe JSON file created by ``export_timeframe_json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Invalid timeframe JSON rows payload: {path}")
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
