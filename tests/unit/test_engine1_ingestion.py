"""Engine 1 unit tests (no live MT5 required)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from atis.engines.engine1_ingestion import detect_gaps, _upsert_parquet


def test_detect_gaps_ignores_weekend() -> None:
    base = datetime(2024, 1, 5, 12, tzinfo=timezone.utc)  # Friday
    # Friday + 1h, then Monday (weekend gap)
    ts = [
        base,
        base + timedelta(hours=1),
        base + timedelta(days=3),  # Monday-ish ~72h — weekend
    ]
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True)})
    gaps = detect_gaps(df, "H1")
    assert gaps == []


def test_detect_gaps_flags_intraday_hole() -> None:
    base = datetime(2024, 1, 3, 10, tzinfo=timezone.utc)
    ts = [base, base + timedelta(hours=1), base + timedelta(hours=5)]  # 4h hole on H1
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True)})
    gaps = detect_gaps(df, "H1")
    assert len(gaps) == 1


def test_upsert_parquet_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "EURUSD_H1.parquet"
    ts = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
        utc=True,
    )
    df1 = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 1.1],
            "high": [1.05, 1.15],
            "low": [0.99, 1.09],
            "close": [1.02, 1.12],
            "tick_volume": [10, 11],
            "spread": [1, 1],
            "real_volume": [0, 0],
            "symbol": ["EURUSD", "EURUSD"],
            "timeframe": ["H1", "H1"],
        }
    )
    m1 = _upsert_parquet(path, df1)
    assert len(m1) == 2

    # Same rows again — no duplicates
    m2 = _upsert_parquet(path, df1)
    assert len(m2) == 2

    # Update last bar close
    df3 = df1.copy()
    df3.loc[1, "close"] = 1.20
    m3 = _upsert_parquet(path, df3)
    assert len(m3) == 2
    assert float(m3.iloc[-1]["close"]) == 1.20
