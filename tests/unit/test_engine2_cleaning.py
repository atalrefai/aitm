"""Engine 2 unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from atis.engines.engine2_cleaning import clean_dataframe, flag_outliers, validate_ohlc


def _sample_h1(n: int = 48) -> pd.DataFrame:
    start = datetime(2024, 1, 2, 0, tzinfo=timezone.utc)  # Tuesday
    rows = []
    price = 1.10
    for i in range(n):
        ts = start + timedelta(hours=i)
        # skip weekend hours naturally by stopping before Sat if needed
        if ts.weekday() >= 5:
            continue
        o = price
        c = price + 0.0001
        rows.append(
            {
                "timestamp": ts,
                "open": o,
                "high": max(o, c) + 0.0002,
                "low": min(o, c) - 0.0002,
                "close": c,
                "tick_volume": 100,
                "spread": 1,
                "real_volume": 0,
                "symbol": "EURUSD",
                "timeframe": "H1",
            }
        )
        price = c
    return pd.DataFrame(rows)


def test_no_duplicate_timestamps() -> None:
    df = _sample_h1()
    cleaned, meta = clean_dataframe(df, "H1")
    assert cleaned["timestamp"].is_unique
    assert "is_imputed" in cleaned.columns
    assert meta.quality_score >= 0


def test_invalid_ohlc_flagged() -> None:
    df = _sample_h1(10)
    df.loc[3, "high"] = df.loc[3, "low"] - 0.01
    mask = validate_ohlc(df)
    assert bool(mask.iloc[3])


def test_outlier_flag_not_deleted() -> None:
    df = _sample_h1(30)
    df.loc[20, "close"] = df.loc[19, "close"] * 1.05  # huge spike
    df.loc[20, "high"] = max(df.loc[20, "high"], df.loc[20, "close"])
    df.loc[20, "low"] = min(df.loc[20, "low"], df.loc[20, "open"])
    cleaned, meta = clean_dataframe(df, "H1")
    assert "is_outlier" in cleaned.columns
    assert meta.outliers_flagged >= 1
    # row still present
    assert len(cleaned) >= 20
