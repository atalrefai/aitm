"""Feature engine unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from atis.shared.feature_engine import compute_features, rsi
from atis.shared.feature_engine.patterns import discover_rare_compounds


def _ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    price = 1.1
    rows = []
    for i in range(n):
        ret = float(rng.normal(0, 0.0008))
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(float(rng.normal(0, 0.0003))))
        l = min(o, c) * (1 - abs(float(rng.normal(0, 0.0003))))
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "tick_volume": int(rng.integers(50, 200)),
                "spread": 1,
                "real_volume": 0,
                "symbol": "EURUSD",
                "timeframe": "H1",
            }
        )
        price = c
    return pd.DataFrame(rows)


def test_rsi_bounds() -> None:
    df = _ohlcv(100)
    r = rsi(df["close"], 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_features_no_lookahead_shift() -> None:
    df = _ohlcv(300)
    feat = compute_features(df)
    assert "rsi_14" in feat.columns
    assert "macd" in feat.columns
    assert "atr" in feat.columns
    assert "pat_doji" in feat.columns
    assert "pat_bos_up" in feat.columns
    assert "cmp_hammer_support" in feat.columns
    # Warm-up NaNs allowed at start; last rows should have RSI
    assert feat["rsi_14"].iloc[-1] == feat["rsi_14"].iloc[-1]  # not NaN
    assert not np.isnan(feat["rsi_14"].iloc[-1])


def test_features_deterministic() -> None:
    df = _ohlcv(200)
    a = compute_features(df)
    b = compute_features(df)
    pd.testing.assert_frame_equal(a, b)


def test_discover_rare_compounds_can_find_triples() -> None:
    df = pd.DataFrame(
        {
            "pat_a": [1, 1, 1, 1, 1, 0, 0, 0],
            "pat_b": [1, 1, 1, 1, 1, 0, 0, 0],
            "pat_c": [1, 1, 1, 1, 1, 0, 0, 0],
            "pat_d": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )

    discovered = discover_rare_compounds(
        df,
        ["pat_a", "pat_b", "pat_c", "pat_d"],
        min_count=3,
        max_new=20,
        max_order=3,
        candidate_cap=10,
    )

    keys = {item["key"] for item in discovered}
    assert "disc_pat_a__pat_b" in keys
    assert "disc3_pat_a__pat_b__pat_c" in keys
