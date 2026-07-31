"""Tests for the advanced LLModel training pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from atis.engines.engine4_training.deep_learning import HAS_TORCH, prepare_multitimeframe_sequences


def _feature_frame(timeframe: str, bars: int, minutes: int) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 1900.0
    for i in range(bars):
        ts = start + timedelta(minutes=i * minutes)
        drift = 0.001 if i % 7 else -0.0005
        price = price * (1.0 + drift)
        rows.append(
            {
                "timestamp": ts,
                "open": price * 0.999,
                "high": price * 1.001,
                "low": price * 0.998,
                "close": price,
                "tick_volume": 100 + i,
                "spread": 2,
                "real_volume": 0,
                "symbol": "XAUUSD",
                "timeframe": timeframe,
                "session": "europe" if i % 3 else "us",
                "vol_regime": "normal" if i % 4 else "violent",
                "pat_bias": 1.0 if i % 5 else -1.0,
                "pat_strength": 0.2 + (i % 9) * 0.05,
                "trend_strength": 0.1 + i * 0.001,
                "chart_pattern_score": 0.15,
                "structure_hh_hl": 1.0,
                "dist_to_support": 2.0,
                "dist_to_resist": 3.0,
                "rsi_14": 45 + (i % 10),
                "atr": 4.0 + (i % 6) * 0.1,
                "pat_doji": 1 if i % 11 == 0 else 0,
                "pat_bos_up": 1 if i % 13 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_prepare_multitimeframe_sequences(monkeypatch, tmp_path) -> None:
    import atis.engines.engine4_training.deep_learning as dl

    base = tmp_path / "features"
    for tf, minutes, bars in (("M5", 5, 180), ("H1", 60, 180)):
        path = base / "XAUUSD" / tf
        path.mkdir(parents=True, exist_ok=True)
        _feature_frame(tf, bars, minutes).to_parquet(path / "features.parquet", index=False)

    monkeypatch.setattr(
        dl,
        "features_parquet_path",
        lambda symbol, timeframe: base / symbol / timeframe / "features.parquet",
    )
    monkeypatch.setattr(
        dl.PatternKnowledgeBase,
        "list_stats",
        lambda self, symbol, timeframe, min_occurrences=1, limit=5000: [
            {
                "pattern_key": "pat_doji",
                "occurrences": 10,
                "success_rate": 0.6,
                "confidence": 0.7,
                "avg_forward_return": 0.01,
                "bias": "bullish",
            }
        ],
    )

    prepared = prepare_multitimeframe_sequences("XAUUSD", ["M5", "H1"])

    assert prepared.base_timeframe == "M5"
    assert prepared.timeframes == ["M5", "H1"]
    assert prepared.inputs["M5"].shape[1] == prepared.sequence_length
    assert prepared.inputs["H1"].shape[0] == prepared.context.shape[0]
    assert prepared.context.shape[1] == len(prepared.context_features)
    assert set(np.unique(prepared.labels)).issubset({0, 1, 2})
