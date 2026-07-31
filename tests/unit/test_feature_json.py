"""Unit tests for discovery bars JSONL export/load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from atis.shared.feature_json import (
    export_discovery_bars_jsonl,
    load_discovery_bars_jsonl,
)


def test_export_and_load_full_jsonl(tmp_path: Path, monkeypatch) -> None:
    import atis.shared.feature_json as fj

    monkeypatch.setattr(fj, "get_path", lambda key: tmp_path if key == "data_features" else tmp_path)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "pat_doji": [1, 0],
            "pat_hammer": [0, 1],
            "pat_strength": [0.2, 0.8],
        }
    )
    path = export_discovery_bars_jsonl(df, "XAUUSD", "M5")
    assert path.exists()
    assert path.name == "discovery_bars.jsonl"

    loaded = load_discovery_bars_jsonl("XAUUSD", "M5")
    assert loaded is not None
    assert len(loaded) == 2
    assert int(loaded.iloc[0]["pat_doji"]) == 1
    assert int(loaded.iloc[1]["pat_hammer"]) == 1
