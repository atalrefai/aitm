from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atis.shared.data_json import export_timeframe_json, load_timeframe_json


def test_export_timeframe_json_writes_metadata_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "XAUUSD_M5.json"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "close": [101.0, 102.0],
        }
    )

    export_timeframe_json(
        df,
        path,
        symbol="XAUUSD",
        timeframe="M5",
        layer="raw",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "XAUUSD"
    assert payload["timeframe"] == "M5"
    assert payload["layer"] == "raw"
    assert payload["row_count"] == 2
    assert payload["columns"][:3] == ["timestamp", "open", "close"]
    assert payload["rows"][0]["timestamp"].startswith("2024-01-01")
    assert payload["rows"][0]["symbol"] == "XAUUSD"
    assert payload["rows"][0]["timeframe"] == "M5"


def test_load_timeframe_json_reads_exported_rows(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "open": [100.0],
            "close": [101.5],
            "pat_bias": [1],
        }
    )

    export_timeframe_json(
        df,
        path,
        symbol="XAUUSD",
        timeframe="M5",
        layer="features",
    )

    loaded = load_timeframe_json(path)
    assert len(loaded) == 1
    assert float(loaded.iloc[0]["close"]) == 101.5
    assert int(loaded.iloc[0]["pat_bias"]) == 1
