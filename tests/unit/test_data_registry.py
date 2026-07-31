"""Unit tests for data state registry (no MT5 required)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from atis.shared.data_registry import DataStateRegistry, compute_checksum


def test_registry_upsert_and_incremental(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    reg = DataStateRegistry(root)

    assert reg.get("EURUSD", "H1") is None
    assert reg.last_updated_ts("EURUSD", "H1") is None

    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2024, 6, 1, tzinfo=timezone.utc)

    reg.upsert(
        symbol="EURUSD",
        timeframe="H1",
        layer="raw",
        first_available_ts=ts1,
        last_updated_ts=ts2,
        row_count=100,
        checksum=compute_checksum(["a", "b"]),
        last_run_status="success",
    )

    row = reg.get("EURUSD", "H1", "raw")
    assert row is not None
    assert row.row_count == 100
    assert row.first_available_ts == ts1.isoformat()
    assert reg.last_updated_ts("EURUSD", "H1") == ts2

    # Per-timeframe JSON file exists and is isolated
    h1_path = root / "H1.json"
    assert h1_path.exists()
    payload = json.loads(h1_path.read_text(encoding="utf-8"))
    assert payload["timeframe"] == "H1"
    assert "EURUSD" in payload["symbols"]
    assert "raw" in payload["symbols"]["EURUSD"]

    # Keep earliest first_available; update last
    ts0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    ts3 = datetime(2024, 12, 1, tzinfo=timezone.utc)
    reg.upsert(
        symbol="EURUSD",
        timeframe="H1",
        layer="raw",
        first_available_ts=ts0,
        last_updated_ts=ts3,
        row_count=200,
        last_run_status="success",
    )
    row2 = reg.get("EURUSD", "H1")
    assert row2 is not None
    assert row2.row_count == 200
    assert row2.first_available_ts == ts0.isoformat()
    assert row2.last_updated_ts == ts3.isoformat()


def test_timeframes_are_separate_files(tmp_path: Path) -> None:
    reg = DataStateRegistry(tmp_path / "registry")
    reg.upsert(symbol="XAUUSD", timeframe="M5", layer="raw", row_count=10, last_run_status="success")
    reg.upsert(symbol="XAUUSD", timeframe="H1", layer="raw", row_count=20, last_run_status="success")

    assert (tmp_path / "registry" / "M5.json").exists()
    assert (tmp_path / "registry" / "H1.json").exists()
    assert reg.get("XAUUSD", "M5", "raw") is not None
    assert reg.get("XAUUSD", "H1", "raw") is not None
    assert reg.get("XAUUSD", "M5", "raw").row_count == 10
    assert reg.get("XAUUSD", "H1", "raw").row_count == 20


def test_audit_trail(tmp_path: Path) -> None:
    reg = DataStateRegistry(tmp_path / "registry")
    reg.audit("engine1", "ingestion_start", symbol="EURUSD", timeframe="H1")
    doc = json.loads((tmp_path / "registry" / "H1.json").read_text(encoding="utf-8"))
    assert len(doc["audit"]) == 1
    assert doc["audit"][0]["event"] == "ingestion_start"

    # Events without timeframe go to shared audit.json
    reg.audit("engine4", "training_done")
    audit = json.loads((tmp_path / "registry" / "audit.json").read_text(encoding="utf-8"))
    assert audit[-1]["event"] == "training_done"


def test_legacy_db_path_uses_parent_dir(tmp_path: Path) -> None:
    """Old callers/config that pass a .db path still resolve to the registry folder."""
    reg = DataStateRegistry(tmp_path / "registry" / "data_state_registry.db")
    reg.upsert(symbol="EURUSD", timeframe="M15", layer="clean", row_count=5)
    assert (tmp_path / "registry" / "M15.json").exists()


def test_prune_symbols_keeps_only_allowed(tmp_path: Path) -> None:
    reg = DataStateRegistry(tmp_path / "registry")
    reg.upsert(symbol="XAUUSD", timeframe="M5", layer="raw", row_count=10)
    reg.upsert(symbol="EURUSD", timeframe="M5", layer="raw", row_count=20)

    changed = reg.prune_symbols(["XAUUSD"])

    assert changed == 1
    doc = reg.load_timeframe_doc("M5")
    assert doc is not None
    assert sorted((doc.get("symbols") or {}).keys()) == ["XAUUSD"]
