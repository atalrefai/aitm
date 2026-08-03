"""Unit tests for MT5 pattern overlay (Explainable AI drawings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atis.shared.mt5_pattern_overlay.bridge import AsyncOverlayBridge, OverlaySnapshot, atomic_write_json
from atis.shared.mt5_pattern_overlay.detector import extract_active_overlays, top_signal_patterns
from atis.shared.mt5_pattern_overlay.history import append_history, read_history
from atis.shared.mt5_pattern_overlay.models import PatternStatus
from atis.shared.mt5_pattern_overlay.service import PatternOverlayService


def _fake_featured(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 2000 + np.cumsum(rng.normal(0, 1.5, size=n))
    high = close + rng.uniform(0.5, 3.0, size=n)
    low = close - rng.uniform(0.5, 3.0, size=n)
    open_ = close + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pat_strength": rng.uniform(0.2, 2.0, size=n),
            "chart_pattern_score": rng.normal(0, 1, size=n),
            "pat_hammer": np.zeros(n, dtype=int),
            "pat_bear_engulf": np.zeros(n, dtype=int),
            "pat_double_bottom": np.zeros(n, dtype=int),
            "pat_bos_up": np.zeros(n, dtype=int),
        }
    )
    df.loc[n - 1, "pat_hammer"] = 1
    df.loc[n - 3, "pat_double_bottom"] = 1
    df.loc[n - 2, "pat_bear_engulf"] = 1
    return df


def test_extract_active_overlays_builds_geometry():
    df = _fake_featured()
    overlays = extract_active_overlays(df, symbol="XAUUSD", timeframe="H1", lookback_bars=6)
    assert overlays
    keys = {o.key for o in overlays}
    assert "pat_hammer" in keys
    hammer = next(o for o in overlays if o.key == "pat_hammer")
    assert hammer.status in {PatternStatus.CONFIRMED, PatternStatus.FORMING}
    assert hammer.objects
    types = {o.type for o in hammer.objects}
    assert "arrow" in types
    assert "text" in types
    assert "Hammer" in hammer.label_text()
    assert "Confidence" in hammer.tooltip_text()


def test_top_signal_patterns_prefers_bias():
    df = _fake_featured()
    overlays = extract_active_overlays(df, symbol="XAUUSD", timeframe="H1")
    buys = top_signal_patterns(overlays, side="buy", limit=3)
    assert buys


def test_overlay_service_link_trade(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atis.shared.mt5_pattern_overlay.history.history_path",
        lambda: tmp_path / "patterns_history.jsonl",
    )
    monkeypatch.setattr(
        "atis.shared.mt5_pattern_overlay.bridge.local_overlay_dir",
        lambda: tmp_path / "overlay",
    )
    bridge = AsyncOverlayBridge(terminal_info_provider=lambda: None)
    service = PatternOverlayService(enabled=True, bridge=bridge, lookback_bars=8)
    df = _fake_featured()
    overlays = service.sync_from_features(df, symbol="XAUUSD", timeframe="H1", publish=True)
    assert overlays
    snap = OverlaySnapshot(
        seq=1,
        symbol="XAUUSD",
        broker_symbol="XAUUSD@",
        timeframe="H1",
        patterns=overlays,
    )
    paths = bridge.write_sync(snap)
    assert paths
    state = tmp_path / "overlay" / "overlay_state.json"
    assert state.exists()
    linked = service.link_trade(side="buy", ticket=12345, reason="pred=1;conf=0.7")
    assert linked
    assert all(p.status == PatternStatus.LINKED for p in linked)
    assert linked[0].trade.ticket == 12345
    hist = read_history(symbol="XAUUSD", timeframe="H1", limit=50)
    assert hist


def test_atomic_write_json(tmp_path: Path):
    path = tmp_path / "x.json"
    atomic_write_json(path, {"a": 1})
    assert path.read_text(encoding="utf-8")
    append_history([])
