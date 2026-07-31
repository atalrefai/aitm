"""Tests for per-section / per-timeframe pattern JSON storage."""

from __future__ import annotations

from atis.shared.pattern_store import (
    SECTIONS,
    load_section,
    save_timeframe_pattern_bundle,
)


def test_save_timeframe_pattern_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "atis.shared.pattern_store.patterns_root",
        lambda: tmp_path / "patterns",
    )

    paths = save_timeframe_pattern_bundle(
        symbol="XAUUSD",
        timeframe="M1",
        stats=[
            {
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "pattern_key": "pat_doji",
                "bias": "neutral",
                "occurrences": 12,
                "evaluated": 10,
                "successes": 6,
                "success_rate": 0.6,
                "avg_forward_return": 0.01,
                "confidence": 0.7,
                "last_seen_ts": "2026-01-01T00:00:00+00:00",
                "conditions": "body_pct<0.1",
            },
            {
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "pattern_key": "pat_bos_up",
                "bias": "bullish",
                "occurrences": 5,
                "evaluated": 5,
                "successes": 3,
                "success_rate": 0.6,
                "avg_forward_return": 0.02,
                "confidence": 0.65,
                "last_seen_ts": "2026-01-01T01:00:00+00:00",
                "conditions": "bos",
            },
        ],
        events=[
            {
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "pattern_key": "pat_doji",
                "ts": "2026-01-01T00:00:00+00:00",
                "close": 1900.0,
                "strength": 0.5,
                "forward_return": 0.01,
                "success": 1,
                "meta": {"horizon": 12, "bias": "neutral"},
            }
        ],
        compounds=[
            {
                "key": "disc_pat_doji__pat_bos_up",
                "name": "Doji + BOS",
                "legs": ["pat_doji", "pat_bos_up"],
                "lift": 2.1,
                "occurrences": 4,
                "success_rate": 0.75,
                "confidence": 0.8,
                "conditions": "co-occur",
                "bias": "bullish",
            }
        ],
        bars_scanned=1000,
    )

    assert set(paths.keys()) == set(SECTIONS)
    candles = load_section("XAUUSD", "M1", "candlesticks")
    structural = load_section("XAUUSD", "M1", "structural")
    compounds = load_section("XAUUSD", "M1", "compounds")
    knowledge = load_section("XAUUSD", "M1", "knowledge")
    discovery = load_section("XAUUSD", "M1", "discovery_log")

    assert candles and candles["count"] == 1
    assert candles["items"][0]["pattern_key"] == "pat_doji"
    assert structural and structural["count"] == 1
    assert structural["items"][0]["pattern_key"] == "pat_bos_up"
    assert compounds and compounds["count"] == 1
    assert knowledge and knowledge["count"] == 2
    assert discovery and discovery["count"] == 1
    assert (tmp_path / "patterns" / "XAUUSD" / "M1" / "candlesticks.json").exists()
