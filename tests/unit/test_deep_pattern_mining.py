"""Unit tests for Deep Pattern Mining gaps: NewN, gates, relations, resume."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atis.shared.feature_engine.patterns import PATTERN_CATALOG, structural_patterns
from atis.shared.pattern_discovery.checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    mark_stage,
    save_checkpoint,
    stage_done,
)
from atis.shared.pattern_discovery.deep_miner import discover_deep_patterns
from atis.shared.pattern_discovery.relations import build_pattern_relations
from atis.shared.pattern_discovery.validation import gate_pattern
from atis.shared.pattern_store import SECTIONS, save_timeframe_pattern_bundle


REQUIRED_ADVANCED = [
    "pat_cup_handle",
    "pat_pennant_bull",
    "pat_pennant_bear",
    "pat_rectangle_bull",
    "pat_rectangle_bear",
    "pat_broadening_up",
    "pat_compression",
    "pat_harmonic_gartley_bull",
    "pat_harmonic_bat_bull",
    "pat_harmonic_butterfly_bull",
    "pat_harmonic_crab_bull",
    "pat_elliott_impulse_bull",
    "pat_wolfe_bull",
]


def _synth_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 0.8, size=n))
    open_ = close + rng.normal(0, 0.3, size=n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.5, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.5, size=n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 1000, size=n),
        }
    )


def test_catalog_has_advanced_patterns() -> None:
    for key in REQUIRED_ADVANCED:
        assert key in PATTERN_CATALOG
        meta = PATTERN_CATALOG[key]
        assert meta.get("name")
        assert meta.get("category") == "chart"
        assert meta.get("bias") in {"bullish", "bearish", "neutral"}
        assert meta.get("conditions")


def test_structural_detectors_emit_new_columns() -> None:
    df = _synth_ohlcv(500)
    out = structural_patterns(df)
    for key in REQUIRED_ADVANCED:
        assert key in out.columns
        assert out[key].dtype != object


def test_newn_naming_and_schema() -> None:
    df = _synth_ohlcv(500)
    # seed a few pat columns for sequential compounds
    df["pat_hammer"] = 0
    df["pat_bos_up"] = 0
    df.loc[50:60, "pat_hammer"] = 1
    df.loc[55:70, "pat_bos_up"] = 1
    found = discover_deep_patterns(df, max_new=10, min_count=5, existing_keys={"New1"})
    assert found
    assert all(item["name"].startswith("New") for item in found)
    assert "New1" not in {item["name"] for item in found}
    assert found[0]["name"] == "New2"
    for item in found:
        assert item["id"] == item["name"]
        assert "description" in item
        assert "mathematical_rules" in item
        assert "logical_rules" in item
        assert "appearance_conditions" in item
        assert item["signal"] is not None
        assert int(item["occurrences"]) >= 5


def test_validation_gate_rejects_weak_and_approves_strong() -> None:
    weak = np.array([0.001, -0.002, 0.0005, -0.001, 0.0002] * 3)
    strong = np.concatenate([np.full(40, 0.01), np.full(10, -0.002)])
    weak_res = gate_pattern(weak, bias="bullish")
    strong_res = gate_pattern(strong, bias="bullish")
    assert weak_res["approved"] is False
    assert weak_res["reject_reasons"]
    assert "metrics" in strong_res
    assert strong_res["metrics"]["profit_factor"] is not None
    assert strong_res["metrics"]["sharpe"] is not None
    assert strong_res["metrics"]["f1"] is not None
    assert strong_res["quality_score"] > weak_res["quality_score"]
    # Strong bullish edge should soft-promote even if hard gate fails on MC/CI
    assert strong_res.get("soft_promoted") is True


def test_neutral_success_not_always_one() -> None:
    from atis.shared.pattern_discovery.validation import success_mask

    rets = np.array([0.00001, -0.00002, 0.02, -0.015, 0.00003, 0.01, -0.00001, 0.012])
    wins = success_mask(rets, "neutral")
    assert 0 < int(wins.sum()) < len(rets)
    res = gate_pattern(rets, bias="neutral")
    assert res["metrics"]["success_rate"] < 1.0



def test_relations_graph_exports_edges() -> None:
    df = _synth_ohlcv(200)
    df["pat_a"] = 0
    df["pat_b"] = 0
    df["pat_c"] = 0
    df.loc[10:30, "pat_a"] = 1
    df.loc[12:32, "pat_b"] = 1
    df.loc[15:40, "pat_c"] = 1
    graph = build_pattern_relations(df, ["pat_a", "pat_b", "pat_c"], min_count=3)
    assert graph["nodes"]
    assert graph["edges"]
    assert any(e["relation"] == "co_occurrence" for e in graph["edges"])
    assert "summary" in graph


def test_checkpoint_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "atis.shared.pattern_discovery.checkpoint.get_path",
        lambda key: tmp_path / key,
    )
    monkeypatch.setattr(
        "atis.shared.pattern_discovery.checkpoint.PROJECT_ROOT",
        tmp_path,
    )
    # Force patterns root under tmp
    from atis.shared.pattern_discovery import checkpoint as ck

    def _cdir(symbol: str, timeframe: str):
        p = tmp_path / "patterns" / symbol / timeframe / "_checkpoint"
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(ck, "checkpoint_dir", _cdir)
    assert load_checkpoint("XAUUSD", "H1") is None
    state = mark_stage({}, "catalog_eval", patterns=3)
    save_checkpoint("XAUUSD", "H1", state)
    loaded = load_checkpoint("XAUUSD", "H1")
    assert loaded is not None
    assert stage_done(loaded, "catalog_eval")
    clear_checkpoint("XAUUSD", "H1")
    assert load_checkpoint("XAUUSD", "H1") is None


def test_pattern_store_writes_new_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "atis.shared.pattern_store.patterns_root",
        lambda: tmp_path / "patterns",
    )
    paths = save_timeframe_pattern_bundle(
        symbol="XAUUSD",
        timeframe="H1",
        stats=[
            {
                "symbol": "XAUUSD",
                "timeframe": "H1",
                "pattern_key": "pat_doji",
                "bias": "neutral",
                "occurrences": 12,
                "evaluated": 10,
                "successes": 6,
                "success_rate": 0.6,
                "avg_forward_return": 0.01,
                "confidence": 0.7,
                "quality_score": 0.55,
                "approved": True,
                "last_seen_ts": "2026-01-01T00:00:00+00:00",
                "conditions": "body_pct<0.1",
            }
        ],
        events=[],
        compounds=[],
        new_patterns=[
            {
                "key": "New1",
                "name": "New1",
                "description": "test",
                "mathematical_rules": "x",
                "logical_rules": "y",
                "appearance_conditions": "z",
                "occurrences": 20,
                "success_rate": 0.7,
                "approved": True,
                "soft_promoted": True,
                "quality_score": 0.8,
                "bias": "bullish",
            }
        ],
        relations={"nodes": [], "edges": [{"source": "a", "target": "b", "relation": "co_occurrence", "count": 5}], "sequences": [], "summary": "test"},
        bars_scanned=100,
    )
    assert set(paths.keys()) == set(SECTIONS)
    assert "new_patterns" in paths
    assert "relations" in paths
    assert "rankings" in paths
    assert "validation_report" in paths
    from atis.shared.pattern_store import load_section

    ranks = load_section("XAUUSD", "H1", "rankings")
    assert ranks is not None
    assert "engine4_recommended" in ranks
