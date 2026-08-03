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
from atis.shared.pattern_discovery.validation import DEFAULT_GATES, gate_pattern
from atis.shared.pattern_kb import PatternKnowledgeBase
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
    assert all(str(item["name"]).startswith("New") for item in found)
    assert "New1" not in {item["key"] for item in found}
    assert found[0]["key"] == "New2"
    assert found[0]["id"] == "New2"
    for item in found:
        assert item["id"] == item["key"]
        assert item["description"]
        assert item["mathematical_rules"]
        assert item["logical_rules"]
        assert item["appearance_conditions"]
        # Descriptive display name (NewN — motif)
        assert "—" in item["name"] or item["name"] == item["key"]
        assert item["signal"] is not None
        assert int(item["occurrences"]) >= 5


def test_newn_metadata_survives_store_and_kb_export(tmp_path, monkeypatch) -> None:
    """P0: description/rules/validation must not become null after save+KB export."""
    import json

    monkeypatch.setattr(
        "atis.shared.pattern_store.patterns_root",
        lambda: tmp_path / "patterns",
    )
    monkeypatch.setattr(
        "atis.shared.pattern_kb.get_path",
        lambda key: tmp_path / key if key != "data" else tmp_path,
    )
    # Point KB db under tmp
    from atis.shared import pattern_kb as pkb

    monkeypatch.setattr(pkb, "PROJECT_ROOT", tmp_path)

    validation = {
        "evaluated": 30,
        "approved": False,
        "soft_promoted": True,
        "reject_reasons": ["monte_carlo"],
        "metrics": {"success_rate": 0.6, "profit_factor": 1.2},
        "quality_score": 0.55,
    }
    new_item = {
        "key": "New9",
        "id": "New9",
        "name": "New9 — Hammer then strong bull",
        "description": "Hammer then strong bull",
        "mathematical_rules": "pat_hammer[t-1]=1 AND close>open",
        "logical_rules": "hammer then bull",
        "appearance_conditions": "window=2",
        "occurrences": 30,
        "success_rate": 0.6,
        "approved": False,
        "soft_promoted": True,
        "validation": validation,
        "bias": "bullish",
        "quality_score": 0.55,
        "confidence": 0.7,
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "conditions": "pat_hammer[t-1]=1 AND close>open",
        "lift": None,
    }
    paths = save_timeframe_pattern_bundle(
        symbol="XAUUSD",
        timeframe="H1",
        stats=[],
        events=[],
        compounds=[],
        new_patterns=[new_item],
        bars_scanned=100,
    )
    assert "new_patterns" in paths
    from atis.shared.pattern_store import load_section

    stored = load_section("XAUUSD", "H1", "new_patterns")
    assert stored and stored["items"]
    row = stored["items"][0]
    assert row["description"] == "Hammer then strong bull"
    assert row["mathematical_rules"]
    assert row["logical_rules"]
    assert row["appearance_conditions"]
    assert row["validation"] is not None
    assert row["validation"]["soft_promoted"] is True
    assert "Hammer" in row["name"]

    kb = PatternKnowledgeBase(db_path=tmp_path / "pattern_knowledge.db")
    kb.upsert_discovered([new_item])
    # Flood KB with lift-ranked compounds that previously crowded NewN out
    flood = [
        {
            "key": f"disc_flood_{i}",
            "name": f"flood{i}",
            "legs": ["pat_a", "pat_b"],
            "lift": 10.0 - i * 0.01,
            "occurrences": 50,
            "success_rate": 0.6,
            "confidence": 0.5,
            "conditions": "x",
            "symbol": "XAUUSD",
            "timeframe": "H1",
        }
        for i in range(80)
    ]
    kb.upsert_discovered(flood)
    # Old buggy path: list_discovered(40) by lift alone would miss New9
    listed = kb.list_new_patterns(symbol="XAUUSD", timeframe="H1", limit=40)
    assert any(d["compound_key"] == "New9" for d in listed)
    meta = json.loads(listed[0]["meta_json"] or "{}") if listed[0]["compound_key"] == "New9" else None
    if meta is None:
        hit = next(d for d in listed if d["compound_key"] == "New9")
        meta = json.loads(hit["meta_json"] or "{}")
    assert meta.get("description") == "Hammer then strong bull"
    assert meta.get("mathematical_rules")
    assert meta.get("validation") is not None
    assert meta.get("soft_promoted") is True


def test_meeting_and_counterattack_not_identical() -> None:
    from atis.shared.feature_engine.patterns import candlestick_patterns

    # Crafted bars: moderate meeting (quiet) vs strong gap counterattack
    n = 40
    close = np.full(n, 2000.0)
    open_ = np.full(n, 2000.0)
    high = np.full(n, 2005.0)
    low = np.full(n, 1995.0)
    # Bar i-1 bear moderate, bar i bull moderate, equal close, open near prior close → meeting
    open_[20] = 2002.0
    close[20] = 2000.0
    high[20] = 2002.5
    low[20] = 1999.5  # body=2, rng=3 → ~0.67 still high — widen range
    high[20] = 2003.5
    low[20] = 1998.5  # body=2, rng=5 → 0.4
    open_[21] = 1999.8  # near prior close (quiet)
    close[21] = 2000.0  # bull + equal close
    high[21] = 2002.0
    low[21] = 1998.0  # body=0.2, rng=4 → too small body_pct
    # Raise body while staying moderate
    open_[21] = 1998.5
    close[21] = 2000.0
    high[21] = 2001.5
    low[21] = 1997.5  # body=1.5, rng=4 → 0.375
    # Bar 30-31: strong bodies + gap-down open → counterattack not meeting
    open_[30] = 2010.0
    close[30] = 1995.0
    high[30] = 2011.0
    low[30] = 1994.0
    open_[31] = 1993.0  # gap below prior close
    close[31] = 1995.0  # equal close
    high[31] = 1996.0
    low[31] = 1992.0
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 100),
        }
    )
    out = candlestick_patterns(df)
    meet = out["pat_meeting_bull"].to_numpy()
    counter = out["pat_counterattack_bull"].to_numpy()
    assert not np.array_equal(meet, counter)
    assert PATTERN_CATALOG["pat_meeting_bull"]["conditions"] != PATTERN_CATALOG["pat_counterattack_bull"]["conditions"]
    # Meeting fires without counterattack on quiet equal-close
    assert int(meet[21]) == 1
    assert int(counter[21]) == 0
    # Counterattack fires on strong + gap
    assert int(counter[31]) == 1
    assert int(meet[31]) == 0


def test_hammer_family_requires_trend_context() -> None:
    from atis.shared.feature_engine.patterns import candlestick_patterns

    n = 80
    # Strong uptrend then hanging-man shape
    close = np.concatenate([np.linspace(1900, 2000, 60), np.full(20, 2000.0)])
    open_ = close.copy()
    open_[-1] = 2005.0
    close[-1] = 2002.0  # small body
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    # Long lower wick on last bar (hammer/hanging shape)
    low[-1] = 1990.0
    high[-1] = 2005.5
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 100),
        }
    )
    out = candlestick_patterns(df)
    # In uptrend: hanging man yes, hammer no
    assert int(out["pat_hanging_man"].iloc[-1]) == 1
    assert int(out["pat_hammer"].iloc[-1]) == 0

    # Downtrend then hammer shape
    close2 = np.concatenate([np.linspace(2000, 1900, 60), np.full(20, 1900.0)])
    open2 = close2.copy()
    open2[-1] = 1902.0
    close2[-1] = 1904.0
    high2 = np.maximum(open2, close2) + 0.5
    low2 = np.minimum(open2, close2) - 0.5
    low2[-1] = 1890.0
    high2[-1] = 1904.5
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").astype(str),
            "open": open2,
            "high": high2,
            "low": low2,
            "close": close2,
            "volume": np.full(n, 100),
        }
    )
    out2 = candlestick_patterns(df2)
    assert int(out2["pat_hammer"].iloc[-1]) == 1
    assert int(out2["pat_hanging_man"].iloc[-1]) == 0


def test_tf_gates_and_rare_reject() -> None:
    from atis.shared.pattern_discovery.validation import gates_for_timeframe, gate_pattern

    m1 = gates_for_timeframe("M1")
    h1 = gates_for_timeframe("H1")
    assert m1["min_evaluated"] > DEFAULT_GATES["min_evaluated"]
    assert h1["min_success_rate"] >= 0.56
    # Too few samples → rare reject, no soft promote
    tiny = np.full(5, 0.02)
    res = gate_pattern(tiny, bias="bullish", timeframe="H1")
    assert res["rare_rejected"] is True
    assert res["soft_promoted"] is False
    assert "rare_pattern" in res["reject_reasons"]


def test_htf_confirm_blocks_mismatch() -> None:
    from atis.shared.pattern_discovery.validation import confirm_htf_bias

    ok = confirm_htf_bias(
        bias="bullish",
        htf_bias_values=np.array([1.0, 2.0, 1.0]),
        htf_chart_scores=None,
    )
    assert ok["confirmed"] is True
    bad = confirm_htf_bias(
        bias="bullish",
        htf_bias_values=np.array([-2.0, -1.0, -1.0]),
        htf_chart_scores=None,
    )
    assert bad["confirmed"] is False
    skip = confirm_htf_bias(bias="bullish", htf_bias_values=None, htf_chart_scores=None)
    assert skip["confirmed"] is None


def test_chart_pattern_score_includes_harmonics_and_weights() -> None:
    from atis.shared.feature_engine.patterns import (
        chart_score_weights_from_rankings,
        structural_patterns,
    )

    df = _synth_ohlcv(500)
    out = structural_patterns(df)
    assert "chart_pattern_score" in out.columns
    # Weights from rankings should vary from default 1.0 for known keys
    weights = chart_score_weights_from_rankings(
        [
            {
                "pattern_key": "pat_harmonic_bat_bull",
                "quality_score": 0.9,
                "success_rate": 0.7,
            }
        ]
    )
    assert weights["pat_harmonic_bat_bull"] > 1.0
    assert weights["pat_harmonic_butterfly_bull"] == 1.0
    out_w = structural_patterns(df, score_weights=weights)
    assert "chart_pattern_score" in out_w.columns


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
    assert "counts" in graph
    assert graph["counts"]["co_occurrence"] >= 1
    assert all("relation_ar" in e for e in graph["edges"])
    assert all("label" in n for n in graph["nodes"])


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
