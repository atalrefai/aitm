"""Unit tests for online RL reward scoring and knowledge persistence."""

from __future__ import annotations

from pathlib import Path

from atis.shared.rl_learning.knowledge_store import (
    KnowledgeStore,
    decide_knowledge_status,
    get_monitor_snapshot,
    rl_training_context,
)
from atis.shared.rl_learning.rewards import score_closed_trade
from atis.shared.rl_learning.service import process_closed_trade


def _winning_trade(**overrides):
    base = {
        "ticket": 1001,
        "symbol": "XAUUSD",
        "side": "buy",
        "volume": 0.01,
        "entry_price": 4000.0,
        "exit_price": 4010.0,
        "sl": 3990.0,
        "tp": 4020.0,
        "confidence": 0.78,
        "timeframe": "M15",
        "net_profit": 12.5,
        "is_winner": True,
        "reason": "pred=1;tf=M15;exit=atr_confidence_fallback;rr=1.8",
        "pattern_keys": ["pat_bull_engulf", "pat_tweezer_bottom"],
        "exit_meta": {
            "reward_risk": 1.8,
            "meta": {"atr": 5.0, "min_rr": 1.5, "confidence": 0.78},
        },
        "closed_at": "2026-08-03T12:00:00+00:00",
    }
    base.update(overrides)
    return base


def _losing_soft_trade(**overrides):
    base = _winning_trade(
        ticket=1002,
        exit_price=3992.0,
        net_profit=-8.0,
        is_winner=False,
        confidence=0.55,
        reason="pred=1;e4_soft_meta_fallback;spread_filter_off;mode=independent;rr=1.15",
        exit_meta={"reward_risk": 1.15, "meta": {"atr": 5.0, "min_rr": 1.15}},
        closed_at="2026-08-03T12:05:00+00:00",
    )
    base.update(overrides)
    return base


def test_reward_for_quality_winner() -> None:
    br = score_closed_trade(_winning_trade())
    assert br.kind == "reward"
    assert br.total > 0
    assert br.quality_score >= 0.5
    assert any("رابحة" in r for r in br.reasons)


def test_penalty_for_soft_override_loser() -> None:
    br = score_closed_trade(_losing_soft_trade())
    assert br.kind == "penalty"
    assert br.total < 0
    assert br.components["violation"] > 0
    assert any("ميتا" in r or "سبريد" in r for r in br.reasons)


def test_lucky_win_not_fully_reinforced() -> None:
    br = score_closed_trade(
        _winning_trade(
            confidence=0.4,
            reason="pred=1;e4_soft_meta_fallback;spread_filter_off",
            exit_meta={"reward_risk": 1.1, "meta": {"atr": 5.0}},
        )
    )
    # Winners stay rewards; quality lessons explain the weak decision.
    assert br.kind == "reward"
    assert br.total > 0
    assert any("مخفّض" in x or "جودة" in x or "RR" in x for x in br.lessons)
    assert not any("رفض الإعدادات" in x for x in br.lessons)


def test_early_small_winner_is_reward_not_penalty() -> None:
    """Regression: ATIS winners close with tiny PnL + RR 1.15 was labeled penalty."""
    br = score_closed_trade(
        {
            "ticket": 90694375,
            "symbol": "XAUUSD",
            "side": "sell",
            "volume": 0.01,
            "entry_price": 4055.44,
            "exit_price": 4055.74,
            "sl": 4058.9152556292233,
            "tp": 4051.443456026393,
            "confidence": 0.6329787234042553,
            "timeframe": "M5",
            "net_profit": 0.13,
            "is_winner": True,
            "close_reason": "ATIS winners",
            "reason": "pred=-1;spread_filter_off;rr=1.15;mode=independent",
            "pattern_keys": ["pat_marubozu_bear"],
            "exit_meta": {"reward_risk": 1.15, "meta": {"atr": 5.0, "min_rr": 1.15}},
        }
    )
    assert br.kind == "reward"
    assert br.is_reward is True
    assert br.total > 0


def test_missing_exit_price_does_not_invent_rr() -> None:
    br = score_closed_trade(
        _winning_trade(
            exit_price=None,
            net_profit=5.0,
            is_winner=True,
            close_reason="broker_exit",
        )
    )
    assert br.realized_rr is None
    assert br.kind == "reward"


def test_knowledge_status_gates() -> None:
    assert decide_knowledge_status(0.4, 0.7)[0] == "saved"
    assert decide_knowledge_status(0.02, 0.1)[0] == "rejected"
    assert decide_knowledge_status(0.05, 0.5)[0] == "pending_review"


def test_winner_knowledge_status_saved() -> None:
    status, _ = decide_knowledge_status(
        0.15,
        0.34,
        net_profit=0.24,
        reward_kind="reward",
    )
    assert status == "saved"


def test_process_skips_uncertain_zero_pnl(tmp_path: Path, monkeypatch) -> None:
    import atis.shared.rl_learning.knowledge_store as ks
    import atis.shared.rl_learning.service as svc

    monkeypatch.setattr(ks, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(ks, "enabled", lambda: True)
    monkeypatch.setattr(svc, "enabled", lambda: True)
    monkeypatch.setattr(
        "atis.shared.winning_trade_store.fetch_deal_pnl_for_position",
        lambda *_a, **_k: None,
    )

    ep = process_closed_trade(
        _winning_trade(
            net_profit=0.0,
            is_winner=False,
            exit_price=None,
            close_reason="broker_exit",
            deal_meta={"deal_count": 1, "has_out": False},
        )
    )
    assert ep is None


def test_finalize_defers_uncertain_zero_pnl(monkeypatch) -> None:
    from atis.shared import winning_trade_store as wts

    monkeypatch.setattr(wts, "store_enabled", lambda: True)
    out = wts.finalize_closed_trade(
        ticket=999001,
        net_profit=0.0,
        exit_price=None,
        close_reason="broker_exit",
        deal_meta={},
        open_fallback={
            "ticket": 999001,
            "symbol": "XAUUSD",
            "side": "buy",
            "volume": 0.01,
            "entry_price": 4000.0,
            "timeframe": "M1",
        },
    )
    assert out is None


def test_process_refreshes_zero_pnl_from_broker(tmp_path: Path, monkeypatch) -> None:
    import atis.shared.rl_learning.knowledge_store as ks
    import atis.shared.rl_learning.service as svc

    monkeypatch.setattr(ks, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(ks, "enabled", lambda: True)
    monkeypatch.setattr(svc, "enabled", lambda: True)
    monkeypatch.setattr(svc, "_feedback_pattern_kb_rl", lambda *a, **k: None)
    monkeypatch.setattr(
        "atis.shared.winning_trade_store.fetch_deal_pnl_for_position",
        lambda *_a, **_k: {
            "net_profit": 2.15,
            "exit_price": 4056.75,
            "has_out": True,
            "deal_count": 2,
        },
    )

    ep = process_closed_trade(
        _winning_trade(
            ticket=90935798,
            net_profit=0.0,
            is_winner=False,
            exit_price=None,
            close_reason="broker_exit",
            deal_meta={},
        )
    )
    assert ep is not None
    assert ep["net_profit"] == 2.15
    assert ep["is_winner"] is True
    assert ep["reward_kind"] == "reward"


def test_process_closed_trade_persists(tmp_path: Path, monkeypatch) -> None:
    import atis.shared.rl_learning.knowledge_store as ks
    import atis.shared.rl_learning.service as svc

    monkeypatch.setattr(ks, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(ks, "enabled", lambda: True)
    monkeypatch.setattr(svc, "enabled", lambda: True)
    monkeypatch.setattr(svc, "_feedback_pattern_kb_rl", lambda *a, **k: None)

    ep = process_closed_trade(_winning_trade(ticket=4242))
    assert ep is not None
    assert ep["reward_kind"] in {"reward", "penalty", "neutral"}
    assert ep["knowledge_status"] in {"saved", "pending_review", "rejected"}

    snap = get_monitor_snapshot(episode_limit=10, timeline_limit=10)
    assert snap["counts"]["episodes_total"] >= 1
    assert len(snap["episodes"]) >= 1

    ctx = rl_training_context("XAUUSD", "M15")
    assert "rl_reward_ema" in ctx
    assert "rl_pending_training" in ctx


def test_knowledge_store_queues_saved(tmp_path: Path, monkeypatch) -> None:
    import atis.shared.rl_learning.knowledge_store as ks

    monkeypatch.setattr(ks, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(
        ks,
        "_rl_cfg",
        lambda: {
            "enabled": True,
            "queue_saved_for_training": True,
            "policy_learning_rate": 0.1,
            "ema_alpha": 0.2,
            "save_reward_abs_min": 0.01,
            "save_min_quality": 0.1,
        },
    )
    store = KnowledgeStore()
    ep = {
        "episode_id": "abc123",
        "ticket": 9,
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "side": "buy",
        "reward_total": 0.5,
        "reward_kind": "reward",
        "is_winner": True,
        "quality_score": 0.8,
        "knowledge_status": "saved",
        "lessons": ["تعزيز سلوك جيد"],
        "evaluated_at": "2026-08-03T12:00:00+00:00",
        "trade": {"timeframe": "H1", "side": "buy", "confidence": 0.8, "pattern_keys": []},
    }
    store.persist_episode(ep)
    pending = ks.episodes_pending_for_training()
    assert any(r.get("episode_id") == "abc123" for r in pending)


def test_inject_rl_training_features_bar_varying(tmp_path: Path, monkeypatch) -> None:
    import json

    import pandas as pd

    import atis.shared.rl_learning.knowledge_store as ks

    monkeypatch.setattr(ks, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(
        ks,
        "_rl_cfg",
        lambda: {
            "enabled": True,
            "inject_bar_features": True,
            "inject_causal_timeline": True,
            "queue_saved_for_training": True,
        },
    )
    # Seed episodes + policy state.
    ep = {
        "episode_id": "inj1",
        "ticket": 11,
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "side": "buy",
        "reward_total": 0.6,
        "reward_kind": "reward",
        "is_winner": True,
        "quality_score": 0.8,
        "knowledge_status": "saved",
        "pattern_keys": ["pat_bull_engulf"],
        "evaluated_at": "2026-08-03T10:00:00+00:00",
        "lessons": ["تعزيز"],
        "trade": {"pattern_keys": ["pat_bull_engulf"]},
    }
    (tmp_path / "episodes.jsonl").write_text(json.dumps(ep) + "\n", encoding="utf-8")
    (tmp_path / "rl_state.json").write_text(
        json.dumps(
            {
                "rolling": {"reward_ema": 0.2, "quality_ema": 0.7, "win_rate_ema": 0.6},
                "policy_weights": {"pat:pat_bull_engulf": 0.4, "side:buy": 0.2, "tf:M15": 0.1},
                "training_queued": 0,
                "training_consumed": 0,
            }
        ),
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-03", periods=6, freq="15min", tz="UTC"),
            "close": [100, 101, 102, 101, 103, 104],
            "pat_bull_engulf": [0, 1, 0, 1, 0, 1],
            "pat_bias": [0.0, 1.0, 0.0, 1.0, -1.0, 1.0],
            "trend_strength": [0.1, 0.4, 0.2, 0.5, 0.1, 0.6],
        }
    )
    out, meta = ks.inject_rl_training_features(df, "XAUUSD", "M15")
    assert meta.get("enabled") is True
    assert "feat_rl_pattern_affinity" in out.columns
    assert float(out["feat_rl_pattern_affinity"].std()) > 0
    assert "feat_rl_net_edge" in out.columns
    assert "rl_reward_ema" in out.columns
    # Affinity should be higher on bars where the rewarded pattern fires.
    fired = out.loc[out["pat_bull_engulf"] > 0, "feat_rl_pattern_affinity"].mean()
    quiet = out.loc[out["pat_bull_engulf"] <= 0, "feat_rl_pattern_affinity"].mean()
    assert fired > quiet


def test_consume_rl_for_training(tmp_path: Path, monkeypatch) -> None:
    import json

    import atis.shared.rl_learning.knowledge_store as ks

    monkeypatch.setattr(ks, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(ks, "_rl_cfg", lambda: {"enabled": True, "queue_saved_for_training": True})
    row = {
        "episode_id": "c1",
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "consumed": False,
        "queued_at": "2026-08-03T12:00:00+00:00",
    }
    (tmp_path / "training_queue.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "rl_state.json").write_text(
        json.dumps({"training_queued": 1, "training_consumed": 0}),
        encoding="utf-8",
    )
    res = ks.consume_rl_for_training("XAUUSD", "M15")
    assert res["consumed"] == 1
    pending = ks.episodes_pending_for_training()
    assert not any(r.get("episode_id") == "c1" for r in pending)
