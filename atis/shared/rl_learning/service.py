"""Orchestrate score → knowledge → optional pattern-KB feedback."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from atis.config import load_engine_config
from atis.shared.logging_utils import get_logger
from atis.shared.rl_learning.knowledge_store import (
    KnowledgeStore,
    _rl_cfg,
    decide_knowledge_status,
    enabled,
)
from atis.shared.rl_learning.rewards import score_closed_trade

logger = get_logger("atis.rl_learning.service")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _episode_id(ticket: Any, closed_at: Any) -> str:
    raw = f"{ticket}|{closed_at}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _feedback_pattern_kb_rl(record: dict[str, Any], reward_total: float) -> None:
    """Update pattern stats for wins *and* losses using RL outcome."""
    cfg = _rl_cfg()
    if not bool(cfg.get("update_pattern_kb", True)):
        return
    keys = [str(k) for k in (record.get("pattern_keys") or []) if k]
    if not keys:
        return
    symbol = str(record.get("symbol") or "")
    timeframe = str(record.get("timeframe") or "")
    if not symbol or not timeframe:
        return
    try:
        from atis.shared.pattern_kb import PatternKnowledgeBase

        kb = PatternKnowledgeBase()
        existing = {
            str(r.get("pattern_key")): r
            for r in kb.list_stats(symbol=symbol, timeframe=timeframe, limit=5000)
        }
        is_win = bool(record.get("is_winner"))
        net = float(record.get("net_profit") or 0.0)
        rows: list[dict[str, Any]] = []
        for key in keys:
            prev = existing.get(key) or {}
            occ = int(prev.get("occurrences") or 0) + 1
            evaluated = int(prev.get("evaluated") or 0) + 1
            successes = int(prev.get("successes") or 0) + (1 if is_win else 0)
            prev_avg = float(prev.get("avg_forward_return") or 0.0)
            avg_fwd = ((prev_avg * max(evaluated - 1, 0)) + (net * 1e-4)) / max(evaluated, 1)
            # Blend RL reward into quality_score proxy when present.
            prev_q = prev.get("quality_score")
            try:
                pq = float(prev_q) if prev_q is not None else 0.5
            except (TypeError, ValueError):
                pq = 0.5
            # Map reward [-1.5,1.5] → quality nudge.
            q_nudge = max(-0.15, min(0.15, float(reward_total) * 0.08))
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "pattern_key": key,
                    "occurrences": occ,
                    "evaluated": evaluated,
                    "successes": successes,
                    "success_rate": successes / max(evaluated, 1),
                    "avg_forward_return": avg_fwd,
                    "confidence": record.get("confidence") or prev.get("confidence"),
                    "last_seen_ts": record.get("closed_at") or _utc(),
                    "conditions": prev.get("conditions"),
                    "quality_score": max(0.0, min(1.0, pq + q_nudge)),
                    "approved": prev.get("approved"),
                    "std_dev": prev.get("std_dev"),
                    "strength": prev.get("strength"),
                }
            )
        if rows:
            kb.upsert_stats(rows)
    except Exception as exc:
        logger.warning("rl_pattern_kb_feedback_failed", error=str(exc))


def process_closed_trade(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Score a closed-trade record and persist RL knowledge.

    Safe to call multiple times for the same ticket — duplicate episode_ids
    are still appended (timeline audit) but callers should finalize once.
    """
    if not enabled():
        return None
    if not record or not isinstance(record, dict):
        return None

    try:
        net = float(record.get("net_profit") or 0.0)
    except (TypeError, ValueError):
        net = 0.0

    # If PnL is still zero/unknown, refresh once from broker deal history.
    exit_px = record.get("exit_price")
    deal = record.get("deal_meta") if isinstance(record.get("deal_meta"), dict) else {}
    try:
        exit_f = float(exit_px) if exit_px not in (None, "") else 0.0
    except (TypeError, ValueError):
        exit_f = 0.0
    needs_broker = abs(net) < 1e-12 or not bool(deal.get("has_out")) or exit_f <= 0
    if needs_broker and record.get("ticket") is not None:
        try:
            from atis.shared.winning_trade_store import fetch_deal_pnl_for_position

            fresh = fetch_deal_pnl_for_position(int(record["ticket"]))
            if isinstance(fresh, dict) and fresh.get("has_out"):
                net = float(fresh.get("net_profit") or 0.0)
                if fresh.get("exit_price") is not None:
                    exit_px = fresh.get("exit_price")
                    exit_f = float(exit_px)
                deal = {**deal, **fresh}
                record = {
                    **record,
                    "net_profit": net,
                    "exit_price": exit_px,
                    "deal_meta": deal,
                }
        except Exception as exc:
            logger.info(
                "rl_pnl_refresh_failed",
                ticket=record.get("ticket"),
                error=str(exc),
            )

    # Normalize outcome flags from realized PnL (source of truth).
    if net > 0:
        record = {**record, "net_profit": net, "is_winner": True}
    elif net < 0:
        record = {**record, "net_profit": net, "is_winner": False}
    else:
        record = {**record, "net_profit": net}

    # Skip inventing labels when broker PnL is still unknown / incomplete.
    close_reason = str(record.get("close_reason") or "").lower()
    has_out = bool(deal.get("has_out")) or exit_f > 0
    uncertain = abs(net) < 1e-12 and not has_out
    if uncertain:
        logger.info(
            "rl_skip_uncertain_pnl",
            ticket=record.get("ticket"),
            close_reason=record.get("close_reason"),
            deal_count=deal.get("deal_count"),
        )
        return None

    cfg = _rl_cfg()
    reward_cfg = dict(cfg.get("reward") or {})
    live = load_engine_config().get("engine5_live", {}) or {}
    reward_cfg.setdefault("min_confidence_quality", float(live.get("confidence_threshold", 0.58)))
    dyn = live.get("dynamic_exits") or {}
    reward_cfg.setdefault("min_planned_rr", float(dyn.get("min_rr", 1.50)))
    reward_cfg["soft_cap_good_quality_loss"] = bool(cfg.get("soft_cap_good_quality_loss", False))

    breakdown = score_closed_trade(record, cfg=reward_cfg)
    status, status_reason = decide_knowledge_status(
        breakdown.total,
        breakdown.quality_score,
        cfg=cfg,
        net_profit=net,
        reward_kind=breakdown.kind,
    )

    closed_at = record.get("closed_at") or _utc()
    eid = _episode_id(record.get("ticket"), closed_at)
    episode: dict[str, Any] = {
        "episode_id": eid,
        "evaluated_at": _utc(),
        "ticket": record.get("ticket"),
        "symbol": record.get("symbol"),
        "timeframe": record.get("timeframe"),
        "side": record.get("side"),
        "net_profit": record.get("net_profit"),
        "is_winner": bool(record.get("is_winner")) if net == 0 else (net > 0),
        "confidence": record.get("confidence"),
        "close_reason": record.get("close_reason"),
        "pattern_keys": list(record.get("pattern_keys") or []),
        "reward_total": breakdown.total,
        "reward_kind": breakdown.kind,
        "is_reward": breakdown.is_reward,
        "reward_components": breakdown.components,
        "reward_reasons": breakdown.reasons,
        "lessons": breakdown.lessons,
        "quality_score": breakdown.quality_score,
        "process_score": breakdown.process_score,
        "realized_rr": breakdown.realized_rr,
        "planned_rr": breakdown.planned_rr,
        "pnl_norm": breakdown.pnl_norm,
        "impact_hint": breakdown.impact_hint,
        "knowledge_status": status,
        "knowledge_status_reason": status_reason,
        "added_to_training_kb": status == "saved",
        "deal_meta": deal,
        "trade": {
            "ticket": record.get("ticket"),
            "symbol": record.get("symbol"),
            "timeframe": record.get("timeframe"),
            "side": record.get("side"),
            "volume": record.get("volume"),
            "entry_price": record.get("entry_price"),
            "exit_price": record.get("exit_price"),
            "sl": record.get("sl"),
            "tp": record.get("tp"),
            "confidence": record.get("confidence"),
            "reason": record.get("reason"),
            "close_reason": record.get("close_reason"),
            "pattern_keys": list(record.get("pattern_keys") or []),
            "net_profit": record.get("net_profit"),
            "is_winner": bool(record.get("is_winner")) if net == 0 else (net > 0),
            "exit_meta": record.get("exit_meta") or {},
            "multi_tf_context": record.get("multi_tf_context") or {},
            "feature_snapshot": record.get("feature_snapshot") or {},
            "deal_meta": deal,
        },
        "source": "live_reinforcement",
    }

    store = KnowledgeStore()
    store.persist_episode(episode)
    try:
        _feedback_pattern_kb_rl(record, breakdown.total)
    except Exception as exc:
        logger.warning("rl_kb_side_effect_failed", error=str(exc))

    logger.info(
        "rl_episode_processed",
        ticket=episode.get("ticket"),
        kind=breakdown.kind,
        reward=breakdown.total,
        knowledge=status,
    )
    return episode
