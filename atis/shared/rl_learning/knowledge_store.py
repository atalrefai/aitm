"""Persistent RL knowledge store — episodes, lessons, policy, training queue."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atis.config import PROJECT_ROOT, load_engine_config
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.rl_learning")

_LOCK = threading.RLock()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rl_cfg() -> dict[str, Any]:
    live = load_engine_config().get("engine5_live", {}) or {}
    return dict(live.get("reinforcement_learning") or {})


def enabled() -> bool:
    return bool(_rl_cfg().get("enabled", True))


def root_dir() -> Path:
    raw = str(_rl_cfg().get("root") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
    else:
        p = PROJECT_ROOT / "data" / "rl_knowledge"
    p.mkdir(parents=True, exist_ok=True)
    return p


def episodes_path() -> Path:
    return root_dir() / "episodes.jsonl"


def timeline_path() -> Path:
    return root_dir() / "learning_timeline.jsonl"


def state_path() -> Path:
    return root_dir() / "rl_state.json"


def training_queue_path() -> Path:
    return root_dir() / "training_queue.jsonl"


def lessons_path() -> Path:
    return root_dir() / "lessons.json"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


def _load_state() -> dict[str, Any]:
    path = state_path()
    if not path.exists():
        return {
            "version": 1,
            "created_at": _utc(),
            "updated_at": _utc(),
            "reward_count": 0,
            "penalty_count": 0,
            "neutral_count": 0,
            "episodes_total": 0,
            "knowledge_saved": 0,
            "knowledge_pending": 0,
            "knowledge_rejected": 0,
            "training_queued": 0,
            "training_consumed": 0,
            "policy_weights": {},
            "lesson_counts": {},
            "rolling": {
                "reward_ema": 0.0,
                "quality_ema": 0.5,
                "win_rate_ema": 0.5,
                "n": 0,
            },
            "performance_series": [],
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "version": 1,
            "created_at": _utc(),
            "updated_at": _utc(),
            "reward_count": 0,
            "penalty_count": 0,
            "corrupt_reload": True,
            "policy_weights": {},
            "lesson_counts": {},
            "rolling": {"reward_ema": 0.0, "quality_ema": 0.5, "win_rate_ema": 0.5, "n": 0},
            "performance_series": [],
        }


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _utc()
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_lessons() -> dict[str, Any]:
    path = lessons_path()
    if not path.exists():
        return {"version": 1, "items": {}, "updated_at": _utc()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"version": 1, "items": {}, "updated_at": _utc()}


def _save_lessons(doc: dict[str, Any]) -> None:
    doc["updated_at"] = _utc()
    path = lessons_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def decide_knowledge_status(
    reward_total: float,
    quality_score: float,
    *,
    cfg: dict[str, Any] | None = None,
    net_profit: float | None = None,
    reward_kind: str | None = None,
) -> tuple[str, str]:
    """Return (status, reason_ar).

    Statuses: saved | pending_review | rejected
    """
    cfg = cfg or _rl_cfg()
    save_abs = float(cfg.get("save_reward_abs_min", 0.12))
    reject_quality_max = float(cfg.get("reject_if_quality_below", 0.22))
    pending_abs = float(cfg.get("pending_reward_abs_max", 0.12))
    save_min_q = float(cfg.get("save_min_quality", 0.35))
    winner_save_min_q = float(cfg.get("winner_save_min_quality", 0.28))

    net = None if net_profit is None else float(net_profit)
    kind = str(reward_kind or "")

    # Profitable closes: prefer saving for training (outcome-primary).
    if net is not None and net > 0 and kind == "reward":
        if reward_total > 0 and quality_score >= winner_save_min_q:
            return "saved", "صفقة رابحة بمكافأة موجبة — حُفظت في قاعدة المعرفة للتدريب"
        if reward_total > 0:
            return "saved", "صفقة رابحة — حُفظت للتدريب (جودة دخول متوسطة)"
        if quality_score >= save_min_q:
            return "saved", "صفقة رابحة — حُفظت للتدريب مع مكافأة مخفّضة"
        return "pending_review", "رابحة لكن جودة الدخول ضعيفة جداً — قيد المراجعة"

    if quality_score < reject_quality_max and abs(reward_total) < 0.08:
        return "rejected", "جودة قرار منخفضة وتأثير ضعيف — رُفضت كمعرفة تدريب"
    if abs(reward_total) >= save_abs and quality_score >= save_min_q:
        return "saved", "معرفة عالية الإشارة — حُفظت في قاعدة المعرفة للتدريب"
    if abs(reward_total) < pending_abs:
        return "pending_review", "إشارة ضعيفة — قيد المراجعة قبل اعتمادها للتدريب"
    return "pending_review", "بانتظار تأكيد إضافي قبل الدمج في التدريب"


def _update_policy_weights(
    state: dict[str, Any],
    record: dict[str, Any],
    reward_total: float,
    *,
    lr: float,
) -> dict[str, float]:
    """Contextual-bandit style EMA update on sparse state keys."""
    weights = dict(state.get("policy_weights") or {})
    keys: list[str] = []
    tf = str(record.get("timeframe") or "NA").upper()
    side = str(record.get("side") or "na").lower()
    keys.append(f"tf:{tf}")
    keys.append(f"side:{side}")
    keys.append(f"tf_side:{tf}:{side}")
    conf = float(record.get("confidence") or 0.0)
    conf_bucket = "high" if conf >= 0.72 else "mid" if conf >= 0.58 else "low"
    keys.append(f"conf:{conf_bucket}")
    for pk in list(record.get("pattern_keys") or [])[:6]:
        keys.append(f"pat:{pk}")
    reason = str(record.get("reason") or "")
    if "soft_meta" in reason.lower():
        keys.append("flag:soft_meta")
    if "regime_soft" in reason.lower() or "soft_regime" in reason.lower():
        keys.append("flag:soft_regime")
    if "spread_filter_off" in reason.lower():
        keys.append("flag:spread_off")

    delta = float(lr) * float(reward_total)
    for k in keys:
        prev = float(weights.get(k, 0.0) or 0.0)
        # Soft update with decay toward 0 for stability.
        nxt = 0.97 * prev + delta
        weights[k] = round(max(-3.0, min(3.0, nxt)), 6)
    # Keep store bounded.
    if len(weights) > 400:
        ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:300]
        weights = {k: v for k, v in ranked}
    state["policy_weights"] = weights
    return weights


def load_policy_weights() -> dict[str, float]:
    """Public read of sparse RL policy weights for live gating / monitors."""
    state = _load_state()
    raw = state.get("policy_weights") or {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _merge_lessons(lessons: list[str], reward_total: float) -> None:
    doc = _load_lessons()
    items = dict(doc.get("items") or {})
    for lesson in lessons:
        key = str(lesson).strip()
        if not key:
            continue
        prev = dict(items.get(key) or {})
        count = int(prev.get("count") or 0) + 1
        avg_r = float(prev.get("avg_reward") or 0.0)
        avg_r = ((avg_r * (count - 1)) + float(reward_total)) / count
        items[key] = {
            "lesson": key,
            "count": count,
            "avg_reward": round(avg_r, 6),
            "last_seen": _utc(),
            "polarity": "positive" if avg_r >= 0 else "negative",
        }
    doc["items"] = items
    _save_lessons(doc)


class KnowledgeStore:
    """Facade used by service + API."""

    def persist_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        with _LOCK:
            _append_jsonl(episodes_path(), episode)
            _append_jsonl(
                timeline_path(),
                {
                    "ts": episode.get("evaluated_at") or _utc(),
                    "event": "episode_scored",
                    "ticket": episode.get("ticket"),
                    "kind": episode.get("reward_kind"),
                    "reward": episode.get("reward_total"),
                    "knowledge_status": episode.get("knowledge_status"),
                    "symbol": episode.get("symbol"),
                    "timeframe": episode.get("timeframe"),
                    "lessons": episode.get("lessons") or [],
                },
            )
            state = _load_state()
            state["episodes_total"] = int(state.get("episodes_total") or 0) + 1
            kind = str(episode.get("reward_kind") or "neutral")
            if kind == "reward":
                state["reward_count"] = int(state.get("reward_count") or 0) + 1
            elif kind == "penalty":
                state["penalty_count"] = int(state.get("penalty_count") or 0) + 1
            else:
                state["neutral_count"] = int(state.get("neutral_count") or 0) + 1

            status = str(episode.get("knowledge_status") or "pending_review")
            if status == "saved":
                state["knowledge_saved"] = int(state.get("knowledge_saved") or 0) + 1
            elif status == "rejected":
                state["knowledge_rejected"] = int(state.get("knowledge_rejected") or 0) + 1
            else:
                state["knowledge_pending"] = int(state.get("knowledge_pending") or 0) + 1

            cfg = _rl_cfg()
            lr = float(cfg.get("policy_learning_rate", 0.08))
            _update_policy_weights(
                state,
                episode.get("trade") or episode,
                float(episode.get("reward_total") or 0.0),
                lr=lr,
            )

            rolling = dict(state.get("rolling") or {})
            n = int(rolling.get("n") or 0) + 1
            alpha = float(cfg.get("ema_alpha", 0.12))
            r = float(episode.get("reward_total") or 0.0)
            q = float(episode.get("quality_score") or 0.5)
            win = 1.0 if bool(episode.get("is_winner")) else 0.0
            rolling["n"] = n
            rolling["reward_ema"] = round(
                (1 - alpha) * float(rolling.get("reward_ema") or 0.0) + alpha * r, 6
            )
            rolling["quality_ema"] = round(
                (1 - alpha) * float(rolling.get("quality_ema") or 0.5) + alpha * q, 6
            )
            rolling["win_rate_ema"] = round(
                (1 - alpha) * float(rolling.get("win_rate_ema") or 0.5) + alpha * win, 6
            )
            state["rolling"] = rolling

            series = list(state.get("performance_series") or [])
            series.append(
                {
                    "ts": episode.get("evaluated_at") or _utc(),
                    "reward_ema": rolling["reward_ema"],
                    "quality_ema": rolling["quality_ema"],
                    "win_rate_ema": rolling["win_rate_ema"],
                    "reward": r,
                    "n": n,
                }
            )
            state["performance_series"] = series[-200:]

            lessons = list(episode.get("lessons") or [])
            lesson_counts = dict(state.get("lesson_counts") or {})
            for lesson in lessons:
                lesson_counts[str(lesson)] = int(lesson_counts.get(str(lesson)) or 0) + 1
            state["lesson_counts"] = lesson_counts

            if status == "saved" and bool(cfg.get("queue_saved_for_training", True)):
                train_row = {
                    **episode,
                    "queued_at": _utc(),
                    "consumed": False,
                    "for_training": True,
                }
                _append_jsonl(training_queue_path(), train_row)
                state["training_queued"] = int(state.get("training_queued") or 0) + 1
                _append_jsonl(
                    timeline_path(),
                    {
                        "ts": _utc(),
                        "event": "knowledge_queued_for_training",
                        "ticket": episode.get("ticket"),
                        "reward": r,
                    },
                )

            _save_state(state)
            _merge_lessons(lessons, r)
            return episode


def _filter_episodes(
    rows: list[dict[str, Any]],
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    kind: str | None = None,
    knowledge_status: str | None = None,
    ticket: str | int | None = None,
) -> list[dict[str, Any]]:
    ticket_q = str(ticket).strip() if ticket is not None and str(ticket).strip() != "" else ""
    out: list[dict[str, Any]] = []
    for row in rows:
        if symbol and str(row.get("symbol") or "") != str(symbol):
            continue
        if timeframe and str(row.get("timeframe") or "").upper() != str(timeframe).upper():
            continue
        if kind and str(row.get("reward_kind") or "") != str(kind):
            continue
        if knowledge_status and str(row.get("knowledge_status") or "") != str(knowledge_status):
            continue
        if ticket_q:
            row_ticket = str(row.get("ticket") if row.get("ticket") is not None else "").strip()
            if ticket_q not in row_ticket:
                continue
        out.append(row)
    return out


def load_episodes(
    *,
    limit: int | None = 100,
    symbol: str | None = None,
    timeframe: str | None = None,
    kind: str | None = None,
    knowledge_status: str | None = None,
    ticket: str | int | None = None,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(episodes_path(), limit=None)
    out = _filter_episodes(
        rows,
        symbol=symbol,
        timeframe=timeframe,
        kind=kind,
        knowledge_status=knowledge_status,
        ticket=ticket,
    )
    if limit is not None and limit > 0:
        out = out[-limit:]
    return out


def query_episodes(
    *,
    page: int = 1,
    page_size: int = 10,
    symbol: str | None = None,
    timeframe: str | None = None,
    kind: str | None = None,
    knowledge_status: str | None = None,
    ticket: str | int | None = None,
) -> dict[str, Any]:
    """Newest-first paginated episode listing for the Learning Monitor UI."""
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 10), 100))
    filtered = _filter_episodes(
        _read_jsonl(episodes_path(), limit=None),
        symbol=symbol,
        timeframe=timeframe,
        kind=kind,
        knowledge_status=knowledge_status,
        ticket=ticket,
    )
    newest_first = list(reversed(filtered))
    total = len(newest_first)
    pages = max(1, (total + page_size - 1) // page_size) if total else 1
    if page > pages:
        page = pages
    start = (page - 1) * page_size
    slice_rows = newest_first[start : start + page_size]
    return {
        "episodes": slice_rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "ticket": str(ticket).strip() if ticket is not None else "",
    }


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)


def _recount_state_from_episodes(state: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    n_reward = n_penalty = n_neutral = 0
    n_saved = n_pending = n_rejected = 0
    for row in rows:
        kind = str(row.get("reward_kind") or "neutral")
        if kind == "reward":
            n_reward += 1
        elif kind == "penalty":
            n_penalty += 1
        else:
            n_neutral += 1
        st = str(row.get("knowledge_status") or "")
        if st == "saved":
            n_saved += 1
        elif st == "pending_review":
            n_pending += 1
        elif st == "rejected":
            n_rejected += 1
    state["reward_count"] = n_reward
    state["penalty_count"] = n_penalty
    state["neutral_count"] = n_neutral
    state["episodes_total"] = len(rows)
    state["knowledge_saved"] = n_saved
    state["knowledge_pending"] = n_pending
    state["knowledge_rejected"] = n_rejected


def delete_episodes(episode_ids: list[str]) -> dict[str, Any]:
    """Remove evaluated episodes by episode_id; also drop matching training-queue rows."""
    want = {str(x).strip() for x in (episode_ids or []) if str(x).strip()}
    if not want:
        return {"deleted": 0, "remaining": len(_read_jsonl(episodes_path())), "episode_ids": []}

    with _LOCK:
        rows = _read_jsonl(episodes_path())
        kept: list[dict[str, Any]] = []
        removed_ids: list[str] = []
        for row in rows:
            eid = str(row.get("episode_id") or "").strip()
            if eid and eid in want:
                removed_ids.append(eid)
            else:
                kept.append(row)

        if not removed_ids:
            return {"deleted": 0, "remaining": len(rows), "episode_ids": []}

        removed_set = set(removed_ids)
        _rewrite_jsonl(episodes_path(), kept)

        tq_rows = _read_jsonl(training_queue_path())
        tq_kept = [r for r in tq_rows if str(r.get("episode_id") or "").strip() not in removed_set]
        if len(tq_kept) != len(tq_rows):
            _rewrite_jsonl(training_queue_path(), tq_kept)

        state = _load_state()
        _recount_state_from_episodes(state, kept)
        pending = [r for r in tq_kept if not bool(r.get("consumed"))]
        state["training_queued"] = len(pending)
        state["updated_at"] = _utc()
        _save_state(state)
        _append_jsonl(
            timeline_path(),
            {
                "ts": _utc(),
                "event": "episodes_deleted",
                "count": len(removed_ids),
                "episode_ids": removed_ids[:50],
            },
        )
        logger.info("rl_episodes_deleted", deleted=len(removed_ids), remaining=len(kept))
        return {
            "deleted": len(removed_ids),
            "remaining": len(kept),
            "episode_ids": removed_ids,
        }


def episodes_pending_for_training(*, limit: int | None = None) -> list[dict[str, Any]]:
    rows = _read_jsonl(training_queue_path())
    pending = [r for r in rows if not bool(r.get("consumed"))]
    if limit and limit > 0:
        pending = pending[-limit:]
    return pending


def mark_training_consumed(episode_ids: list[str]) -> int:
    """Rewrite queue marking given episode_id values as consumed."""
    if not episode_ids:
        return 0
    want = {str(x) for x in episode_ids}
    path = training_queue_path()
    rows = _read_jsonl(path)
    if not rows:
        return 0
    n = 0
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("episode_id") or "") in want and not row.get("consumed"):
            row = {**row, "consumed": True, "consumed_at": _utc()}
            n += 1
        rewritten.append(row)
    with _LOCK:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rewritten:
                fh.write(json.dumps(row, default=str) + "\n")
        tmp.replace(path)
        state = _load_state()
        state["training_consumed"] = int(state.get("training_consumed") or 0) + n
        state["training_queued"] = max(0, int(state.get("training_queued") or 0) - n)
        _save_state(state)
        _append_jsonl(
            timeline_path(),
            {"ts": _utc(), "event": "training_consumed", "count": n},
        )
    return n


def rl_training_context(symbol: str, timeframe: str) -> dict[str, float]:
    """Scalar features injected into Engine4 frames from RL knowledge."""
    rows = [
        r
        for r in load_episodes(limit=500, symbol=symbol, timeframe=timeframe)
        if str(r.get("knowledge_status") or "") == "saved"
    ]
    pending = episodes_pending_for_training()
    pending_tf = [
        r
        for r in pending
        if (not symbol or str(r.get("symbol") or "") == str(symbol))
        and (
            not timeframe
            or str(r.get("timeframe") or "").upper() == str(timeframe).upper()
        )
    ]
    state = _load_state()
    rolling = state.get("rolling") or {}
    weights = state.get("policy_weights") or {}
    tf_w = float(weights.get(f"tf:{str(timeframe).upper()}", 0.0) or 0.0)
    if not rows and not pending_tf:
        return {
            "rl_episodes_saved": 0.0,
            "rl_avg_reward": 0.0,
            "rl_reward_ema": float(rolling.get("reward_ema") or 0.0),
            "rl_quality_ema": float(rolling.get("quality_ema") or 0.5),
            "rl_win_rate_ema": float(rolling.get("win_rate_ema") or 0.5),
            "rl_pending_training": float(len(pending_tf)),
            "rl_policy_tf_weight": tf_w,
            "rl_penalty_ratio": 0.0,
        }
    rewards = [float(r.get("reward_total") or 0.0) for r in rows]
    pens = sum(1 for r in rows if str(r.get("reward_kind")) == "penalty")
    return {
        "rl_episodes_saved": float(len(rows)),
        "rl_avg_reward": float(sum(rewards) / max(len(rewards), 1)),
        "rl_reward_ema": float(rolling.get("reward_ema") or 0.0),
        "rl_quality_ema": float(rolling.get("quality_ema") or 0.5),
        "rl_win_rate_ema": float(rolling.get("win_rate_ema") or 0.5),
        "rl_pending_training": float(len(pending_tf)),
        "rl_policy_tf_weight": tf_w,
        "rl_penalty_ratio": float(pens / max(len(rows), 1)),
    }


def _episode_pattern_keys(ep: dict[str, Any]) -> list[str]:
    keys = ep.get("pattern_keys") or []
    if not keys:
        trade = ep.get("trade") or {}
        if isinstance(trade, dict):
            keys = trade.get("pattern_keys") or []
    return [str(k) for k in keys if k]


def _pattern_reward_maps(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Per-pattern mean reward, mean |penalty|, and occurrence counts."""
    sums: dict[str, float] = {}
    pen_sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ep in episodes:
        reward = float(ep.get("reward_total") or 0.0)
        kind = str(ep.get("reward_kind") or "")
        for key in _episode_pattern_keys(ep):
            sums[key] = sums.get(key, 0.0) + reward
            counts[key] = counts.get(key, 0) + 1
            if kind == "penalty" or reward < 0:
                pen_sums[key] = pen_sums.get(key, 0.0) + abs(reward)
    avg = {k: sums[k] / max(counts[k], 1) for k in counts}
    pen_avg = {k: pen_sums.get(k, 0.0) / max(counts[k], 1) for k in counts}
    return avg, pen_avg, counts


def inject_rl_training_features(
    df: Any,
    symbol: str,
    timeframe: str,
    *,
    enabled: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Inject bar-varying RL features so Engine4 can learn from live rewards.

    Scalar ``rl_*`` context alone is constant across rows and gets dropped by
    ``drop_constant_features``. This function builds *interactions* between RL
    knowledge (episode rewards, policy weights, causal timeline) and bar-level
    pattern fires / structure — features that survive train/val/test selection.
    """
    import numpy as np
    import pandas as pd

    meta: dict[str, Any] = {
        "enabled": False,
        "n_episodes": 0,
        "n_pattern_keys": 0,
        "columns": [],
        "reason": None,
    }
    if df is None or getattr(df, "empty", True):
        meta["reason"] = "empty_frame"
        return df, meta

    cfg = _rl_cfg()
    e4 = load_engine_config().get("engine4_training", {}) or {}
    if enabled is None:
        enabled = bool(e4.get("rl_features_enabled", True)) and bool(cfg.get("enabled", True))
    if not enabled:
        meta["reason"] = "disabled"
        return df, meta
    if not bool(cfg.get("inject_bar_features", True)):
        meta["reason"] = "inject_bar_features_off"
        return df, meta

    work = df.copy()
    n = len(work)
    episodes = [
        r
        for r in load_episodes(limit=2000, symbol=symbol, timeframe=timeframe)
        if str(r.get("knowledge_status") or "") in {"saved", "pending_review"}
    ]
    pending = [
        r
        for r in episodes_pending_for_training()
        if str(r.get("symbol") or "") == str(symbol)
        and str(r.get("timeframe") or "").upper() == str(timeframe).upper()
    ]
    # Prefer saved/pending_review corpus; fall back to queue rows.
    corpus = episodes or pending
    meta["n_episodes"] = len(corpus)

    state = _load_state()
    policy = dict(state.get("policy_weights") or {})
    rolling = dict(state.get("rolling") or {})
    avg_reward, pen_avg, pat_counts = _pattern_reward_maps(corpus)
    meta["n_pattern_keys"] = len(pat_counts)

    # --- Pattern-fire × RL knowledge (varies when patterns fire) ---
    affinity = np.zeros(n, dtype=float)
    conflict = np.zeros(n, dtype=float)
    policy_hit = np.zeros(n, dtype=float)
    fire_count = np.zeros(n, dtype=float)
    matched_cols: list[str] = []

    for col in work.columns:
        col_s = str(col)
        if not (
            col_s.startswith("pat_")
            or col_s.startswith("cmp_")
            or col_s.startswith("New")
            or col_s.startswith("disc_")
        ):
            continue
        try:
            fire = pd.to_numeric(work[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        except Exception:
            continue
        if float(np.nanmax(np.abs(fire))) <= 0:
            continue
        # Match episode keys: exact, or strip common prefixes.
        key_candidates = [col_s, col_s.replace("pat_sig_", ""), col_s]
        reward_w = 0.0
        pen_w = 0.0
        pol_w = 0.0
        found = False
        for key in key_candidates:
            if key in avg_reward or f"pat:{key}" in policy or key in pat_counts:
                reward_w = float(avg_reward.get(key, 0.0))
                pen_w = float(pen_avg.get(key, 0.0))
                pol_w = float(policy.get(f"pat:{key}", policy.get(key, 0.0)) or 0.0)
                found = True
                break
        if not found:
            # Soft match: any episode key contained in column name or vice versa.
            for key, rw in avg_reward.items():
                if key and (key in col_s or col_s.endswith(key) or col_s in key):
                    reward_w = float(rw)
                    pen_w = float(pen_avg.get(key, 0.0))
                    pol_w = float(policy.get(f"pat:{key}", 0.0) or 0.0)
                    found = True
                    break
        if not found:
            continue
        matched_cols.append(col_s)
        affinity += fire * reward_w
        conflict += fire * pen_w
        policy_hit += fire * pol_w
        fire_count += (fire > 0).astype(float)

    work["feat_rl_pattern_affinity"] = affinity
    work["feat_rl_pattern_conflict"] = conflict
    work["feat_rl_policy_pattern"] = policy_hit
    work["feat_rl_known_fire_count"] = fire_count
    # Net edge: reward affinity minus conflict.
    work["feat_rl_net_edge"] = affinity - conflict

    # Side / TF policy pressure interacted with structure when available.
    buy_w = float(policy.get("side:buy", 0.0) or 0.0)
    sell_w = float(policy.get("side:sell", 0.0) or 0.0)
    tf_w = float(policy.get(f"tf:{str(timeframe).upper()}", 0.0) or 0.0)
    side_pressure = buy_w - sell_w
    if "pat_bias" in work.columns:
        bias = pd.to_numeric(work["pat_bias"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        work["feat_rl_side_x_bias"] = bias * side_pressure
    else:
        work["feat_rl_side_x_bias"] = np.zeros(n, dtype=float)
    if "trend_strength" in work.columns:
        trend = pd.to_numeric(work["trend_strength"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        work["feat_rl_tf_x_trend"] = trend * tf_w
    else:
        work["feat_rl_tf_x_trend"] = np.zeros(n, dtype=float)

    # --- Causal timeline: only episodes that closed *before* each bar ---
    reward_pressure = np.zeros(n, dtype=float)
    quality_pressure = np.zeros(n, dtype=float)
    win_rate_pressure = np.zeros(n, dtype=float)
    if bool(cfg.get("inject_causal_timeline", True)) and "timestamp" in work.columns and corpus:
        try:
            ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
            ep_rows: list[tuple[pd.Timestamp, float, float, float]] = []
            for ep in corpus:
                raw_ts = ep.get("evaluated_at") or ep.get("queued_at") or (ep.get("trade") or {}).get("closed_at")
                et = pd.to_datetime(raw_ts, utc=True, errors="coerce")
                if pd.isna(et):
                    continue
                r = float(ep.get("reward_total") or 0.0)
                q = float(ep.get("quality_score") or 0.5)
                win = 1.0 if ep.get("is_winner") or r > 0 else 0.0
                ep_rows.append((et, r, q, win))
            ep_rows.sort(key=lambda x: x[0])
            if ep_rows:
                ep_times = pd.DatetimeIndex([x[0] for x in ep_rows])
                # Causal expanding means mapped onto bars via searchsorted.
                rewards = np.array([x[1] for x in ep_rows], dtype=float)
                quals = np.array([x[2] for x in ep_rows], dtype=float)
                wins = np.array([x[3] for x in ep_rows], dtype=float)
                csum_r = np.cumsum(rewards)
                csum_q = np.cumsum(quals)
                csum_w = np.cumsum(wins)
                # For each bar, count of episodes with et <= bar_ts
                bar_ts = ts.fillna(pd.Timestamp("1970-01-01", tz="UTC"))
                idxs = ep_times.searchsorted(bar_ts, side="right")
                for i, k in enumerate(idxs):
                    if k <= 0:
                        continue
                    reward_pressure[i] = float(csum_r[k - 1] / k)
                    quality_pressure[i] = float(csum_q[k - 1] / k)
                    win_rate_pressure[i] = float(csum_w[k - 1] / k)
        except Exception as exc:
            meta["timeline_error"] = str(exc)

    # Seed with global EMA when no prior episodes yet for early bars.
    ema_r = float(rolling.get("reward_ema") or 0.0)
    ema_q = float(rolling.get("quality_ema") or 0.5)
    ema_w = float(rolling.get("win_rate_ema") or 0.5)
    if float(np.nanmax(np.abs(reward_pressure))) <= 0 and (ema_r or ema_q):
        # Still create mild bar variation via interaction with fire_count / affinity.
        reward_pressure = ema_r + 0.15 * affinity
        quality_pressure = ema_q + 0.05 * np.tanh(affinity)
        win_rate_pressure = ema_w + 0.05 * np.tanh(affinity - conflict)

    work["feat_rl_reward_pressure"] = reward_pressure
    work["feat_rl_quality_pressure"] = quality_pressure
    work["feat_rl_winrate_pressure"] = win_rate_pressure

    # Interact scalar context with bar-varying signals so constants become learnable.
    ctx = rl_training_context(symbol, timeframe)
    for name, val in ctx.items():
        work[name] = float(val)
    work["feat_rl_ema_x_affinity"] = float(ctx.get("rl_reward_ema") or 0.0) * affinity
    work["feat_rl_penalty_ratio_x_conflict"] = float(ctx.get("rl_penalty_ratio") or 0.0) * conflict
    work["feat_rl_pending_x_fire"] = float(ctx.get("rl_pending_training") or 0.0) * fire_count

    feat_cols = [c for c in work.columns if str(c).startswith("feat_rl_") or str(c).startswith("rl_")]
    meta["enabled"] = True
    meta["columns"] = feat_cols
    meta["matched_pattern_cols"] = matched_cols[:40]
    meta["n_matched_pattern_cols"] = len(matched_cols)
    meta["policy_tf_weight"] = tf_w
    meta["side_pressure"] = side_pressure
    return work, meta


def consume_rl_for_training(symbol: str, timeframe: str) -> dict[str, Any]:
    """Mark pending RL queue rows for this symbol/TF as consumed after a train run."""
    pending = episodes_pending_for_training()
    ids = [
        str(r.get("episode_id"))
        for r in pending
        if r.get("episode_id")
        and str(r.get("symbol") or "") == str(symbol)
        and str(r.get("timeframe") or "").upper() == str(timeframe).upper()
    ]
    n = mark_training_consumed(ids)
    return {
        "consumed": int(n),
        "episode_ids": ids,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def repair_episodes(*, force: bool = False) -> dict[str, Any]:
    """Re-score all stored episodes with outcome-primary reward logic.

    Fixes historical rows where winners were labeled penalty/neutral because of
    weak RR terms or missing exit prices. Idempotent via state flag unless force.
    """
    from atis.shared.rl_learning.rewards import score_closed_trade

    with _LOCK:
        state = _load_state()
        flag = "pnl_sign_v6_repaired"
        if state.get(flag) and not force:
            return {"repaired": 0, "skipped": True, "reason": "already_repaired"}

        rows = _read_jsonl(episodes_path())
        if not rows:
            state[flag] = True
            _save_state(state)
            return {"repaired": 0, "skipped": False}

        # Optional fallback PnL from winning-trade corpus when MT5 history is gone.
        win_by_ticket: dict[str, float] = {}
        try:
            from atis.shared.winning_trade_store import load_winning_trades

            for wr in load_winning_trades(limit=5000):
                t = wr.get("ticket")
                if t is None:
                    continue
                try:
                    win_by_ticket[str(int(t))] = float(wr.get("net_profit") or 0.0)
                except (TypeError, ValueError):
                    continue
        except Exception:
            win_by_ticket = {}

        cfg = _rl_cfg()
        reward_cfg = dict(cfg.get("reward") or {})
        live = load_engine_config().get("engine5_live", {}) or {}
        reward_cfg.setdefault(
            "min_confidence_quality", float(live.get("confidence_threshold", 0.52))
        )
        dyn = live.get("dynamic_exits") or {}
        reward_cfg.setdefault("min_planned_rr", float(dyn.get("min_rr", 1.15)))

        repaired: list[dict[str, Any]] = []
        changed = 0
        n_reward = n_penalty = n_neutral = 0
        n_saved = n_pending = n_rejected = 0
        train_rows: list[dict[str, Any]] = []
        lesson_items: dict[str, Any] = {}

        for ep in rows:
            trade = dict(ep.get("trade") or {})
            net = ep.get("net_profit", trade.get("net_profit"))
            try:
                net_f = float(net if net is not None else 0.0)
            except (TypeError, ValueError):
                net_f = 0.0

            exit_price = trade.get("exit_price")
            deal_meta = dict(ep.get("deal_meta") or trade.get("deal_meta") or {})
            ticket = ep.get("ticket", trade.get("ticket"))

            # Refresh zero/unknown PnL from MT5 deal history (fixes false neutrals).
            # Also refresh when exit_price is missing — those rows were almost always
            # finalized while the IPC bridge was disconnected.
            needs_refresh = abs(net_f) < 1e-12 or exit_price in (None, "", 0, 0.0)
            if needs_refresh and ticket is not None:
                try:
                    from atis.shared.winning_trade_store import fetch_deal_pnl_for_position

                    deal = fetch_deal_pnl_for_position(int(ticket))
                    if isinstance(deal, dict) and deal.get("has_out"):
                        net_f = float(deal.get("net_profit") or 0.0)
                        if deal.get("exit_price") is not None:
                            exit_price = deal.get("exit_price")
                        deal_meta = {**deal_meta, **deal, "refreshed_at": _utc()}
                except Exception as exc:
                    logger.warning(
                        "rl_repair_pnl_refresh_failed",
                        ticket=ticket,
                        error=str(exc),
                    )
                if abs(net_f) < 1e-12:
                    try:
                        alt = win_by_ticket.get(str(int(ticket)))
                    except (TypeError, ValueError):
                        alt = None
                    if alt is not None and abs(float(alt)) > 1e-12:
                        net_f = float(alt)
                        deal_meta = {**deal_meta, "pnl_source": "winning_trade_store"}

            is_winner = net_f > 0

            record = {
                "ticket": ticket,
                "symbol": ep.get("symbol", trade.get("symbol")),
                "timeframe": ep.get("timeframe", trade.get("timeframe")),
                "side": ep.get("side", trade.get("side")),
                "volume": trade.get("volume", 0.01),
                "entry_price": trade.get("entry_price"),
                "exit_price": exit_price,
                "sl": trade.get("sl"),
                "tp": trade.get("tp"),
                "confidence": ep.get("confidence", trade.get("confidence")),
                "net_profit": net_f,
                "is_winner": is_winner,
                "close_reason": ep.get("close_reason") or trade.get("close_reason") or "",
                "reason": trade.get("reason") or ep.get("reason") or "",
                "pattern_keys": list(ep.get("pattern_keys") or trade.get("pattern_keys") or []),
                "exit_meta": trade.get("exit_meta")
                or ep.get("exit_meta")
                or (
                    {"reward_risk": ep.get("planned_rr")}
                    if ep.get("planned_rr") is not None
                    else {}
                ),
                "multi_tf_context": trade.get("multi_tf_context") or {},
                "feature_snapshot": trade.get("feature_snapshot") or {},
                "deal_meta": deal_meta,
            }
            br = score_closed_trade(record, cfg=reward_cfg)
            status, status_reason = decide_knowledge_status(
                br.total,
                br.quality_score,
                cfg=cfg,
                net_profit=net_f,
                reward_kind=br.kind,
            )
            old_kind = str(ep.get("reward_kind") or "")
            old_status = str(ep.get("knowledge_status") or "")
            new_ep = {
                **ep,
                "net_profit": net_f,
                "is_winner": is_winner,
                "reward_total": br.total,
                "reward_kind": br.kind,
                "is_reward": br.is_reward,
                "reward_components": br.components,
                "reward_reasons": br.reasons,
                "lessons": br.lessons,
                "quality_score": br.quality_score,
                "process_score": br.process_score,
                "realized_rr": br.realized_rr,
                "planned_rr": br.planned_rr,
                "pnl_norm": br.pnl_norm,
                "impact_hint": br.impact_hint,
                "knowledge_status": status,
                "knowledge_status_reason": status_reason,
                "added_to_training_kb": status == "saved",
                "deal_meta": deal_meta,
                "repaired_at": _utc(),
                "repair_version": "pnl_sign_v6",
            }
            # Keep trade snapshot aligned with refreshed PnL.
            new_ep["trade"] = {
                **trade,
                "net_profit": net_f,
                "is_winner": is_winner,
                "exit_price": exit_price,
                "deal_meta": deal_meta,
            }
            if (
                old_kind != br.kind
                or bool(ep.get("is_winner")) != is_winner
                or old_status != status
                or abs(float(ep.get("net_profit") or 0) - net_f) > 1e-9
            ):
                changed += 1
            if br.kind == "reward":
                n_reward += 1
            elif br.kind == "penalty":
                n_penalty += 1
            else:
                n_neutral += 1
            if status == "saved":
                n_saved += 1
                if bool(cfg.get("queue_saved_for_training", True)):
                    train_rows.append(
                        {
                            **new_ep,
                            "queued_at": _utc(),
                            "consumed": False,
                            "for_training": True,
                        }
                    )
            elif status == "rejected":
                n_rejected += 1
            else:
                n_pending += 1

            for lesson in br.lessons:
                key = str(lesson)
                item = lesson_items.get(key) or {
                    "lesson": key,
                    "count": 0,
                    "sum_reward": 0.0,
                    "polarity": "positive" if br.total >= 0 else "negative",
                }
                item["count"] = int(item["count"]) + 1
                item["sum_reward"] = float(item["sum_reward"]) + float(br.total)
                item["avg_reward"] = item["sum_reward"] / max(item["count"], 1)
                item["polarity"] = "positive" if item["avg_reward"] >= 0 else "negative"
                lesson_items[key] = item

            repaired.append(new_ep)

        path = episodes_path()
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in repaired:
                fh.write(json.dumps(row, default=str) + "\n")
        tmp.replace(path)

        # Rebuild training queue from current saved episodes (drop stale wrong labels).
        tq = training_queue_path()
        tq_tmp = tq.with_suffix(".tmp")
        with tq_tmp.open("w", encoding="utf-8") as fh:
            for row in train_rows:
                fh.write(json.dumps(row, default=str) + "\n")
        tq_tmp.replace(tq)

        _save_lessons({"version": 1, "updated_at": _utc(), "items": lesson_items})

        state["reward_count"] = n_reward
        state["penalty_count"] = n_penalty
        state["neutral_count"] = n_neutral
        state["episodes_total"] = len(repaired)
        state["knowledge_saved"] = n_saved
        state["knowledge_pending"] = n_pending
        state["knowledge_rejected"] = n_rejected
        state["training_queued"] = len(train_rows)
        state["lesson_counts"] = {k: int(v.get("count") or 0) for k, v in lesson_items.items()}
        state[flag] = True
        state["pnl_sign_v4_repaired"] = True
        state["last_repair_at"] = _utc()
        state["last_repair_changed"] = changed
        _save_state(state)
        _append_jsonl(
            timeline_path(),
            {
                "ts": _utc(),
                "event": "episodes_repaired",
                "version": "pnl_sign_v5",
                "changed": changed,
                "total": len(repaired),
                "rewards": n_reward,
                "penalties": n_penalty,
                "neutral": n_neutral,
                "saved": n_saved,
                "training_queued": len(train_rows),
            },
        )
        logger.info(
            "rl_episodes_repaired",
            total=len(repaired),
            changed=changed,
            rewards=n_reward,
            penalties=n_penalty,
            saved=n_saved,
        )
        return {
            "repaired": len(repaired),
            "changed": changed,
            "rewards": n_reward,
            "penalties": n_penalty,
            "neutral": n_neutral,
            "saved": n_saved,
            "pending": n_pending,
            "training_queued": len(train_rows),
            "skipped": False,
        }


def get_monitor_snapshot(*, episode_limit: int = 40, timeline_limit: int = 60) -> dict[str, Any]:
    # One-shot repair so the Learning Monitor never keeps stale winner→penalty labels.
    try:
        repair_episodes(force=False)
    except Exception as exc:
        logger.warning("rl_auto_repair_failed", error=str(exc))

    state = _load_state()
    episodes = list(reversed(load_episodes(limit=episode_limit)))
    # Defensive display normalization for any residual inconsistency.
    for ep in episodes:
        try:
            net = float(ep.get("net_profit") or 0.0)
        except (TypeError, ValueError):
            net = 0.0
        if net > 0:
            ep["reward_kind"] = "reward"
            ep["is_reward"] = True
            ep["is_winner"] = True
            if float(ep.get("reward_total") or 0.0) <= 0:
                ep["reward_total"] = abs(float(ep.get("reward_total") or 0.08)) or 0.08
        elif net < 0:
            ep["reward_kind"] = "penalty"
            ep["is_reward"] = False
            ep["is_winner"] = False
        else:
            # Exact zero: never invent رابحة/خاسرة from stale reward_kind.
            ep["reward_kind"] = "neutral"
            ep["is_reward"] = False
            ep["is_winner"] = False

    timeline = list(reversed(_read_jsonl(timeline_path(), limit=timeline_limit)))
    lessons_doc = _load_lessons()
    lesson_items = list((lessons_doc.get("items") or {}).values())
    lesson_items.sort(key=lambda x: int(x.get("count") or 0), reverse=True)
    pending = episodes_pending_for_training()
    rolling = state.get("rolling") or {}
    series = list(state.get("performance_series") or [])[-40:]
    return {
        "enabled": enabled(),
        "updated_at": state.get("updated_at") or _utc(),
        "counts": {
            "rewards": int(state.get("reward_count") or 0),
            "penalties": int(state.get("penalty_count") or 0),
            "neutral": int(state.get("neutral_count") or 0),
            "episodes_total": int(state.get("episodes_total") or 0),
            "knowledge_saved": int(state.get("knowledge_saved") or 0),
            "knowledge_pending": int(state.get("knowledge_pending") or 0),
            "knowledge_rejected": int(state.get("knowledge_rejected") or 0),
            "training_queued": len(pending),
            "training_queued_lifetime": int(state.get("training_queued") or 0),
            "training_consumed": int(state.get("training_consumed") or 0),
        },
        "rolling": rolling,
        "performance_series": series,
        "top_lessons": lesson_items[:12],
        "policy_weights_top": sorted(
            [
                {"key": k, "weight": float(v)}
                for k, v in (state.get("policy_weights") or {}).items()
            ],
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )[:15],
        "episodes": episodes,
        "timeline": timeline,
        "pending_for_training": list(reversed(pending[-episode_limit:])),
        "paths": {
            "root": str(root_dir()),
            "episodes": str(episodes_path()),
            "training_queue": str(training_queue_path()),
            "state": str(state_path()),
        },
        "repair": {
            "version": "pnl_sign_v6",
            "done": bool(state.get("pnl_sign_v6_repaired")),
            "last_repair_at": state.get("last_repair_at"),
            "last_repair_changed": state.get("last_repair_changed"),
        },
    }
