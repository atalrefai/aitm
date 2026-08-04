"""Persistent store for live trade outcomes — winners only for training.

Flow
----
1. ``register_open_trade`` — freeze full entry + pattern payload when a trade opens.
2. ``finalize_closed_trade`` — on exit, if net PnL > 0, append a complete record to
   ``data/training_trades/winning_trades.jsonl`` for Engine4.
3. ``reconcile_missing_tickets`` — catch SL/TP broker exits the API did not close.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from atis.config import PROJECT_ROOT, load_engine_config
from atis.shared.logging_utils import get_logger

logger = get_logger("atis.winning_trade_store")

_LOCK = threading.RLock()
_FINALIZED_TICKETS: set[int] = set()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_cfg() -> dict[str, Any]:
    cfg = load_engine_config().get("engine5_live", {}) or {}
    return dict(cfg.get("winning_trade_store") or {})


def store_enabled() -> bool:
    return bool(_store_cfg().get("enabled", True))


def winners_only() -> bool:
    return bool(_store_cfg().get("winners_only", True))


def include_losses() -> bool:
    """When true (and winners_only false), losing closes are stored for training."""
    cfg = _store_cfg()
    if bool(cfg.get("include_losses", False)):
        return True
    return not winners_only()


def keep_closed_for_training(record: dict[str, Any]) -> bool:
    """Decide whether a closed trade belongs in the training jsonl corpus."""
    cfg = _store_cfg()
    if winners_only() and not bool(cfg.get("include_losses", False)):
        return bool(record.get("is_winner"))
    # Mixed corpus: winners always; losses when include_losses / winners_only=false.
    if bool(record.get("is_winner")):
        return True
    if include_losses():
        return float(record.get("net_profit") or 0.0) < 0.0
    return False


def min_profit() -> float:
    try:
        return float(_store_cfg().get("min_profit", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def root_dir() -> Path:
    raw = str(_store_cfg().get("root") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
    else:
        p = PROJECT_ROOT / "data" / "training_trades"
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_trades_path() -> Path:
    return root_dir() / "open_trades.json"


def winning_trades_path() -> Path:
    return root_dir() / "winning_trades.jsonl"


def closed_audit_path() -> Path:
    """Optional audit of every close (winners + losers); not used for training."""
    return root_dir() / "closed_trades_audit.jsonl"


def _load_open_map() -> dict[str, dict[str, Any]]:
    path = open_trades_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _save_open_map(mapping: dict[str, dict[str, Any]]) -> None:
    path = open_trades_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(mapping, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _ticket_key(ticket: int | str | None) -> str | None:
    if ticket is None:
        return None
    try:
        return str(int(ticket))
    except (TypeError, ValueError):
        s = str(ticket).strip()
        return s or None


def register_open_trade(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Persist a full open-trade snapshot keyed by ticket (patterns + details)."""
    if not store_enabled():
        return None
    key = _ticket_key(payload.get("ticket"))
    if key is None:
        # Paper / dry-run: keep a synthetic key so the open book is complete.
        ts = str(payload.get("opened_at") or payload.get("ts") or _utc())
        key = f"paper-{ts}"
    record = {
        **payload,
        "ticket": int(key) if key.isdigit() else key,
        "ticket_key": key,
        "opened_at": payload.get("opened_at") or payload.get("ts") or _utc(),
        "status": "open",
        "registered_at": _utc(),
    }
    with _LOCK:
        mapping = _load_open_map()
        mapping[key] = record
        _save_open_map(mapping)
    logger.info(
        "open_trade_registered",
        ticket=key,
        symbol=record.get("symbol"),
        side=record.get("side"),
        patterns=len(record.get("pattern_keys") or []),
    )
    return record


def get_open_trade(ticket: int | str) -> dict[str, Any] | None:
    key = _ticket_key(ticket)
    if key is None:
        return None
    with _LOCK:
        return _load_open_map().get(key)


def pop_open_trade(ticket: int | str) -> dict[str, Any] | None:
    key = _ticket_key(ticket)
    if key is None:
        return None
    with _LOCK:
        mapping = _load_open_map()
        rec = mapping.pop(key, None)
        if rec is not None:
            _save_open_map(mapping)
        return rec


def _is_winner(net_profit: float) -> bool:
    return float(net_profit) > float(min_profit())


def _build_closed_record(
    open_rec: dict[str, Any] | None,
    *,
    ticket: int | str,
    net_profit: float,
    exit_price: float | None,
    close_reason: str,
    position_snapshot: dict[str, Any] | None = None,
    deal_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pos = position_snapshot or {}
    base = dict(open_rec or {})
    ticket_i: int | str
    try:
        ticket_i = int(ticket)
    except (TypeError, ValueError):
        ticket_i = ticket
    patterns = list(base.get("pattern_keys") or [])
    if not patterns and pos.get("comment"):
        # Fallback: comment may encode a short pattern tag.
        patterns = []
    return {
        "ticket": ticket_i,
        "symbol": base.get("symbol") or pos.get("symbol"),
        "side": base.get("side") or pos.get("side"),
        "volume": base.get("volume") or pos.get("volume"),
        "entry_price": base.get("entry_price") or pos.get("price_open"),
        "exit_price": exit_price if exit_price is not None else pos.get("price_current"),
        "sl": base.get("sl") or pos.get("sl"),
        "tp": base.get("tp") or pos.get("tp"),
        "confidence": base.get("confidence"),
        "timeframe": base.get("timeframe"),
        "mode": base.get("mode"),
        "pred": base.get("pred"),
        "reason": base.get("reason"),
        "opened_at": base.get("opened_at") or base.get("ts"),
        "closed_at": _utc(),
        "close_reason": close_reason,
        "net_profit": float(net_profit),
        "is_winner": _is_winner(net_profit),
        "pattern_keys": patterns,
        "pattern_ids": list(base.get("pattern_ids") or []),
        "pattern_summary": base.get("pattern_summary") or "",
        "pattern_explain": base.get("pattern_explain") or {},
        "exit_meta": base.get("exit_meta") or {},
        "prediction_debug": base.get("prediction_debug") or {},
        "feature_snapshot": base.get("feature_snapshot") or {},
        "multi_tf_context": base.get("multi_tf_context") or {},
        "position_snapshot": pos,
        "deal_meta": deal_meta or {},
        "source": "winning_trade_store",
        "for_training": False,
    }


def _feedback_pattern_kb(record: dict[str, Any]) -> None:
    """Append live-win events into pattern_events without wiping discovery rows."""
    if not bool(_store_cfg().get("update_pattern_kb", True)):
        return
    if not record.get("is_winner"):
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
        # insert_events deletes all events for symbol/tf — avoid that path.
        # Increment stats carefully by reading existing then writing back.
        existing = {
            str(r.get("pattern_key")): r
            for r in kb.list_stats(symbol=symbol, timeframe=timeframe, limit=5000)
        }
        rows: list[dict[str, Any]] = []
        net = float(record.get("net_profit") or 0.0)
        for key in keys:
            prev = existing.get(key) or {}
            occ = int(prev.get("occurrences") or 0) + 1
            evaluated = int(prev.get("evaluated") or 0) + 1
            successes = int(prev.get("successes") or 0) + 1
            prev_avg = float(prev.get("avg_forward_return") or 0.0)
            # Blend realized trade PnL (scaled) into avg_forward_return estimate.
            avg_fwd = ((prev_avg * max(evaluated - 1, 0)) + (net * 1e-4)) / max(evaluated, 1)
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
                    "quality_score": prev.get("quality_score"),
                    "approved": prev.get("approved"),
                    "std_dev": prev.get("std_dev"),
                    "strength": prev.get("strength"),
                }
            )
        if rows:
            kb.upsert_stats(rows)
    except Exception as exc:
        logger.warning("pattern_kb_live_feedback_failed", error=str(exc))


def _pnl_settlement_ok(
    net_profit: float,
    exit_price: float | None,
    deal_meta: dict[str, Any] | None,
) -> bool:
    """True when PnL is broker-confirmed (or a clear non-zero floating/API close)."""
    deal = deal_meta if isinstance(deal_meta, dict) else {}
    has_out = bool(deal.get("has_out"))
    try:
        exit_f = float(exit_price) if exit_price not in (None, "") else 0.0
    except (TypeError, ValueError):
        exit_f = 0.0
    if has_out and exit_f > 0:
        return True
    # Non-zero PnL from API close snapshot is usable even if history lags briefly.
    try:
        if abs(float(net_profit)) >= 1e-9 and exit_f > 0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def finalize_closed_trade(
    *,
    ticket: int | str,
    net_profit: float,
    exit_price: float | None = None,
    close_reason: str = "close",
    position_snapshot: dict[str, Any] | None = None,
    deal_meta: dict[str, Any] | None = None,
    open_fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record a closed trade; persist to training corpus only if it is a winner.

    Reinforcement learning scores every close (win or loss) when RL is enabled,
    even if the winning-trade JSONL corpus is disabled.

    Refuses to finalize when PnL is still unknown (net≈0 without OUT deal) so the
    watcher can retry — never invents a settled zero that mislabels winners.
    """
    key = _ticket_key(ticket)
    if key is None:
        return None
    try:
        ticket_i = int(key)
    except ValueError:
        ticket_i = None

    if not _pnl_settlement_ok(float(net_profit), exit_price, deal_meta):
        logger.info(
            "finalize_deferred_uncertain_pnl",
            ticket=key,
            net_profit=net_profit,
            exit_price=exit_price,
            has_out=bool((deal_meta or {}).get("has_out")),
            close_reason=close_reason,
        )
        return None

    store_on = store_enabled()
    with _LOCK:
        if ticket_i is not None and ticket_i in _FINALIZED_TICKETS:
            return None
        open_rec = pop_open_trade(key) if store_on else (get_open_trade(key) or open_fallback)
        if not store_on and open_rec is None:
            open_rec = open_fallback
        if store_on and open_rec is None:
            open_rec = open_fallback
        record = _build_closed_record(
            open_rec,
            ticket=ticket,
            net_profit=float(net_profit),
            exit_price=exit_price,
            close_reason=close_reason,
            position_snapshot=position_snapshot,
            deal_meta=deal_meta,
        )
        stored_for_training = False
        if store_on:
            if bool(_store_cfg().get("audit_all_closes", True)):
                _append_jsonl(closed_audit_path(), record)
            keep = keep_closed_for_training(record)
            if keep:
                training_row = {
                    **record,
                    "for_training": True,
                    "stored_at": _utc(),
                    "training_role": "winner" if record.get("is_winner") else "counter_example",
                }
                _append_jsonl(winning_trades_path(), training_row)
                stored_for_training = True
                record = training_row
            if ticket_i is not None:
                _FINALIZED_TICKETS.add(ticket_i)
                if len(_FINALIZED_TICKETS) > 5000:
                    _FINALIZED_TICKETS.clear()
        else:
            # Still dedupe RL scoring for vanished tickets in this process.
            if ticket_i is not None:
                _FINALIZED_TICKETS.add(ticket_i)
                if len(_FINALIZED_TICKETS) > 5000:
                    _FINALIZED_TICKETS.clear()

    if store_on and stored_for_training:
        if record.get("is_winner"):
            _feedback_pattern_kb(record)
        logger.info(
            "training_trade_stored",
            ticket=key,
            net_profit=record.get("net_profit"),
            role=record.get("training_role"),
            patterns=record.get("pattern_keys"),
            timeframe=record.get("timeframe"),
        )
    elif store_on:
        logger.info(
            "closed_trade_skipped_training",
            ticket=key,
            net_profit=record.get("net_profit"),
            reason="filtered_out",
        )

    # Online RL: score every closed trade (win or loss) → knowledge + training queue.
    try:
        from atis.shared.rl_learning import process_closed_trade

        rl_ep = process_closed_trade(record)
        if rl_ep is not None:
            record = {
                **record,
                "rl_episode_id": rl_ep.get("episode_id"),
                "rl_reward_total": rl_ep.get("reward_total"),
                "rl_reward_kind": rl_ep.get("reward_kind"),
                "rl_knowledge_status": rl_ep.get("knowledge_status"),
                "rl_lessons": rl_ep.get("lessons") or [],
                "rl_added_to_training_kb": bool(rl_ep.get("added_to_training_kb")),
            }
    except Exception as exc:
        logger.warning("rl_process_closed_trade_failed", ticket=key, error=str(exc))

    return record


def _fetch_deal_pnl_unlocked(ticket: int) -> dict[str, Any] | None:
    """Read deal history while an MT5 IPC connection is already held."""
    try:
        from atis.shared.mt5_client import _mt5_module

        mt5 = _mt5_module()
    except Exception as exc:
        logger.warning("mt5_module_unavailable", error=str(exc))
        return None

    now = datetime.now(timezone.utc)
    date_from = now - timedelta(days=30)
    date_to = now + timedelta(hours=1)

    deals = None
    try:
        deals = mt5.history_deals_get(position=int(ticket))
    except Exception as exc:
        logger.warning("history_deals_get_position_failed", ticket=ticket, error=str(exc))
        deals = None

    if not deals:
        try:
            deals = mt5.history_deals_get(date_from, date_to)
        except Exception as exc:
            logger.warning("history_deals_get_failed", error=str(exc))
            return None
        if deals:
            deals = [
                d
                for d in deals
                if int(getattr(d, "position_id", 0) or 0) == int(ticket)
            ]

    if not deals:
        return None

    profit = 0.0
    swap = 0.0
    commission = 0.0
    exit_price = None
    symbol = ""
    has_out = False
    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
    entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    entry_out_by = getattr(mt5, "DEAL_ENTRY_OUT_BY", 2)
    for d in deals:
        profit += float(getattr(d, "profit", 0) or 0)
        swap += float(getattr(d, "swap", 0) or 0)
        commission += float(getattr(d, "commission", 0) or 0)
        entry = int(getattr(d, "entry", -1))
        if entry in (entry_out, entry_out_by, 1, 2) and entry != entry_in:
            has_out = True
            px = float(getattr(d, "price", 0) or 0)
            if px > 0:
                exit_price = px
        symbol = str(getattr(d, "symbol", "") or symbol)

    # Opening deal alone appears in history before the close is booked — wait.
    if not has_out:
        logger.info(
            "deal_history_incomplete_no_out",
            ticket=ticket,
            deal_count=len(deals),
        )
        return None

    # OUT without a price is incomplete — do not invent a settled PnL.
    if exit_price is None or float(exit_price) <= 0:
        logger.info(
            "deal_history_incomplete_no_exit_price",
            ticket=ticket,
            deal_count=len(deals),
            profit=profit,
        )
        return None

    return {
        "ticket": int(ticket),
        "symbol": symbol,
        "net_profit": profit + swap + commission,
        "profit": profit,
        "swap": swap,
        "commission": commission,
        "exit_price": exit_price,
        "deal_count": len(deals),
        "has_out": True,
    }


def fetch_deal_pnl_for_position(ticket: int) -> dict[str, Any] | None:
    """Pull realized PnL for a closed position from MT5 deal history.

    Note: the MetaTrader5 Python package has no ``history_select`` (MQL-only).
    Use ``history_deals_get`` directly by position id, then by date window.

    Opens a short-lived MT5 session when the IPC bridge is down (e.g. repair
    jobs, or callers that finished their own ``mt5_session``).
    """
    try:
        from atis.shared.mt5_client import _ipc_healthy, mt5_session
    except Exception as exc:
        logger.warning("mt5_module_unavailable", error=str(exc))
        return None

    if _ipc_healthy():
        return _fetch_deal_pnl_unlocked(int(ticket))

    try:
        with mt5_session():
            return _fetch_deal_pnl_unlocked(int(ticket))
    except Exception as exc:
        logger.warning("fetch_deal_pnl_session_failed", ticket=ticket, error=str(exc))
        return None


def reconcile_missing_tickets(
    previous_tickets: set[int],
    current_tickets: set[int],
    *,
    close_reason: str = "broker_exit",
) -> list[dict[str, Any]]:
    """Finalize tickets that vanished from the open book (SL/TP/manual elsewhere).

    If MT5 deal history is not ready yet, the ticket is skipped (not finalized)
    so the watcher can retry on the next poll with real PnL.
    """
    try:
        from atis.shared.rl_learning.knowledge_store import enabled as rl_enabled
    except Exception:
        def rl_enabled() -> bool:  # type: ignore[misc]
            return False

    if not store_enabled() and not rl_enabled():
        return []
    missing = previous_tickets - current_tickets
    out: list[dict[str, Any]] = []
    for ticket in sorted(missing):
        if ticket in _FINALIZED_TICKETS:
            continue
        open_rec = get_open_trade(ticket)
        if open_rec is None:
            continue
        deal = fetch_deal_pnl_for_position(int(ticket))
        if deal is None:
            # History not ready — retry later; do NOT invent net_profit=0.
            logger.info("reconcile_waiting_deal_history", ticket=ticket)
            continue
        net = float(deal.get("net_profit") or 0.0)
        exit_price = deal.get("exit_price")
        rec = finalize_closed_trade(
            ticket=ticket,
            net_profit=net,
            exit_price=float(exit_price) if exit_price is not None else None,
            close_reason=close_reason,
            deal_meta=deal,
            open_fallback=open_rec,
        )
        if rec is not None:
            out.append(rec)
    return out


def load_winning_trades(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = winning_trades_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if symbol and str(row.get("symbol") or "") != str(symbol):
            continue
        if timeframe and str(row.get("timeframe") or "").upper() != str(timeframe).upper():
            continue
        rows.append(row)
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


def winning_trade_training_context(symbol: str, timeframe: str) -> dict[str, float]:
    """Aggregate scalars injected into Engine4 feature frames."""
    rows = load_winning_trades(symbol=symbol, timeframe=timeframe)
    if not rows:
        return {
            "live_winning_trades_count": 0.0,
            "live_winning_avg_confidence": 0.0,
            "live_winning_avg_profit": 0.0,
            "live_winning_pattern_diversity": 0.0,
            "live_winning_buy_ratio": 0.0,
        }
    confs = [float(r["confidence"]) for r in rows if r.get("confidence") is not None]
    profits = [float(r["net_profit"]) for r in rows if r.get("net_profit") is not None]
    keys: set[str] = set()
    buys = 0
    for r in rows:
        for k in r.get("pattern_keys") or []:
            keys.add(str(k))
        if str(r.get("side") or "").lower() == "buy":
            buys += 1
    n = float(len(rows))
    return {
        "live_winning_trades_count": n,
        "live_winning_avg_confidence": float(sum(confs) / max(len(confs), 1)) if confs else 0.0,
        "live_winning_avg_profit": float(sum(profits) / max(len(profits), 1)) if profits else 0.0,
        "live_winning_pattern_diversity": float(len(keys)),
        "live_winning_buy_ratio": buys / max(n, 1.0),
    }


def feature_snapshot_from_row(featured: Any, *, max_cols: int = 48) -> dict[str, Any]:
    """Compact last-bar feature dump for training replay (numeric-ish only)."""
    try:
        import pandas as pd

        if featured is None or len(featured) == 0:
            return {}
        row = featured.iloc[-1]
        out: dict[str, Any] = {}
        preferred = (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "atr",
            "rsi_14",
            "adx",
            "macd_hist",
            "trend_strength",
            "pat_bias",
            "pat_strength",
            "chart_pattern_score",
            "dist_to_support",
            "dist_to_resist",
        )
        cols = [c for c in preferred if c in featured.columns]
        for c in featured.columns:
            if c in cols:
                continue
            if len(cols) >= max_cols:
                break
            if str(c).startswith(("pat_", "NewN_", "cdl_", "struct_")):
                cols.append(c)
        for c in cols[:max_cols]:
            val = row[c]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            if isinstance(val, (pd.Timestamp, datetime)):
                val = str(val)
            elif hasattr(val, "isoformat"):
                try:
                    val = val.isoformat()
                except Exception:
                    val = str(val)
            out[str(c)] = val
        return out
    except Exception as exc:
        logger.warning("feature_snapshot_failed", error=str(exc))
        return {}
