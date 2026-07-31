"""Comprehensive Deep Pattern Mining with progress, cancel, and resume.

Discovery reads the full per-timeframe JSONL bar matrix
(``data/features/{symbol}/{tf}/discovery_bars.jsonl``), evaluates every
pattern column, mines NewN motifs, validates statistically, builds relation
graphs, updates the knowledge base, and exports section JSON files under
``data/patterns/{symbol}/{tf}/``.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from atis.config import get_path, load_engine_config, load_timeframes
from atis.engines.engine3_features import run_features
from atis.shared.data_json import load_timeframe_json
from atis.shared.feature_engine.patterns import (
    PATTERN_CATALOG,
    discover_rare_compounds,
    pattern_category_map,
    pattern_labels,
)
from atis.shared.feature_json import ensure_discovery_bars_jsonl
from atis.shared.logging_utils import get_logger
from atis.shared.pattern_discovery.checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    mark_stage,
    save_checkpoint,
    stage_done,
)
from atis.shared.pattern_discovery.deep_miner import attach_signals_to_frame, discover_deep_patterns
from atis.shared.pattern_discovery.relations import build_pattern_relations
from atis.shared.pattern_discovery.validation import gate_pattern
from atis.shared.pattern_kb import PatternKnowledgeBase
from atis.shared.pattern_store import save_timeframe_pattern_bundle, write_discovery_report

logger = get_logger("atis.pattern_discovery")

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], None]
DetailsFn = Callable[[dict[str, Any]], None]

MAX_EVENTS_PER_PATTERN = 5000
RARE_MIN_COUNT = 5
RARE_MAX_NEW = 120
DEEP_MAX_NEW = 40
DEEP_MIN_COUNT = 8


def _noop_progress(_pct: float, _msg: str) -> None:
    return None


def _noop_cancel() -> None:
    return None


def _horizon_for_tf(tf: str) -> int:
    mapping = {
        "M1": 12,
        "M5": 6,
        "M15": 4,
        "M30": 4,
        "H1": 3,
        "H4": 2,
        "D1": 2,
        "W1": 1,
        "MN1": 1,
    }
    return int(mapping.get(tf, 4))


def _evaluate_pattern_vectorized(
    df: pd.DataFrame,
    col: str,
    *,
    bias: str,
    horizon: int,
    max_events: int = MAX_EVENTS_PER_PATTERN,
    run_gates: bool = True,
) -> dict[str, Any]:
    """Vectorized evaluation + optional statistical validation gates."""
    empty = {
        "occurrences": 0,
        "evaluated": 0,
        "successes": 0,
        "success_rate": None,
        "avg_forward_return": None,
        "std_dev": None,
        "risk_ratio": None,
        "expectancy": None,
        "profit_factor": None,
        "sharpe": None,
        "max_drawdown": None,
        "win_rate": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "quality_score": 0.0,
        "strength": 0.0,
        "confidence": 0.0,
        "approved": False,
        "validation": None,
        "last_seen_ts": None,
        "events": [],
        "forward_returns": np.array([]),
    }
    if col not in df.columns:
        return empty

    mask = df[col].fillna(0).astype(int).to_numpy() == 1
    occ = int(mask.sum())
    if occ == 0:
        return empty

    close = df["close"].astype(float).to_numpy()
    n = len(close)
    fwd = np.full(n, np.nan)
    if horizon > 0 and n > horizon:
        fwd[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0

    hit_idx = np.flatnonzero(mask)
    fr = fwd[hit_idx]
    valid = ~np.isnan(fr)
    evaluated = int(valid.sum())
    rets = fr[valid]
    if evaluated:
        from atis.shared.pattern_discovery.validation import success_mask, signed_returns

        wins_m = success_mask(rets, bias)
        successes = int(wins_m.sum())
        success_rate = successes / evaluated
        avg_ret = float(np.mean(rets))
        std_dev = float(np.std(rets)) if evaluated > 1 else 0.0
        edge = abs(success_rate - 0.5)
        conf = min(0.99, 0.35 + 0.4 * edge + 0.25 * min(1.0, evaluated / 40))
        signed = signed_returns(rets, bias)
        wins = signed[signed > 0]
        losses = -signed[signed < 0]
        risk_ratio = float(wins.mean() / max(losses.mean(), 1e-12)) if len(wins) and len(losses) else None
    else:
        successes = 0
        success_rate = None
        avg_ret = None
        std_dev = None
        risk_ratio = None
        conf = min(0.5, occ / 100)

    validation = None
    approved = False
    soft_promoted = False
    quality = 0.0
    metrics: dict[str, Any] = {}
    if run_gates and evaluated:
        validation = gate_pattern(rets, bias=bias)
        approved = bool(validation.get("approved"))
        soft_promoted = bool(validation.get("soft_promoted"))
        quality = float(validation.get("quality_score") or 0.0)
        metrics = validation.get("metrics") or {}
        # Prefer gate metrics for success_rate consistency
        if metrics.get("success_rate") is not None:
            success_rate = float(metrics["success_rate"])
            successes = int(metrics.get("successes") or successes)

    sample_idx = hit_idx[-max_events:]
    ts_col = (
        df["timestamp"].astype(str).to_numpy()
        if "timestamp" in df.columns
        else np.array([str(i) for i in range(n)])
    )
    strength_col = (
        df["pat_strength"].to_numpy()
        if "pat_strength" in df.columns
        else np.full(n, np.nan)
    )
    events: list[dict[str, Any]] = []
    for i in sample_idx:
        fr_i = fwd[i]
        if np.isnan(fr_i):
            success = None
        else:
            from atis.shared.pattern_discovery.validation import success_mask

            success = int(bool(success_mask(np.array([fr_i]), bias)[0]))
        events.append(
            {
                "ts": str(ts_col[i]),
                "close": float(close[i]),
                "strength": None if np.isnan(strength_col[i]) else float(strength_col[i]),
                "forward_return": None if np.isnan(fr_i) else float(fr_i),
                "success": success,
            }
        )

    last_ts = str(ts_col[hit_idx[-1]]) if len(hit_idx) else None
    strength = float(quality * (success_rate or 0.0)) if success_rate is not None else quality
    return {
        "occurrences": occ,
        "evaluated": evaluated,
        "successes": successes,
        "success_rate": success_rate,
        "avg_forward_return": avg_ret,
        "std_dev": std_dev if evaluated else metrics.get("std_dev"),
        "risk_ratio": risk_ratio,
        "expectancy": metrics.get("expectancy"),
        "profit_factor": metrics.get("profit_factor"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate", success_rate),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "quality_score": quality,
        "strength": strength,
        "confidence": float(conf),
        "approved": approved,
        "soft_promoted": soft_promoted,
        "validation": validation,
        "last_seen_ts": last_ts,
        "events": events,
        "forward_returns": rets if evaluated else np.array([]),
    }


def _stat_row(sym: str, tf: str, col: str, bias: str, ev: dict[str, Any], conditions: Any) -> dict[str, Any]:
    return {
        "symbol": sym,
        "timeframe": tf,
        "pattern_key": col,
        "bias": bias,
        "occurrences": ev["occurrences"],
        "evaluated": ev["evaluated"],
        "successes": ev["successes"],
        "success_rate": ev["success_rate"],
        "avg_forward_return": ev["avg_forward_return"],
        "std_dev": ev.get("std_dev"),
        "risk_ratio": ev.get("risk_ratio"),
        "expectancy": ev.get("expectancy"),
        "profit_factor": ev.get("profit_factor"),
        "sharpe": ev.get("sharpe"),
        "max_drawdown": ev.get("max_drawdown"),
        "win_rate": ev.get("win_rate"),
        "precision": ev.get("precision"),
        "recall": ev.get("recall"),
        "f1": ev.get("f1"),
        "quality_score": ev.get("quality_score"),
        "strength": ev.get("strength"),
        "confidence": ev["confidence"],
        "approved": ev.get("approved"),
        "soft_promoted": ev.get("soft_promoted"),
        "validation": ev.get("validation"),
        "best_timeframe": tf,
        "last_seen_ts": ev["last_seen_ts"],
        "conditions": conditions,
    }


def run_pattern_discovery(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    force_rebuild: bool = False,
    forward_horizon: int | None = None,
    progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
    details: DetailsFn | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Full deep discovery pipeline per timeframe (independent scan):
      1) rebuild features
      2) export + open full discovery JSONL
      3) evaluate catalog / compound columns
      4) rare compounds + deep NewN mining + validation gates
      5) relations graph
      6) KB upsert + JSON/report export
    """
    report: ProgressFn = progress or _noop_progress
    check_cancel: CancelFn = cancel_check or _noop_cancel
    push_details: DetailsFn = details or (lambda _d: None)

    cfg = load_engine_config()
    trading = cfg.get("trading", {})
    if not symbols:
        symbols = [str(trading.get("primary_symbol", "XAUUSD"))]
    if not timeframes:
        timeframes = list(load_timeframes().keys())

    kb = PatternKnowledgeBase()
    labels = pattern_labels()
    cats = pattern_category_map()
    bias_map = {k: v.get("bias", "neutral") for k, v in PATTERN_CATALOG.items()}

    stages = [
        ("إعادة بناء الميزات والأنماط", 10),
        ("تصدير وفتح ملفات JSON لكل إطار", 12),
        ("مسح الاكتشافات على كامل البيانات", 28),
        ("تقييم إحصائي وبوابات الاعتماد", 18),
        ("Deep Mining للأنماط الجديدة NewN", 14),
        ("شبكة العلاقات وتحديث KB/JSON", 18),
    ]
    total_w = sum(w for _, w in stages)
    stages = [(n, 100.0 * w / total_w) for n, w in stages]
    cursor = 0.0
    t0 = time.perf_counter()
    patterns_found = 0
    bars_done = 0

    def advance(weight: float, msg: str, **extra: Any) -> None:
        nonlocal cursor
        check_cancel()
        cursor = min(99.5, cursor + weight)
        elapsed = max(1e-6, time.perf_counter() - t0)
        speed = bars_done / elapsed if bars_done else 0.0
        eta = None
        if cursor > 1 and speed > 0:
            # rough ETA from remaining progress mass
            rem_pct = max(0.0, 100.0 - cursor)
            eta = rem_pct / max(cursor / elapsed, 1e-6)
        detail = {
            "pct": cursor,
            "message": msg,
            "bars_scanned": bars_done,
            "patterns_found": patterns_found,
            "speed_bars_s": round(speed, 1),
            "eta_sec": None if eta is None else round(eta, 1),
            **extra,
        }
        push_details(detail)
        speed_txt = f"{speed:.0f} bars/s" if speed else "—"
        eta_txt = f"{eta:.0f}s" if eta is not None else "—"
        report(
            cursor,
            f"{msg} · {patterns_found} نمط · {speed_txt} · ETA {eta_txt}",
        )

    report(1.0, f"بدء Deep Pattern Mining · {symbols} · {len(timeframes)} أطر")
    check_cancel()
    feat_report = run_features(symbols, timeframes, force_rebuild=force_rebuild)
    advance(stages[0][1], "اكتمل إعادة بناء الميزات")

    all_stats: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_discovered: list[dict[str, Any]] = []
    json_files: dict[str, dict[str, dict[str, str]]] = {}
    source_jsonl: dict[str, dict[str, str]] = {}
    reports: dict[str, dict[str, str]] = {}
    n_jobs = max(1, len(symbols) * len(timeframes))
    json_unit = stages[1][1] / n_jobs
    scan_unit = stages[2][1] / n_jobs
    eval_unit = stages[3][1] / n_jobs
    deep_unit = stages[4][1] / n_jobs
    rel_unit = stages[5][1] / n_jobs

    pattern_cols_global = list(labels.keys())

    for sym in symbols:
        json_files.setdefault(sym, {})
        source_jsonl.setdefault(sym, {})
        reports.setdefault(sym, {})
        for tf in timeframes:
            check_cancel()
            ckpt = load_checkpoint(sym, tf) if resume else None
            state: dict[str, Any] = dict(ckpt or {"completed_stages": []})

            json_path = get_path("data_features") / sym / tf / "features.json"
            report(cursor, f"تجهيز JSON · {sym} · {tf}")
            if not json_path.exists():
                advance(
                    json_unit + scan_unit + eval_unit + deep_unit + rel_unit,
                    f"تخطّي {sym}/{tf} — لا ملف features.json",
                    timeframe=tf,
                )
                continue

            source_df = load_timeframe_json(json_path)
            jsonl_path, df = ensure_discovery_bars_jsonl(
                sym,
                tf,
                source_df=source_df,
                force=True,
            )
            source_jsonl[sym][tf] = str(jsonl_path)
            bars_done += int(len(df))
            advance(
                json_unit,
                f"فُتح JSON كامل · {sym}/{tf} · {len(df)} شمعة",
                timeframe=tf,
                bars=len(df),
            )
            mark_stage(state, "json_loaded", bars=len(df))
            save_checkpoint(sym, tf, state)

            # Reload prior discoveries into frame (batch-assign to avoid fragmentation)
            _extra_cols: dict[str, pd.Series] = {}
            for disc in kb.list_discovered(500):
                legs = []
                try:
                    import json

                    legs = json.loads(disc.get("legs_json") or "[]")
                except Exception:
                    legs = []
                key = disc["compound_key"]
                if key.startswith("New") and key in df.columns:
                    labels.setdefault(key, disc.get("name") or key)
                    cats.setdefault(key, "discovered")
                    bias_map.setdefault(key, disc.get("bias") or "neutral")
                    continue
                if len(legs) >= 2 and all(l in df.columns for l in legs):
                    if key not in df.columns and key not in _extra_cols:
                        mask = pd.Series(True, index=df.index)
                        for leg in legs:
                            mask = mask & (df[leg] == 1)
                        _extra_cols[key] = mask.astype(int)
                    labels.setdefault(key, disc.get("name") or key)
                    cats.setdefault(key, "compound")
                    bias_map.setdefault(key, "neutral")
            if _extra_cols:
                df = pd.concat([df, pd.DataFrame(_extra_cols, index=df.index)], axis=1)

            pat_cols = [c for c in pattern_cols_global if c in df.columns]
            extra = [
                c
                for c in df.columns
                if (
                    c.startswith("pat_")
                    or c.startswith("cmp_")
                    or c.startswith("disc_")
                    or c.startswith("New")
                )
                and c not in pat_cols
                and c not in {"pat_bias", "pat_strength"}
            ]
            pat_cols = pat_cols + extra
            advance(
                scan_unit * 0.35,
                f"وُجد {len(pat_cols)} عمود نمط · {sym}/{tf}",
                timeframe=tf,
            )

            horizon = forward_horizon or _horizon_for_tf(tf)
            tf_stats: list[dict[str, Any]] = []
            tf_events: list[dict[str, Any]] = []
            n_cols = max(1, len(pat_cols))

            if stage_done(state, "catalog_eval"):
                # Resume: reload prior stats/events from KB for this TF
                tf_stats = [
                    {
                        "symbol": s.get("symbol") or sym,
                        "timeframe": s.get("timeframe") or tf,
                        "pattern_key": s.get("pattern_key"),
                        "bias": s.get("bias") or "neutral",
                        "occurrences": s.get("occurrences"),
                        "evaluated": s.get("evaluated"),
                        "successes": s.get("successes"),
                        "success_rate": s.get("success_rate"),
                        "avg_forward_return": s.get("avg_forward_return"),
                        "std_dev": s.get("std_dev"),
                        "quality_score": s.get("quality_score"),
                        "strength": s.get("strength"),
                        "confidence": s.get("confidence"),
                        "approved": bool(s["approved"]) if s.get("approved") is not None else None,
                        "soft_promoted": None,
                        "last_seen_ts": s.get("last_seen_ts"),
                        "conditions": s.get("conditions") or s.get("catalog_conditions"),
                        "best_timeframe": tf,
                    }
                    for s in kb.list_stats(sym, tf, min_occurrences=1, limit=5000)
                ]
                patterns_found += len(tf_stats)
                advance(scan_unit * 0.65 + eval_unit * 0.4, f"استئناف تقييم كتالوج · {len(tf_stats)} · {sym}/{tf}")
            else:
                start_i = int(state.get("eval_col_offset") or 0)
                done_keys = {r.get("pattern_key") for r in (state.get("partial_stat_rows") or [])}
                if state.get("partial_stat_rows"):
                    tf_stats = list(state["partial_stat_rows"])
                    patterns_found += len(tf_stats)
                for i, col in enumerate(pat_cols):
                    if i < start_i or col in done_keys:
                        continue
                    check_cancel()
                    meta = PATTERN_CATALOG.get(col, {})
                    bias = bias_map.get(col) or meta.get("bias") or "neutral"
                    ev = _evaluate_pattern_vectorized(
                        df,
                        col,
                        bias=bias,
                        horizon=horizon,
                        max_events=MAX_EVENTS_PER_PATTERN,
                    )
                    if ev["occurrences"] > 0:
                        row = _stat_row(
                            sym,
                            tf,
                            col,
                            bias,
                            ev,
                            meta.get("conditions") or cats.get(col, "pattern"),
                        )
                        tf_stats.append(row)
                        all_stats.append(row)
                        patterns_found += 1
                        for e in ev["events"]:
                            event = {
                                "symbol": sym,
                                "timeframe": tf,
                                "pattern_key": col,
                                **e,
                                "meta": {
                                    "horizon": horizon,
                                    "bias": bias,
                                    "approved": ev.get("approved"),
                                    "soft_promoted": ev.get("soft_promoted"),
                                },
                            }
                            tf_events.append(event)
                            all_events.append(event)
                    if i % 8 == 0:
                        # Fine-grained resume checkpoint (column offset)
                        mark_stage(
                            state,
                            "catalog_eval_partial",
                            eval_col_offset=i + 1,
                            partial_stat_rows=[
                                {k: v for k, v in r.items() if k != "validation"}
                                for r in tf_stats
                            ],
                        )
                        # Keep stage list without marking catalog_eval done yet
                        done = [s for s in (state.get("completed_stages") or []) if s != "catalog_eval"]
                        state["completed_stages"] = done
                        save_checkpoint(sym, tf, state)
                        advance(
                            0.0,
                            f"تقييم {col} · {i + 1}/{n_cols} · {sym}/{tf}",
                            timeframe=tf,
                            evaluated=i + 1,
                            total=n_cols,
                        )
                mark_stage(state, "catalog_eval", patterns=len(tf_stats), eval_col_offset=len(pat_cols))
                state.pop("partial_stat_rows", None)
                save_checkpoint(sym, tf, state)
                advance(scan_unit * 0.65 + eval_unit * 0.4, f"تقييم كتالوج · {len(tf_stats)} · {sym}/{tf}")

            # Rare compounds
            tf_discovered: list[dict[str, Any]] = []
            if not stage_done(state, "rare_compounds"):
                compound_seed = [
                    c for c in pat_cols if c.startswith("pat_") or c.startswith("cmp_")
                ]
                rare = discover_rare_compounds(
                    df,
                    compound_seed,
                    min_count=RARE_MIN_COUNT,
                    max_new=RARE_MAX_NEW,
                )
                for item in rare:
                    check_cancel()
                    legs = item["legs"]
                    leg_biases = [bias_map.get(leg, "neutral") for leg in legs]
                    first_bias = leg_biases[0] if leg_biases else "neutral"
                    bias = (
                        first_bias
                        if leg_biases and all(b == first_bias for b in leg_biases)
                        else "neutral"
                    )
                    tmp = df.copy()
                    mask = pd.Series(True, index=df.index)
                    for leg in legs:
                        mask = mask & (df[leg] == 1)
                    tmp[item["key"]] = mask.astype(int)
                    ev = _evaluate_pattern_vectorized(
                        tmp,
                        item["key"],
                        bias=bias,
                        horizon=horizon,
                    )
                    item.update(
                        {
                            "symbol": sym,
                            "timeframe": tf,
                            "bias": bias,
                            "success_rate": ev["success_rate"],
                            "confidence": ev["confidence"],
                            "occurrences": ev["occurrences"] or item["occurrences"],
                            "avg_forward_return": ev["avg_forward_return"],
                            "std_dev": ev.get("std_dev"),
                            "quality_score": ev.get("quality_score"),
                            "strength": ev.get("strength"),
                            "approved": ev.get("approved"),
                            "soft_promoted": ev.get("soft_promoted"),
                            "validation": ev.get("validation"),
                            "best_timeframe": tf,
                        }
                    )
                    tf_discovered.append(item)
                    all_discovered.append(item)
                    patterns_found += 1
                    if ev["occurrences"] > 0:
                        tf_stats.append(
                            _stat_row(sym, tf, item["key"], bias, ev, item.get("conditions"))
                        )
                        all_stats.append(tf_stats[-1])
                        for e in ev["events"]:
                            event = {
                                "symbol": sym,
                                "timeframe": tf,
                                "pattern_key": item["key"],
                                **e,
                                "meta": {"horizon": horizon, "bias": bias},
                            }
                            tf_events.append(event)
                            all_events.append(event)
                mark_stage(state, "rare_compounds", rare=len(rare))
                save_checkpoint(sym, tf, state)
            advance(eval_unit * 0.3, f"مركّبات نادرة · {len(tf_discovered)} · {sym}/{tf}")

            # Deep NewN mining
            new_patterns: list[dict[str, Any]] = []
            if not stage_done(state, "deep_newn"):
                existing_new = {c for c in df.columns if c.startswith("New")}
                for d in kb.list_discovered(500):
                    if str(d.get("compound_key", "")).startswith("New"):
                        existing_new.add(d["compound_key"])
                deep = discover_deep_patterns(
                    df,
                    max_new=DEEP_MAX_NEW,
                    min_count=DEEP_MIN_COUNT,
                    existing_keys=existing_new,
                )
                df = attach_signals_to_frame(df, deep)
                for item in deep:
                    check_cancel()
                    ev = _evaluate_pattern_vectorized(
                        df,
                        item["key"],
                        bias=item.get("bias") or "neutral",
                        horizon=horizon,
                    )
                    regime = "trend" if (item.get("kind") in {"momentum", "sequential_compound"}) else "mixed"
                    payload = {
                        **item,
                        "symbol": sym,
                        "timeframe": tf,
                        "success_rate": ev["success_rate"],
                        "avg_move_after": ev["avg_forward_return"],
                        "std_dev": ev.get("std_dev"),
                        "risk_ratio": ev.get("risk_ratio"),
                        "confidence": ev["confidence"],
                        "quality_score": ev.get("quality_score"),
                        "strength": ev.get("strength"),
                        "approved": bool(ev.get("approved")),
                        "soft_promoted": bool(ev.get("soft_promoted")),
                        "validation": ev.get("validation"),
                        "best_timeframe": tf,
                        "best_market_regime": regime,
                        "occurrences": ev["occurrences"] or item.get("occurrences"),
                        "lift": None,
                    }
                    sig = payload.pop("signal", None)
                    if sig is not None:
                        df[item["key"]] = np.asarray(sig, dtype=int)
                    new_patterns.append(payload)
                    all_discovered.append(payload)
                    patterns_found += 1
                    labels[item["key"]] = item["name"]
                    cats[item["key"]] = "discovered"
                    bias_map[item["key"]] = item.get("bias") or "neutral"
                    if ev["occurrences"] > 0:
                        row = _stat_row(
                            sym,
                            tf,
                            item["key"],
                            item.get("bias") or "neutral",
                            ev,
                            item.get("mathematical_rules"),
                        )
                        row["approved"] = payload["approved"]
                        row["soft_promoted"] = payload["soft_promoted"]
                        row["description"] = item.get("description")
                        row["mathematical_rules"] = item.get("mathematical_rules")
                        row["logical_rules"] = item.get("logical_rules")
                        row["appearance_conditions"] = item.get("appearance_conditions")
                        row["best_market_regime"] = regime
                        tf_stats.append(row)
                        all_stats.append(row)
                        for e in ev["events"]:
                            event = {
                                "symbol": sym,
                                "timeframe": tf,
                                "pattern_key": item["key"],
                                **e,
                                "meta": {
                                    "horizon": horizon,
                                    "bias": item.get("bias"),
                                    "approved": payload["approved"],
                                    "new_pattern": True,
                                },
                            }
                            tf_events.append(event)
                            all_events.append(event)
                mark_stage(
                    state,
                    "deep_newn",
                    new_count=len(new_patterns),
                    approved=sum(1 for p in new_patterns if p.get("approved")),
                )
                save_checkpoint(sym, tf, state)
            advance(
                deep_unit,
                f"NewN: {len(new_patterns)} · معتمد {sum(1 for p in new_patterns if p.get('approved'))} · {sym}/{tf}",
            )

            # Relations graph
            relations: dict[str, Any] = {}
            if not stage_done(state, "relations"):
                rel_cols = [
                    c
                    for c in df.columns
                    if (
                        c.startswith("pat_")
                        or c.startswith("cmp_")
                        or c.startswith("disc_")
                        or c.startswith("New")
                    )
                    and c not in {"pat_bias", "pat_strength"}
                ]
                relations = build_pattern_relations(df, rel_cols)
                mark_stage(state, "relations", edges=len(relations.get("edges") or []))
                save_checkpoint(sym, tf, state)

            # Persist TF results (partial-safe)
            check_cancel()
            # Bar-level NewN / promoted pattern matrix for Engine4
            from atis.shared.pattern_store import save_pattern_signal_matrix

            promote_keys = [
                str(r.get("pattern_key") or r.get("key") or "")
                for r in (tf_stats + new_patterns)
                if (r.get("approved") or r.get("soft_promoted"))
                and str(r.get("pattern_key") or r.get("key") or "").startswith(("New", "disc_", "pat_", "cmp_"))
            ]
            promote_keys = [k for k in dict.fromkeys(promote_keys) if k in df.columns]
            signal_path = save_pattern_signal_matrix(
                symbol=sym,
                timeframe=tf,
                df=df,
                columns=promote_keys,
            )

            written = save_timeframe_pattern_bundle(
                symbol=sym,
                timeframe=tf,
                stats=tf_stats,
                events=tf_events,
                compounds=tf_discovered + [p for p in new_patterns if not str(p.get("key", "")).startswith("New")],
                new_patterns=new_patterns,
                relations=relations,
                bars_scanned=int(len(df)),
            )
            if signal_path:
                written["signal_matrix"] = str(signal_path)
            report_path = write_discovery_report(
                symbol=sym,
                timeframe=tf,
                stats=tf_stats,
                new_patterns=new_patterns,
                relations=relations,
                bars_scanned=int(len(df)),
            )
            written["report"] = str(report_path)
            json_files[sym][tf] = written
            reports[sym][tf] = str(report_path)

            # Incremental KB write so cancel doesn't lose TF work
            kb.upsert_stats(tf_stats)
            kb.insert_events(tf_events)
            kb.upsert_discovered(tf_discovered + new_patterns)

            mark_stage(state, "exported")
            clear_checkpoint(sym, tf)
            advance(
                rel_unit,
                f"صدّر JSON+تقرير · علاقات {len(relations.get('edges') or [])} · {sym}/{tf}",
                timeframe=tf,
            )

    check_cancel()
    report(cursor, "إعادة بناء أقسام JSON من KB…")
    final_export = export_pattern_json_from_kb(symbols, timeframes)
    for sym, tf_map in (final_export.get("json_files") or {}).items():
        json_files.setdefault(sym, {}).update(tf_map)

    advance(0.5, "اكتمل حفظ قاعدة المعرفة وملفات JSON")
    summary = kb.summary(symbols[0], None)
    report(100.0, f"اكتمل الاستكشاف · {summary['patterns_with_hits']} نمطاً بإصابات")

    return {
        "symbols": symbols,
        "timeframes": timeframes,
        "features": feat_report.to_dict() if hasattr(feat_report, "to_dict") else str(feat_report),
        "patterns_evaluated": len(all_stats),
        "events_stored": len(all_events),
        "compounds_discovered": len(all_discovered),
        "patterns_found": patterns_found,
        "catalog": summary["catalog"],
        "knowledge": summary,
        "source_jsonl": source_jsonl,
        "json_files": json_files,
        "reports": reports,
        "patterns_root": str(final_export.get("root") or ""),
    }


def export_pattern_json_from_kb(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> dict[str, Any]:
    """Rebuild section JSON files from the existing PatternKnowledgeBase."""
    import json

    cfg = load_engine_config()
    trading = cfg.get("trading", {})
    if not symbols:
        symbols = [str(trading.get("primary_symbol", "XAUUSD"))]
    if not timeframes:
        timeframes = list(load_timeframes().keys())

    kb = PatternKnowledgeBase()
    written: dict[str, dict[str, dict[str, str]]] = {}

    for sym in symbols:
        written[sym] = {}
        for tf in timeframes:
            stats = kb.list_stats(sym, tf, min_occurrences=1, limit=5000)
            compounds = []
            new_patterns = []
            for d in kb.list_discovered(500):
                if d.get("timeframe") and d.get("timeframe") != tf:
                    continue
                if d.get("symbol") and d.get("symbol") != sym:
                    continue
                legs = []
                try:
                    legs = json.loads(d.get("legs_json") or "[]")
                except Exception:
                    legs = []
                meta = {}
                try:
                    meta = json.loads(d.get("meta_json") or "{}")
                except Exception:
                    meta = {}
                item = {
                    "key": d.get("compound_key"),
                    "name": d.get("name"),
                    "legs": legs,
                    "lift": d.get("lift"),
                    "occurrences": d.get("occurrences"),
                    "success_rate": d.get("success_rate"),
                    "confidence": d.get("confidence"),
                    "conditions": d.get("conditions"),
                    "bias": meta.get("bias") or "neutral",
                    "approved": meta.get("approved"),
                    "quality_score": meta.get("quality_score"),
                    "description": meta.get("description"),
                    "mathematical_rules": meta.get("mathematical_rules"),
                    "logical_rules": meta.get("logical_rules"),
                    "appearance_conditions": meta.get("appearance_conditions"),
                    "std_dev": meta.get("std_dev"),
                    "risk_ratio": meta.get("risk_ratio"),
                    "strength": meta.get("strength"),
                    "best_timeframe": meta.get("best_timeframe") or tf,
                    "best_market_regime": meta.get("best_market_regime"),
                    "validation": meta.get("validation"),
                }
                if str(item["key"] or "").startswith("New"):
                    new_patterns.append(item)
                else:
                    compounds.append(item)

            events: list[dict[str, Any]] = []
            with kb._conn() as con:
                rows = con.execute(
                    """
                    SELECT symbol, timeframe, pattern_key, ts, close, strength,
                           forward_return, success, meta_json
                    FROM pattern_events
                    WHERE symbol=? AND timeframe=?
                    ORDER BY ts DESC
                    LIMIT 50000
                    """,
                    (sym, tf),
                ).fetchall()
            for r in rows:
                try:
                    meta = json.loads(r["meta_json"] or "{}")
                except Exception:
                    meta = {}
                events.append(
                    {
                        "symbol": r["symbol"],
                        "timeframe": r["timeframe"],
                        "pattern_key": r["pattern_key"],
                        "ts": r["ts"],
                        "close": r["close"],
                        "strength": r["strength"],
                        "forward_return": r["forward_return"],
                        "success": r["success"],
                        "meta": meta,
                    }
                )

            bars = None
            jsonl = get_path("data_features") / sym / tf / "discovery_bars.jsonl"
            if jsonl.exists():
                try:
                    with jsonl.open("r", encoding="utf-8") as fh:
                        bars = sum(1 for line in fh if line.strip())
                except Exception:
                    bars = None
            if bars is None:
                feat_path = get_path("data_features") / sym / tf / "features.parquet"
                if feat_path.exists():
                    try:
                        bars = int(len(pd.read_parquet(feat_path, columns=["timestamp"])))
                    except Exception:
                        bars = None

            written[sym][tf] = save_timeframe_pattern_bundle(
                symbol=sym,
                timeframe=tf,
                stats=stats,
                events=events,
                compounds=compounds,
                new_patterns=new_patterns,
                bars_scanned=bars,
            )

    from atis.shared.pattern_store import patterns_root

    return {
        "symbols": symbols,
        "timeframes": timeframes,
        "json_files": written,
        "root": str(patterns_root()),
    }
