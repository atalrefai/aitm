"""Per-timeframe JSON storage for advanced pattern results.
Layout:
  data/patterns/{symbol}/{timeframe}/candlesticks.json
  data/patterns/{symbol}/{timeframe}/structural.json
  data/patterns/{symbol}/{timeframe}/compounds.json
  data/patterns/{symbol}/{timeframe}/knowledge.json
  data/patterns/{symbol}/{timeframe}/discovery_log.json
  data/patterns/{symbol}/{timeframe}/new_patterns.json
  data/patterns/{symbol}/{timeframe}/rankings.json
  data/patterns/{symbol}/{timeframe}/relations.json
  data/patterns/{symbol}/{timeframe}/validation_report.json
  data/patterns/{symbol}/{timeframe}/discovery_report.md
"""
from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from atis.config import PROJECT_ROOT, get_path
from atis.shared.feature_engine.patterns import (
    PATTERN_CATALOG,
    pattern_category_map,
    pattern_labels,
)

SECTION_CANDLESTICKS = "candlesticks"

SECTION_STRUCTURAL = "structural"

SECTION_COMPOUNDS = "compounds"

SECTION_KNOWLEDGE = "knowledge"

SECTION_DISCOVERY_LOG = "discovery_log"

SECTION_NEW_PATTERNS = "new_patterns"

SECTION_RANKINGS = "rankings"

SECTION_RELATIONS = "relations"

SECTION_VALIDATION = "validation_report"

_STORE_LOCK = threading.RLock()

SECTIONS = (
    SECTION_CANDLESTICKS,
    SECTION_STRUCTURAL,
    SECTION_COMPOUNDS,
    SECTION_KNOWLEDGE,
    SECTION_DISCOVERY_LOG,
    SECTION_NEW_PATTERNS,
    SECTION_RANKINGS,
    SECTION_RELATIONS,
    SECTION_VALIDATION,
)

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def patterns_root() -> Path:
    try:
        root = get_path("data_patterns")
    except KeyError:
        root = PROJECT_ROOT / "data" / "patterns"
    root.mkdir(parents=True, exist_ok=True)
    return root

def timeframe_dir(symbol: str, timeframe: str) -> Path:
    path = patterns_root() / symbol / timeframe
    path.mkdir(parents=True, exist_ok=True)
    return path

def section_path(symbol: str, timeframe: str, section: str) -> Path:
    if section not in SECTIONS:
        raise ValueError(f"Unknown pattern section: {section}")
    return timeframe_dir(symbol, timeframe) / f"{section}.json"

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STORE_LOCK:
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return path

def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def _resolve_category(pattern_key: str, cats: dict[str, str] | None = None) -> str:
    cats = cats or pattern_category_map()
    if pattern_key in cats:
        return cats[pattern_key]
    if pattern_key.startswith("New"):
        return "discovered"
    if pattern_key.startswith("disc_") or pattern_key.startswith("cmp_"):
        return "compound"
    meta = PATTERN_CATALOG.get(pattern_key) or {}
    return str(meta.get("category") or "candle")

def _enrich_stat(row: dict[str, Any], labels: dict[str, str], cats: dict[str, str]) -> dict[str, Any]:
    key = str(row.get("pattern_key") or "")
    category = _resolve_category(key, cats)
    meta = PATTERN_CATALOG.get(key) or {}
    return {
        "pattern_key": key,
        "name": labels.get(key) or row.get("name") or key,
        "category": category,
        "bias": row.get("bias") or meta.get("bias") or "neutral",
        "conditions": row.get("conditions") or meta.get("conditions"),
        "occurrences": int(row.get("occurrences") or 0),
        "evaluated": int(row.get("evaluated") or 0),
        "successes": int(row.get("successes") or 0),
        "success_rate": row.get("success_rate"),
        "avg_forward_return": row.get("avg_forward_return"),
        "std_dev": row.get("std_dev"),
        "risk_ratio": row.get("risk_ratio"),
        "expectancy": row.get("expectancy"),
        "profit_factor": row.get("profit_factor"),
        "sharpe": row.get("sharpe"),
        "max_drawdown": row.get("max_drawdown"),
        "win_rate": row.get("win_rate"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "quality_score": row.get("quality_score"),
        "strength": row.get("strength"),
        "confidence": row.get("confidence"),
        "approved": row.get("approved"),
        "soft_promoted": row.get("soft_promoted"),
        "validation": row.get("validation"),
        "best_timeframe": row.get("best_timeframe"),
        "best_market_regime": row.get("best_market_regime"),
        "description": row.get("description"),
        "mathematical_rules": row.get("mathematical_rules"),
        "logical_rules": row.get("logical_rules"),
        "appearance_conditions": row.get("appearance_conditions"),
        "last_seen_ts": row.get("last_seen_ts"),
        "symbol": row.get("symbol"),
        "timeframe": row.get("timeframe"),
    }

def _section_payload(
    *,
    symbol: str,
    timeframe: str,
    section: str,
    items: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "section": section,
        "symbol": symbol,
        "timeframe": timeframe,
        "updated_at": _utc(),
        "count": len(items),
        "items": items,
    }
    if extra:
        body.update(extra)
    return body

def _rank_key(row: dict[str, Any], field: str) -> float:
    v = row.get(field)
    try:
        return float(v) if v is not None else -1e18
    except (TypeError, ValueError):
        return -1e18


def _engine4_eligible(row: dict[str, Any], timeframe: str) -> bool:
    """Engine4 recommend gate: approve/soft + TF sample floors; reject ultra-rare."""
    from atis.shared.pattern_discovery.validation import gates_for_timeframe

    g = gates_for_timeframe(timeframe)
    min_occ = int(g.get("min_occurrences_rank", g.get("min_evaluated", 25)))
    min_sr = float(g.get("min_success_rate", 0.54))
    min_eval = int(g.get("min_evaluated", 25))
    if not (row.get("approved") or row.get("soft_promoted")):
        return False
    if row.get("htf_confirm") is False:
        return False
    if (row.get("occurrences") or 0) < min_occ:
        return False
    if (row.get("evaluated") or 0) < min_eval:
        return False
    if (row.get("success_rate") or 0) < min_sr:
        return False
    return True

def save_timeframe_pattern_bundle(
    *,
    symbol: str,
    timeframe: str,
    stats: list[dict[str, Any]],
    events: list[dict[str, Any]],
    compounds: list[dict[str, Any]],
    bars_scanned: int | None = None,
    new_patterns: list[dict[str, Any]] | None = None,
    relations: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write section JSON files for one symbol/timeframe."""
    labels = pattern_labels()
    cats = pattern_category_map()
    new_patterns = new_patterns or []
    relations = relations or {}
    enriched = [_enrich_stat(r, labels, cats) for r in stats if r.get("occurrences")]
    candles = [r for r in enriched if r["category"] == "candle"]
    structural = [r for r in enriched if r["category"] == "chart"]
    compound_stats = [r for r in enriched if r["category"] == "compound"]
    discovered_stats = [r for r in enriched if r["category"] == "discovered" or str(r["pattern_key"]).startswith("New")]
    compound_items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in compounds:
        key = str(item.get("key") or item.get("compound_key") or item.get("pattern_key") or "")
        if not key or key in seen_keys or key.startswith("New"):
            continue
        seen_keys.add(key)
        compound_items.append(
            {
                "pattern_key": key,
                "name": item.get("name") or labels.get(key) or key,
                "category": "compound",
                "legs": item.get("legs"),
                "lift": item.get("lift"),
                "bias": item.get("bias") or "neutral",
                "conditions": item.get("conditions"),
                "occurrences": item.get("occurrences"),
                "success_rate": item.get("success_rate"),
                "confidence": item.get("confidence"),
                "quality_score": item.get("quality_score"),
                "approved": item.get("approved"),
                "source": "discovered" if key.startswith("disc_") else "compound",
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )
    for row in compound_stats:
        key = row["pattern_key"]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        compound_items.append({**row, "source": "compound"})
    new_items: list[dict[str, Any]] = []
    seen_new: set[str] = set()
    for item in new_patterns:
        key = str(item.get("key") or item.get("id") or item.get("pattern_key") or "")
        if not key or key in seen_new:
            continue
        seen_new.add(key)
        new_items.append(
            {
                "id": key,
                "name": item.get("name") or key,
                "description": item.get("description"),
                "mathematical_rules": item.get("mathematical_rules"),
                "logical_rules": item.get("logical_rules"),
                "appearance_conditions": item.get("appearance_conditions"),
                "occurrences": item.get("occurrences"),
                "success_rate": item.get("success_rate"),
                "avg_move_after": item.get("avg_move_after") or item.get("avg_forward_return"),
                "std_dev": item.get("std_dev"),
                "risk_ratio": item.get("risk_ratio"),
                "best_timeframe": item.get("best_timeframe") or timeframe,
                "best_market_regime": item.get("best_market_regime"),
                "confidence": item.get("confidence"),
                "quality_score": item.get("quality_score"),
                "strength": item.get("strength"),
                "approved": item.get("approved"),
                "soft_promoted": item.get("soft_promoted"),
                "validation": item.get("validation"),
                "htf_confirm": item.get("htf_confirm"),
                "bias": item.get("bias") or "neutral",
                "category": "discovered",
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )
    for row in discovered_stats:
        key = row["pattern_key"]
        if key in seen_new:
            continue
        seen_new.add(key)
        new_items.append(
            {
                "id": key,
                "name": row.get("name") or key,
                **{k: row.get(k) for k in (
                    "description", "mathematical_rules", "logical_rules", "appearance_conditions",
                    "occurrences", "success_rate", "std_dev", "risk_ratio", "confidence",
                    "quality_score", "strength", "approved", "soft_promoted", "htf_confirm",
                    "validation", "bias",
                    "best_timeframe", "best_market_regime",
                )},
                "avg_move_after": row.get("avg_forward_return"),
                "category": "discovered",
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )
    knowledge_items = [
        {
            "pattern_key": r["pattern_key"],
            "name": r["name"],
            "category": r["category"],
            "bias": r["bias"],
            "occurrences": r["occurrences"],
            "evaluated": r["evaluated"],
            "successes": r["successes"],
            "success_rate": r["success_rate"],
            "avg_forward_return": r["avg_forward_return"],
            "std_dev": r.get("std_dev"),
            "quality_score": r.get("quality_score"),
            "strength": r.get("strength"),
            "confidence": r["confidence"],
            "approved": r.get("approved"),
            "last_seen_ts": r["last_seen_ts"],
            "conditions": r["conditions"],
        }
        for r in enriched
    ]
    knowledge_items.sort(
        key=lambda x: (
            -(x.get("quality_score") or 0.0),
            -(x.get("success_rate") or 0.0),
            -(x.get("occurrences") or 0),
            x.get("pattern_key") or "",
        )
    )
    discovery_items = []
    for e in events:
        if e.get("symbol") != symbol or e.get("timeframe") != timeframe:
            continue
        key = str(e.get("pattern_key") or "")
        discovery_items.append(
            {
                "ts": e.get("ts"),
                "pattern_key": key,
                "name": labels.get(key) or key,
                "category": _resolve_category(key, cats),
                "close": e.get("close"),
                "strength": e.get("strength"),
                "forward_return": e.get("forward_return"),
                "success": e.get("success"),
                "meta": e.get("meta") or {},
            }
        )
    discovery_items.sort(key=lambda x: str(x.get("ts") or ""), reverse=True)
    rankings = {
        "by_strength": sorted(enriched, key=lambda r: _rank_key(r, "strength"), reverse=True)[:50],
        "by_profitability": sorted(
            enriched, key=lambda r: _rank_key(r, "avg_forward_return"), reverse=True
        )[:50],
        "by_confidence": sorted(enriched, key=lambda r: _rank_key(r, "confidence"), reverse=True)[:50],
        "by_success_rate": sorted(enriched, key=lambda r: _rank_key(r, "success_rate"), reverse=True)[:50],
        "by_quality": sorted(enriched, key=lambda r: _rank_key(r, "quality_score"), reverse=True)[:50],
        "engine4_recommended": [
            r
            for r in sorted(enriched, key=lambda x: _rank_key(x, "quality_score"), reverse=True)
            if _engine4_eligible(r, timeframe)
        ][:40],
    }
    validation_items = []
    for r in enriched:
        val = r.get("validation")
        if not val and r.get("approved") is None:
            continue
        validation_items.append(
            {
                "pattern_key": r["pattern_key"],
                "name": r["name"],
                "approved": r.get("approved"),
                "quality_score": r.get("quality_score"),
                "validation": val,
                "success_rate": r.get("success_rate"),
                "occurrences": r.get("occurrences"),
            }
        )
    paths = {section: section_path(symbol, timeframe, section) for section in SECTIONS}
    write_json(
        paths[SECTION_CANDLESTICKS],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_CANDLESTICKS,
            items=candles,
            extra={"bars_scanned": bars_scanned, "title": "أنماط الشموع اليابانية"},
        ),
    )
    write_json(
        paths[SECTION_STRUCTURAL],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_STRUCTURAL,
            items=structural,
            extra={"bars_scanned": bars_scanned, "title": "الأنماط الهيكلية"},
        ),
    )
    write_json(
        paths[SECTION_COMPOUNDS],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_COMPOUNDS,
            items=compound_items,
            extra={"bars_scanned": bars_scanned, "title": "الأنماط المركّبة / المكتشفة"},
        ),
    )
    write_json(
        paths[SECTION_KNOWLEDGE],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_KNOWLEDGE,
            items=knowledge_items,
            extra={
                "bars_scanned": bars_scanned,
                "title": "قاعدة معرفة الأنماط (ظهور · نجاح · ثقة)",
                "summary": {
                    "patterns_with_hits": len(knowledge_items),
                    "total_occurrences": int(sum(r.get("occurrences") or 0 for r in knowledge_items)),
                    "approved_count": int(sum(1 for r in knowledge_items if r.get("approved"))),
                    "avg_success_rate": (
                        float(
                            sum(r["success_rate"] for r in knowledge_items if r.get("success_rate") is not None)
                            / max(1, sum(1 for r in knowledge_items if r.get("success_rate") is not None))
                        )
                        if any(r.get("success_rate") is not None for r in knowledge_items)
                        else None
                    ),
                    "avg_confidence": (
                        float(
                            sum(r["confidence"] for r in knowledge_items if r.get("confidence") is not None)
                            / max(1, sum(1 for r in knowledge_items if r.get("confidence") is not None))
                        )
                        if any(r.get("confidence") is not None for r in knowledge_items)
                        else None
                    ),
                },
            },
        ),
    )
    write_json(
        paths[SECTION_DISCOVERY_LOG],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_DISCOVERY_LOG,
            items=discovery_items,
            extra={"bars_scanned": bars_scanned, "title": "سجل الاكتشافات التفصيلي"},
        ),
    )
    write_json(
        paths[SECTION_NEW_PATTERNS],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_NEW_PATTERNS,
            items=new_items,
            extra={
                "bars_scanned": bars_scanned,
                "title": "أنماط جديدة NewN",
                "approved": sum(1 for r in new_items if r.get("approved")),
                "rejected": sum(1 for r in new_items if r.get("approved") is False),
            },
        ),
    )
    write_json(
        paths[SECTION_RANKINGS],
        {
            "section": SECTION_RANKINGS,
            "symbol": symbol,
            "timeframe": timeframe,
            "updated_at": _utc(),
            "title": "ترتيب الأنماط",
            "bars_scanned": bars_scanned,
            **{k: _json_safe(v) for k, v in rankings.items()},
        },
    )
    # Never clobber a populated relations graph with an empty resume shell
    existing_rel = read_json(paths[SECTION_RELATIONS]) or {}
    if relations_has_content(relations) or not relations_has_content(existing_rel):
        write_json(
            paths[SECTION_RELATIONS],
            {
                "section": SECTION_RELATIONS,
                "symbol": symbol,
                "timeframe": timeframe,
                "updated_at": _utc(),
                "title": "شبكة علاقات الأنماط",
                "bars_scanned": bars_scanned,
                **_json_safe(relations),
            },
        )
    write_json(
        paths[SECTION_VALIDATION],
        _section_payload(
            symbol=symbol,
            timeframe=timeframe,
            section=SECTION_VALIDATION,
            items=validation_items,
            extra={
                "bars_scanned": bars_scanned,
                "title": "تقرير التحقق الإحصائي",
                "approved": sum(1 for r in validation_items if r.get("approved")),
                "rejected": sum(1 for r in validation_items if r.get("approved") is False),
            },
        ),
    )
    return {section: str(path) for section, path in paths.items()}


def relations_has_content(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("edges"))


def save_relations_section(
    *,
    symbol: str,
    timeframe: str,
    relations: dict[str, Any],
    bars_scanned: int | None = None,
) -> Path:
    """Persist / overwrite the relations.json section for one TF."""
    path = section_path(symbol, timeframe, SECTION_RELATIONS)
    payload = {
        "section": SECTION_RELATIONS,
        "symbol": symbol,
        "timeframe": timeframe,
        "updated_at": _utc(),
        "title": "شبكة علاقات الأنماط",
        "bars_scanned": bars_scanned if bars_scanned is not None else relations.get("bars"),
        **_json_safe(relations),
    }
    return write_json(path, payload)


def rebuild_relations_from_features(
    symbol: str,
    timeframe: str,
    *,
    lookback: int | None = None,
    df=None,
) -> dict[str, Any]:
    """Build relations from features and persist relations.json."""
    from atis.shared.pattern_discovery.relations import (
        build_pattern_relations,
        pattern_relation_columns,
    )

    if df is None:
        import pandas as pd

        path = get_path("data_features") / symbol / timeframe / "features.parquet"
        json_path = get_path("data_features") / symbol / timeframe / "features.json"
        if path.exists():
            df = pd.read_parquet(path)
        elif json_path.exists():
            from atis.shared.data_json import load_timeframe_json

            df = load_timeframe_json(json_path)
        else:
            return {
                "nodes": [],
                "edges": [],
                "sequences": [],
                "summary": "لا ميزات متاحة لبناء الشبكة",
                "empty": True,
            }
        if lookback and lookback > 0 and len(df) > lookback:
            df = df.tail(lookback).copy()
    labels = pattern_labels()
    rel_cols = pattern_relation_columns(df)
    graph = build_pattern_relations(df, rel_cols, labels=labels)
    save_relations_section(
        symbol=symbol,
        timeframe=timeframe,
        relations=graph,
        bars_scanned=int(len(df)),
    )
    return graph


def write_discovery_report(
    *,
    symbol: str,
    timeframe: str,
    stats: list[dict[str, Any]],
    new_patterns: list[dict[str, Any]],
    relations: dict[str, Any] | None = None,
    bars_scanned: int | None = None,
) -> Path:
    """Operational Markdown report under data/patterns/{symbol}/{tf}/."""
    labels = pattern_labels()
    cats = pattern_category_map()
    enriched = [_enrich_stat(r, labels, cats) for r in stats if r.get("occurrences")]
    known = [r for r in enriched if not str(r["pattern_key"]).startswith("New")]
    approved_new = [p for p in new_patterns if p.get("approved")]
    rejected_new = [p for p in new_patterns if p.get("approved") is False]
    best = sorted(enriched, key=lambda r: _rank_key(r, "quality_score"), reverse=True)[:10]
    weakest = sorted(
        [r for r in enriched if r.get("success_rate") is not None],
        key=lambda r: _rank_key(r, "success_rate"),
    )[:10]
    soft_new = [p for p in new_patterns if p.get("soft_promoted") or p.get("approved")]
    engine4 = [
        r
        for r in sorted(enriched, key=lambda x: _rank_key(x, "quality_score"), reverse=True)
        if _engine4_eligible(r, timeframe)
    ][:15]
    htf_blocked = sum(1 for r in enriched if r.get("htf_confirm") is False)
    lines = [
        f"# تقرير استكشاف الأنماط — {symbol} / {timeframe}",
        "",
        f"- التاريخ: `{_utc()}`",
        f"- الشموع الممسوحة: **{bars_scanned if bars_scanned is not None else '—'}**",
        f"- أنماط معروفة بإصابات: **{len(known)}**",
        f"- أنماط جديدة NewN: **{len(new_patterns)}** (معتمد: {len(approved_new)} · soft: {len(soft_new)} · مرفوض: {len(rejected_new)})",
        f"- علاقات: **{len((relations or {}).get('edges') or [])}** حافة",
        "",
        "## قواعد الترقية (Engine4)",
        "",
        "- بوابات التحقق حسب الإطار الزمني (`TF_GATE_OVERRIDES`): عينات أعلى لـ M1، ونجاح/PF أشد لـ H1/H4.",
        "- الأنماط النادرة جداً تُرفض تلقائياً (`rare_pattern` / `min_occurrences_rank`) ولا تدخل `engine4_recommended`.",
        "- قبل soft-promote / engine4_recommended: يُشترط اتساق اتجاه مع إطار أعلى (HTF) عبر `pat_bias` أو `chart_pattern_score` عند توفره؛ عدم التوافق يمنع الترقية.",
        f"- أنماط حُجبت بسبب عدم توافق HTF في هذا التشغيل: **{htf_blocked}**",
        "",
        "## أفضل الأنماط (quality_score)",
        "",
    ]
    for r in best:
        lines.append(
            f"- `{r['pattern_key']}` · نجاح={r.get('success_rate')} · ثقة={r.get('confidence')} · "
            f"جودة={r.get('quality_score')} · معتمد={r.get('approved')}"
        )
    lines.extend(["", "## أضعف الأنماط (success_rate)", ""])
    for r in weakest:
        lines.append(
            f"- `{r['pattern_key']}` · نجاح={r.get('success_rate')} · ظهور={r.get('occurrences')}"
        )
    lines.extend(["", "## توصيات دمج Engine4 / إعادة التدريب", ""])
    if engine4:
        for r in engine4:
            lines.append(
                f"- **دمج**: `{r['pattern_key']}` — جودة {r.get('quality_score')} · "
                f"نجاح {r.get('success_rate')} · ظهور {r.get('occurrences')}"
            )
    else:
        lines.append("- لا توجد أنماط اجتازت بوابة الاعتماد بجودة كافية حالياً.")
    lines.extend(
        [
            "",
            "## شبكة العلاقات (ملخص)",
            "",
            (relations or {}).get("summary") or "—",
            "",
            "## توصيات KB",
            "",
            "- تحديث الكتالوج بالأنماط NewN المعتمدة فقط.",
            "- إعادة مزامنة `pattern_knowledge.db` قبل تدريب Engine4 التالي.",
            "- استخدام `rankings.json → engine4_recommended` كمصدر ميزات مُروَّج.",
            "",
        ]
    )
    path = timeframe_dir(symbol, timeframe) / "discovery_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def list_pattern_files(symbol: str | None = None, timeframe: str | None = None) -> list[dict[str, Any]]:
    root = patterns_root()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for sym_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if symbol and sym_dir.name != symbol:
            continue
        for tf_dir in sorted(p for p in sym_dir.iterdir() if p.is_dir()):
            if timeframe and tf_dir.name != timeframe:
                continue
            for section in SECTIONS:
                path = tf_dir / f"{section}.json"
                if not path.exists():
                    continue
                meta = read_json(path) or {}
                rows.append(
                    {
                        "symbol": sym_dir.name,
                        "timeframe": tf_dir.name,
                        "section": section,
                        "path": str(path),
                        "count": meta.get("count"),
                        "updated_at": meta.get("updated_at"),
                        "title": meta.get("title"),
                    }
                )
    return rows

def load_section(symbol: str, timeframe: str, section: str) -> dict[str, Any] | None:
    return read_json(section_path(symbol, timeframe, section))

def save_pattern_signal_matrix(
    *,
    symbol: str,
    timeframe: str,
    df,
    columns: list[str],
):
    """Persist bar-level promoted pattern signals for Engine4 merge."""
    if not columns:
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None
    frame = {}
    out_cols = []
    if "timestamp" in df.columns:
        frame["timestamp"] = df["timestamp"]
        out_cols.append("timestamp")
    for c in cols:
        frame[c] = df[c].fillna(0).astype(int)
        out_cols.append(c)
    path = timeframe_dir(symbol, timeframe) / "pattern_signal_matrix.parquet"
    with _STORE_LOCK:
        pd.DataFrame(frame)[out_cols].to_parquet(path, index=False)
    return path

def load_pattern_signal_matrix(symbol: str, timeframe: str):
    import pandas as pd
    path = timeframe_dir(symbol, timeframe) / "pattern_signal_matrix.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
