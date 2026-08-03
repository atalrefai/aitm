"""Pattern relation graph: co-occurrence, precedes, cancels, optimal sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

RELATION_LABELS_AR = {
    "co_occurrence": "تزامن",
    "precedes": "يسبق",
    "cancels": "يتعارض",
    "optimal_sequence": "تسلسل أمثل",
}


def pattern_relation_columns(df: pd.DataFrame) -> list[str]:
    """Binary pattern columns eligible for the relation graph."""
    return [
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


def _pattern_bias(col: str) -> str:
    name = col.lower()
    bear = ("bear", "top", "shooting", "crows", "dark", "down", "sweep_high")
    bull = ("bull", "bottom", "hammer", "soldiers", "piercing", "up", "sweep_low")
    if any(x in name for x in bear):
        return "bearish"
    if any(x in name for x in bull):
        return "bullish"
    return "neutral"


def build_pattern_relations(
    df: pd.DataFrame,
    pattern_cols: list[str],
    *,
    lag_max: int = 5,
    min_count: int = 5,
    top_k: int = 80,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build a lightweight relation graph from binary pattern columns (causal lags).

    Relations:
      - co_occurrence: same bar
      - precedes: A at t, B at t+1..t+lag (A precedes B)
      - cancels: opposite bias patterns on the same bar
    """
    labels = labels or {}
    cols = [c for c in pattern_cols if c in df.columns]
    if len(cols) < 2:
        return {
            "nodes": [],
            "edges": [],
            "sequences": [],
            "summary": "لا علاقات كافية — أنماط ثنائية أقل من 2",
            "counts": {"co_occurrence": 0, "precedes": 0, "cancels": 0},
            "lag_max": lag_max,
            "bars": int(len(df)),
        }

    mat = {c: df[c].fillna(0).astype(int).to_numpy() for c in cols}
    n = len(df)

    biases = {c: _pattern_bias(c) for c in cols}
    # Prefer frequent patterns first so the 60-cap keeps the densest graph
    ranked = sorted(cols, key=lambda c: int(mat[c].sum()), reverse=True)
    active = [c for c in ranked if int(mat[c].sum()) >= min_count][:60]

    if len(active) < 2:
        return {
            "nodes": [],
            "edges": [],
            "sequences": [],
            "summary": f"لا علاقات كافية — أنماط نشطة (≥{min_count}) أقل من 2",
            "counts": {"co_occurrence": 0, "precedes": 0, "cancels": 0},
            "lag_max": lag_max,
            "bars": n,
        }

    co_counts: dict[tuple[str, str], int] = defaultdict(int)
    prec_counts: dict[tuple[str, str], int] = defaultdict(int)
    cancel_counts: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(n):
        fired = [c for c in active if mat[c][i] == 1]
        for a_i, a in enumerate(fired):
            for b in fired[a_i + 1 :]:
                key = tuple(sorted((a, b)))
                co_counts[key] += 1
                if biases[a] != "neutral" and biases[b] != "neutral" and biases[a] != biases[b]:
                    cancel_counts[key] += 1
        for lag in range(1, lag_max + 1):
            j = i + lag
            if j >= n:
                break
            later = [c for c in active if mat[c][j] == 1]
            for a in fired:
                for b in later:
                    if a == b:
                        continue
                    prec_counts[(a, b)] += 1

    edges: list[dict[str, Any]] = []
    for (a, b), cnt in co_counts.items():
        if cnt < min_count:
            continue
        edges.append(
            {
                "source": a,
                "target": b,
                "source_label": labels.get(a, a),
                "target_label": labels.get(b, b),
                "relation": "co_occurrence",
                "relation_ar": RELATION_LABELS_AR["co_occurrence"],
                "count": cnt,
                "weight": float(cnt) / n,
            }
        )
    for (a, b), cnt in sorted(prec_counts.items(), key=lambda x: -x[1])[: top_k * 2]:
        if cnt < min_count:
            continue
        edges.append(
            {
                "source": a,
                "target": b,
                "source_label": labels.get(a, a),
                "target_label": labels.get(b, b),
                "relation": "precedes",
                "relation_ar": RELATION_LABELS_AR["precedes"],
                "count": cnt,
                "weight": float(cnt) / n,
                "lag_max": lag_max,
            }
        )
    for (a, b), cnt in cancel_counts.items():
        if cnt < min_count:
            continue
        edges.append(
            {
                "source": a,
                "target": b,
                "source_label": labels.get(a, a),
                "target_label": labels.get(b, b),
                "relation": "cancels",
                "relation_ar": RELATION_LABELS_AR["cancels"],
                "count": cnt,
                "weight": float(cnt) / n,
            }
        )

    edges.sort(key=lambda e: e["count"], reverse=True)

    # Reserve slots per relation type so precedes volume cannot starve co/cancels
    def _take(rel: str, n: int) -> list[dict[str, Any]]:
        return [e for e in edges if e["relation"] == rel][:n]

    # Approximate split of top_k: ~35% co, ~45% precedes, ~20% cancels
    n_co = max(8, int(top_k * 0.35))
    n_prec = max(10, int(top_k * 0.45))
    n_cancel = max(4, int(top_k * 0.20))
    curated = _take("co_occurrence", n_co) + _take("precedes", n_prec) + _take("cancels", n_cancel)
    # Fill remaining budget with highest leftover edges
    used = {(e["source"], e["target"], e["relation"]) for e in curated}
    for e in edges:
        if len(curated) >= top_k:
            break
        key = (e["source"], e["target"], e["relation"])
        if key in used:
            continue
        curated.append(e)
        used.add(key)
    curated.sort(key=lambda e: e["count"], reverse=True)
    edges = curated[:top_k]

    sequences: list[dict[str, Any]] = []
    for e in edges:
        if e["relation"] != "precedes":
            continue
        a, b = e["source"], e["target"]
        base_b = max(int(mat[b].sum()), 1)
        lift = e["count"] / max(base_b * (lag_max / max(n, 1)), 1e-12)
        sequences.append(
            {
                "sequence": [a, b],
                "sequence_labels": [labels.get(a, a), labels.get(b, b)],
                "count": e["count"],
                "score": float(lift),
                "relation": "optimal_sequence",
                "relation_ar": RELATION_LABELS_AR["optimal_sequence"],
            }
        )
    sequences.sort(key=lambda s: s["score"], reverse=True)
    sequences = sequences[:30]

    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    # Keep nodes that appear in edges, plus top hubs by occurrence
    edge_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
    node_ids = list(edge_ids) if edge_ids else active[:24]
    for c in active:
        if c not in edge_ids and len(node_ids) < 40:
            node_ids.append(c)

    nodes = [
        {
            "id": c,
            "label": labels.get(c, c),
            "occurrences": int(mat[c].sum()),
            "bias": biases[c],
            "degree": int(degree.get(c, 0)),
        }
        for c in node_ids
    ]
    nodes.sort(key=lambda x: (x["degree"], x["occurrences"]), reverse=True)

    counts = {
        "co_occurrence": sum(1 for e in edges if e["relation"] == "co_occurrence"),
        "precedes": sum(1 for e in edges if e["relation"] == "precedes"),
        "cancels": sum(1 for e in edges if e["relation"] == "cancels"),
    }
    top_co = [e for e in edges if e["relation"] == "co_occurrence"][:5]
    top_prec = [e for e in edges if e["relation"] == "precedes"][:5]
    summary_parts = [
        f"عقد: {len(nodes)}",
        f"حواف: {len(edges)}",
        f"تزامن: {counts['co_occurrence']}",
        f"سبق: {counts['precedes']}",
        f"تعارض: {counts['cancels']}",
        f"تسلسلات: {len(sequences)}",
    ]
    if top_co:
        summary_parts.append(
            "أقوى تزامن: "
            + ", ".join(
                f"{e.get('source_label', e['source'])}+{e.get('target_label', e['target'])}({e['count']})"
                for e in top_co[:3]
            )
        )
    if top_prec:
        summary_parts.append(
            "أقوى سبق: "
            + ", ".join(
                f"{e.get('source_label', e['source'])}→{e.get('target_label', e['target'])}({e['count']})"
                for e in top_prec[:3]
            )
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "sequences": sequences,
        "summary": " · ".join(summary_parts),
        "counts": counts,
        "lag_max": lag_max,
        "bars": n,
        "active_patterns": len(active),
    }


def relations_has_graph(payload: dict[str, Any] | None) -> bool:
    """True when a relations payload contains usable graph edges."""
    if not payload:
        return False
    return bool(payload.get("edges"))
