"""Pattern relation graph: co-occurrence, precedes, cancels, optimal sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def build_pattern_relations(
    df: pd.DataFrame,
    pattern_cols: list[str],
    *,
    lag_max: int = 5,
    min_count: int = 5,
    top_k: int = 80,
) -> dict[str, Any]:
    """
    Build a lightweight relation graph from binary pattern columns (causal lags).

    Relations:
      - co_occurrence: same bar
      - precedes: A at t, B at t+1..t+lag (A precedes B)
      - cancels: opposite bias patterns within lag of each other
    """
    cols = [c for c in pattern_cols if c in df.columns]
    if len(cols) < 2:
        return {"nodes": [], "edges": [], "sequences": [], "summary": "لا علاقات كافية"}

    mat = {c: df[c].fillna(0).astype(int).to_numpy() for c in cols}
    n = len(df)

    # Bias map from column name heuristics
    def _bias(col: str) -> str:
        name = col.lower()
        bear = ("bear", "top", "shooting", "crows", "dark", "down", "sweep_high")
        bull = ("bull", "bottom", "hammer", "soldiers", "piercing", "up", "sweep_low")
        if any(x in name for x in bear):
            return "bearish"
        if any(x in name for x in bull):
            return "bullish"
        return "neutral"

    biases = {c: _bias(c) for c in cols}
    active = [c for c in cols if int(mat[c].sum()) >= min_count]
    active = active[:60]  # cap for memory

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
        # precedes within lag
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
                "relation": "co_occurrence",
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
                "relation": "precedes",
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
                "relation": "cancels",
                "count": cnt,
                "weight": float(cnt) / n,
            }
        )

    edges.sort(key=lambda e: e["count"], reverse=True)
    edges = edges[:top_k]

    # Optimal sequences: top precedes chains of length 2 (A→B) by lift-ish score
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
                "count": e["count"],
                "score": float(lift),
                "relation": "optimal_sequence",
            }
        )
    sequences.sort(key=lambda s: s["score"], reverse=True)
    sequences = sequences[:30]

    nodes = [
        {
            "id": c,
            "occurrences": int(mat[c].sum()),
            "bias": biases[c],
        }
        for c in active
    ]

    top_co = [e for e in edges if e["relation"] == "co_occurrence"][:5]
    top_prec = [e for e in edges if e["relation"] == "precedes"][:5]
    summary_parts = [
        f"عقد: {len(nodes)}",
        f"حواف: {len(edges)}",
        f"تسلسلات: {len(sequences)}",
    ]
    if top_co:
        summary_parts.append(
            "أقوى تزامن: " + ", ".join(f"{e['source']}+{e['target']}({e['count']})" for e in top_co[:3])
        )
    if top_prec:
        summary_parts.append(
            "أقوى سبق: " + ", ".join(f"{e['source']}→{e['target']}({e['count']})" for e in top_prec[:3])
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "sequences": sequences,
        "summary": " · ".join(summary_parts),
        "lag_max": lag_max,
        "bars": n,
    }
