"""Pattern relation graph → training feature injection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from atis.engines.engine4_training.data_sources import inject_pattern_relation_features
from atis.shared.pattern_discovery.relations import build_pattern_relations


def _frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "close": 2000 + np.cumsum(rng.normal(0, 1, n)),
            "pat_hammer": 0,
            "pat_engulfing_bull": 0,
            "pat_shooting_star": 0,
            "pat_marubozu_bear": 0,
        }
    )
    df.loc[10:40, "pat_hammer"] = 1
    df.loc[12:42, "pat_engulfing_bull"] = 1
    df.loc[20:50, "pat_shooting_star"] = 1
    df.loc[22:45, "pat_marubozu_bear"] = 1
    return df


def test_inject_relation_features_creates_feat_rel_columns() -> None:
    df = _frame()
    graph = build_pattern_relations(
        df,
        ["pat_hammer", "pat_engulfing_bull", "pat_shooting_star", "pat_marubozu_bear"],
        min_count=3,
    )
    assert graph["edges"]
    out, meta = inject_pattern_relation_features(
        df.copy(), graph, top_co=5, top_prec=8, top_cancel=4, top_seq=4
    )
    assert meta["enabled"] is True
    assert meta["n_injected"] >= 5
    rel_cols = [c for c in out.columns if str(c).startswith("feat_rel_")]
    assert "feat_rel_net_confirm" in rel_cols
    assert "feat_rel_hub_activity" in rel_cols
    assert "feat_rel_graph_active" in rel_cols
    prec = [
        c
        for c in rel_cols
        if c.startswith("feat_rel_prec_") and not c.startswith("feat_rel_prec_w_")
    ]
    assert prec
    # First bar has no past window after shift → precede hit must be 0
    assert float(out[prec[0]].iloc[0]) == 0.0
    assert out[rel_cols].isna().sum().sum() == 0


def test_inject_relation_features_empty_graph() -> None:
    df = _frame(30)
    out, meta = inject_pattern_relation_features(df.copy(), {"edges": [], "nodes": []})
    assert meta["enabled"] is False
    assert [c for c in out.columns if str(c).startswith("feat_rel_")] == []
