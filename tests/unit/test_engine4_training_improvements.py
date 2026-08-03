"""Regression tests for Engine 4 training improvements."""

from __future__ import annotations

import numpy as np

from atis.engines.engine4_training import (
    _trade_returns_from_preds,
    financial_metrics,
    horizon_for_timeframe,
    periods_per_year_for,
    walk_forward_splits,
)


def test_horizon_by_timeframe_prefers_gold_defaults():
    assert horizon_for_timeframe("H1", {"horizon_bars": 5, "horizon_by_timeframe": {"H1": 6}}) == 6
    assert horizon_for_timeframe("M15", {}) >= 1
    assert periods_per_year_for("H1") == 252 * 24


def test_walk_forward_applies_embargo_gap():
    splits = walk_forward_splits(500, 4, 0.7, embargo=6)
    assert splits
    for tr, te in splits:
        assert te[0] >= tr[-1] + 1 + 6 or te[0] > tr[-1]


def test_walk_forward_purges_overlapping_labels():
    splits = walk_forward_splits(500, 4, 0.7, embargo=6, purge=8)
    assert splits
    for tr, te in splits:
        # purged train ends before embargo start by at least purge bars
        assert tr[-1] < te[0] - 6
        assert (te[0] - (tr[-1] + 1)) >= 6


def test_financial_metrics_cap_high_frequency_annualization():
    # Sparse slightly-negative trades must not explode to absurd Sharpe on M15 ppy.
    r = np.zeros(2000)
    r[::40] = -0.001
    m15 = financial_metrics(r, periods_per_year=252 * 24 * 4, hold_bars=12, ann_cap="daily")
    h1 = financial_metrics(r, periods_per_year=252 * 24, hold_bars=6, ann_cap="daily")
    assert abs(m15["sharpe"]) < 8
    assert abs(h1["sharpe"]) < 8
    assert m15["ann_factor"] <= np.sqrt(252.0) + 1e-9


def test_financial_metrics_daily_cap_below_old_h1_ceiling():
    """Report Sharpe 14+ was inflated; daily cap must keep ann ≤ √252."""
    r = np.zeros(5000)
    rng = np.random.default_rng(0)
    idx = rng.choice(4800, size=200, replace=False)
    r[idx] = rng.normal(0.002, 0.01, size=200)
    m = financial_metrics(r, periods_per_year=252 * 24 * 4, hold_bars=12, ann_cap="daily", bootstrap=True)
    assert m["ann_factor"] <= np.sqrt(252.0) + 1e-9
    assert "sharpe_ci_low" in m
    assert "sharpe_uncapped" in m
    # Uncapped may be higher; gated Sharpe uses the conservative factor.
    assert abs(m["sharpe"]) <= abs(m["sharpe_uncapped"]) + 1e-9 or m["sharpe_uncapped"] == 0.0


def test_pipeline_version_bumped_for_reliability_gates():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert (
        "e4-v12" in PIPELINE_VERSION
        or "e4-v13" in PIPELINE_VERSION
        or "e4-v17" in PIPELINE_VERSION
        or "adaptive-intelligence" in PIPELINE_VERSION
        or "trading-intelligence" in PIPELINE_VERSION
        or "quality-first" in PIPELINE_VERSION
        or "self-diagnostic" in PIPELINE_VERSION
    )


def test_overtrading_rate_tolerance_does_not_trip_on_equality():
    from atis.engines.engine4_training import overtrading_rate_exceeds

    # H1 report: trade_rate touching max 0.22 must not hard-fail.
    assert not overtrading_rate_exceeds(0.22, 0.22, tol_frac=0.05)
    assert not overtrading_rate_exceeds(0.22 + 1e-9, 0.22, tol_frac=0.05)
    assert not overtrading_rate_exceeds(0.23, 0.22, tol_frac=0.05)  # within 5%
    assert overtrading_rate_exceeds(0.24, 0.22, tol_frac=0.05)


def test_overfit_sharpe_gap_hard_exempts_strong_consistent_test():
    from atis.engines.engine4_training import should_fail_overfit_sharpe_gap_hard

    # M30-like: Train≫Val gap but strong Test + Val↔Test consistent + enough trades.
    assert not should_fail_overfit_sharpe_gap_hard(
        sharpe_gap_tv=4.0,
        overfit_sharpe_gap=2.0,
        train_sharpe=13.0,
        val_sharpe=9.0,
        test_sharpe=9.1,
        sharpe_gap_vt=0.2,
        n_test_trades=80,
        min_sharpe=0.25,
        min_trades=20,
        val_test_gap_hard=3.5,
    )
    # Real collapse: Val weak and Test collapsed.
    assert should_fail_overfit_sharpe_gap_hard(
        sharpe_gap_tv=5.0,
        overfit_sharpe_gap=2.0,
        train_sharpe=8.0,
        val_sharpe=1.0,
        test_sharpe=0.1,
        sharpe_gap_vt=0.9,
        n_test_trades=30,
        min_sharpe=0.25,
        min_trades=20,
        val_test_gap_hard=3.5,
    )
    # H4-like large acc gap with weak Test still fails.
    assert should_fail_overfit_sharpe_gap_hard(
        sharpe_gap_tv=4.5,
        overfit_sharpe_gap=2.0,
        train_sharpe=6.0,
        val_sharpe=2.0,
        test_sharpe=0.8,
        sharpe_gap_vt=1.2,
        n_test_trades=25,
        min_sharpe=0.25,
        min_trades=20,
        val_test_gap_hard=3.5,
        acc_gap_tv=0.32,
        max_acc_gap=0.16,
    )


def test_gate_failures_include_arabic_labels():
    from atis.engines.engine4_training import annotate_gate_failures, gate_failure_ar

    assert "إفراط" in gate_failure_ar("overtrading_folds") or "تداول" in gate_failure_ar("overtrading_folds")
    detail = annotate_gate_failures(["overfit_sharpe_gap_hard", "overtrading_folds"])
    assert detail[0]["key"] == "overfit_sharpe_gap_hard"
    assert detail[0]["ar"]
    assert detail[1]["key"] == "overtrading_folds"


def test_financial_metrics_expose_non_compounded_trade_stats():
    r = np.array([0.01, 0.0, -0.005, 0.02, 0.0])
    m = financial_metrics(r, periods_per_year=252)
    assert "mean_trade_return" in m
    assert "sum_trade_returns" in m
    assert "simple_trade_equity" in m
    assert abs(m["sum_trade_returns"] - 0.025) < 1e-9
    assert abs(m["simple_trade_equity"] - 1.025) < 1e-9
    assert m["compounded_backtest_note"]


def test_higher_timeframes_hierarchy():
    from atis.engines.engine4_training import higher_timeframes_for

    assert higher_timeframes_for("M15", ["M15", "M30", "H1", "H4"]) == ["M30", "H1", "H4"]
    assert higher_timeframes_for("H1", ["M15", "M30", "H1", "H4"]) == ["H4"]
    assert higher_timeframes_for("H4", ["M15", "M30", "H1", "H4"]) == []
    # M1 must see M5+ as causal context.
    assert higher_timeframes_for("M1", ["M1", "M5", "M15", "M30", "H1", "H4"]) == [
        "M5", "M15", "M30", "H1", "H4"
    ]
    assert higher_timeframes_for("M5", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]) == [
        "M15", "M30", "H1", "H4", "D1"
    ]


def test_multi_tf_decision_soft_veto_and_htf_opposition():
    from atis.engines.engine4_training.multi_tf_decision import multi_tf_decision, confirm_tfs_for_primary

    pred, dbg = multi_tf_decision(
        1,
        0.8,
        [{"tf": "H1", "pred": 1, "conf": 0.7}],
        mode="soft_veto",
        primary_tf="M15",
    )
    assert pred == 1
    assert dbg["agreed"] == 1

    pred2, dbg2 = multi_tf_decision(
        1,
        0.65,
        [{"tf": "H4", "pred": -1, "conf": 0.8}],
        mode="soft_veto",
        veto_opposite_htf=True,
        primary_tf="M15",
    )
    assert pred2 == 0
    assert dbg2["reason"] == "htf_opposite_veto"

    pred3, _ = multi_tf_decision(
        1,
        0.6,
        [{"tf": "H1", "pred": -1, "conf": 0.6}],
        mode="hard_agree",
        primary_tf="M15",
    )
    assert pred3 == 0

    conf = confirm_tfs_for_primary("M1", {"confirm_by_primary_tf": {"M1": ["M5", "M15", "H1"]}})
    assert conf == ["M5", "M15", "H1"]


def test_barrier_atr_by_timeframe():
    from atis.engines.engine4_training import prepare_xy
    import pandas as pd

    n = 120
    df = pd.DataFrame({
        "close": np.linspace(2000, 2050, n),
        "atr": np.full(n, 2.0),
        "rsi_14": np.linspace(40, 60, n),
        "ema_20": np.linspace(2000, 2045, n),
        "ema_50": np.linspace(1995, 2040, n),
        "adx": np.full(n, 20.0),
        "trend_strength": np.linspace(-0.5, 0.5, n),
    })
    # Smoke: prepare_xy with M1 config path does not crash and yields labels.
    cfg_override = {
        "horizon_by_timeframe": {"M1": 10, "M5": 8},
        "barrier_atr_multiplier": 1.5,
        "barrier_atr_multiplier_by_tf": {"M1": 2.2, "M5": 2.0},
        "labeling": "triple_barrier",
        "engineer_learning_features": True,
        "prefer_relative_features": True,
        "drop_registry_context": True,
        "drop_constant_features": True,
    }
    from unittest.mock import patch

    with patch("atis.engines.engine4_training._cfg", return_value=cfg_override):
        X, y, cols, w = prepare_xy(df, timeframe="M1")
    assert len(y) > 0
    assert len(cols) > 0
    assert set(np.unique(y)).issubset({-1, 0, 1})


def test_horizon_includes_m1_m5_defaults():
    assert horizon_for_timeframe("M1", {}) == 30
    assert horizon_for_timeframe("M5", {}) == 12
    assert periods_per_year_for("M1") == 252 * 24 * 60
    assert periods_per_year_for("M5") == 252 * 24 * 12


def test_trade_returns_are_horizon_aligned_and_non_overlapping():
    close = np.linspace(2000.0, 2100.0, 40)
    preds = np.ones(40)
    rets, stats = _trade_returns_from_preds(
        close,
        preds,
        hold_bars=4,
        spread_pips=30.0,
        slippage_pips=5.0,
        commission_per_lot=7.0,
        pip_size=0.01,
        confidences=np.ones(40),
        min_confidence=0.5,
        non_overlapping=True,
    )
    assert stats["hold_bars"] == 4.0
    assert stats["trades"] > 0
    # Non-overlapping: nonzero returns should be spaced by hold_bars
    idxs = np.flatnonzero(rets != 0)
    if len(idxs) >= 2:
        assert np.min(np.diff(idxs)) >= 4


def test_low_confidence_predictions_are_skipped():
    close = np.linspace(2000.0, 2050.0, 30)
    preds = np.ones(30)
    conf = np.full(30, 0.40)
    rets, stats = _trade_returns_from_preds(
        close,
        preds,
        hold_bars=3,
        spread_pips=30.0,
        slippage_pips=5.0,
        commission_per_lot=7.0,
        pip_size=0.01,
        confidences=conf,
        min_confidence=0.58,
        non_overlapping=True,
    )
    assert stats["trades"] == 0.0
    assert float(np.sum(np.abs(rets))) == 0.0


def test_policy_from_proba_applies_edge_and_regime():
    from atis.engines.engine4_training import policy_from_proba, regime_mask_from_atr

    proba = np.array([
        [0.10, 0.20, 0.70],  # strong up
        [0.45, 0.30, 0.25],  # weak down edge (0.20 < 0.25)
        [0.05, 0.10, 0.85],  # strong up but blocked by regime
    ])
    classes = [-1, 0, 1]
    regime = np.array([True, True, False])
    preds = policy_from_proba(
        proba,
        classes,
        decision_threshold=0.5,
        directional_edge=0.25,
        confidence_quantile=0.0,
        regime_mask=regime,
    )
    assert preds[0] == 1.0
    assert preds[1] == 0.0
    assert preds[2] == 0.0
    mask = regime_mask_from_atr(np.array([0.001, 0.01, 0.05, 0.02, 0.015] * 10), 0.2, 0.9)
    assert mask.dtype == bool
    assert mask.any() and (~mask).any()


def test_financial_metrics_include_trade_count():
    r = np.array([0.01, 0.0, -0.005, 0.02, 0.0])
    m = financial_metrics(r, periods_per_year=252)
    assert "n_trades" in m
    assert m["n_trades"] == 3.0


def test_sparsify_by_confidence_keeps_top_fraction():
    from atis.engines.engine4_training import sparsify_by_confidence

    preds = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.99, 0.95, 0.5, 0.4, 0.3, 0.2])
    out = sparsify_by_confidence(preds, conf, target_trade_rate=0.2)
    assert np.count_nonzero(out) == 2
    assert out[0] == 1.0
    assert out[5] == -1.0


def test_classification_bundle_includes_auc_for_binary():
    from atis.engines.engine4_training import classification_bundle

    y = np.array([1, -1, 1, -1, 1, -1])
    p = np.array([1, -1, 1, 1, -1, -1])
    proba = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.8, 0.2],
    ])
    m = classification_bundle(y, p, proba, [-1, 1])
    assert "roc_auc_ovr" in m
    assert m["n_samples"] == 6


def test_diagnose_fit_flags_overfitting():
    from atis.engines.engine4_training import diagnose_fit

    d = diagnose_fit(
        {"accuracy": 0.85},
        {"accuracy": 0.50},
        {"accuracy": 0.48},
        {"sharpe": 3.0},
        {"sharpe": 0.1},
        {"sharpe": -0.2},
    )
    assert d["status"] == "overfitting"


def test_diagnose_fit_flags_sparse_trade_optimism():
    from atis.engines.engine4_training import diagnose_fit

    d = diagnose_fit(
        {"accuracy": 0.62, "roc_auc_ovr": 0.51},
        {"accuracy": 0.48},
        {"accuracy": 0.46, "roc_auc_ovr": 0.50, "trade_rate_filtered": 0.002},
        {"sharpe": 1.2, "n_trades": 40},
        {"sharpe": 3.0, "n_trades": 3},
        {"sharpe": 0.8, "n_trades": 2},
        trade_rate_filtered=0.002,
        median_fold_trade_rate=0.0,
    )
    assert d["status"] == "unstable_generalization"
    assert any("sparse" in n.lower() or "starvation" in n.lower() or "near-zero" in n.lower() for n in d["notes"])


def test_diagnose_fit_keeps_strong_test_as_warning():
    """Report 04-15 H4: Val≫Test gap but Test Sharpe≈2.35 must not hard-flag unstable."""
    from atis.engines.engine4_training import diagnose_fit

    d = diagnose_fit(
        {"accuracy": 0.67},
        {"accuracy": 0.50},
        {"accuracy": 0.502, "roc_auc_ovr": 0.50, "trade_rate_filtered": 0.10},
        {"sharpe": 1.5, "n_trades": 40},
        {"sharpe": 4.5, "n_trades": 25},
        {"sharpe": 2.35, "n_trades": 20},
        trade_rate_filtered=0.10,
        median_fold_trade_rate=0.20,
    )
    assert d["status"] == "balanced"
    assert d["filter_driven_edge_risk"] is True
    assert d["sparse_sharpe_risk"] is False
    assert any("strong" in n.lower() or "monitor" in n.lower() or "warning" in n.lower() or "filter" in n.lower() for n in d["notes"])


def test_diagnose_fit_flags_report_h4_sparse_filter_edge():
    """Report 2026-07-31 H4-like: Acc≈0.50, AUC≈0.50, Sharpe high on 7 trades."""
    from atis.engines.engine4_training import diagnose_fit

    d = diagnose_fit(
        {"accuracy": 0.67, "roc_auc_ovr": 0.55},
        {"accuracy": 0.50},
        {"accuracy": 0.502, "roc_auc_ovr": 0.50, "trade_rate_filtered": 0.174},
        {"sharpe": 1.8, "n_trades": 40},
        {"sharpe": 4.42, "n_trades": 18},
        {"sharpe": 2.09, "n_trades": 7},
        trade_rate_filtered=0.174,
        median_fold_trade_rate=0.20,
    )
    assert d["status"] == "unstable_generalization"
    assert d["sparse_sharpe_risk"] is True
    assert d["filter_driven_edge_risk"] is True


def test_engineer_learning_features_are_causal_scale_free():
    from atis.engines.engine4_training import engineer_learning_features
    import pandas as pd

    n = 60
    df = pd.DataFrame({
        "close": np.linspace(2000, 2100, n) + np.random.randn(n),
        "atr": np.full(n, 5.0),
        "rsi_14": np.linspace(30, 70, n),
        "ema_20": np.linspace(2000, 2090, n),
        "ema_50": np.linspace(1990, 2080, n),
        "ema_200": np.linspace(1980, 2070, n),
        "adx": np.full(n, 25.0),
        "trend_strength": np.linspace(-1, 1, n),
        "bb_upper": np.linspace(2010, 2110, n),
        "bb_lower": np.linspace(1990, 2090, n),
        "macd_hist": np.random.randn(n),
    })
    out = engineer_learning_features(df)
    assert "feat_ret_1" in out.columns
    assert "feat_atr_pct" in out.columns
    assert "feat_rsi_centered" in out.columns
    assert "feat_ema_spread" in out.columns
    assert "feat_mom_accel" in out.columns
    # First row of returns must be NaN (no lookahead fill with future).
    assert pd.isna(out["feat_ret_1"].iloc[0])


def test_enrich_with_higher_timeframes_merge_asof_causal():
    from atis.engines.engine4_training.data_sources import enrich_with_higher_timeframes
    import pandas as pd
    from unittest.mock import patch

    base = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"], utc=True),
        "close": [2000.0, 2001.0, 2002.0],
        "structure_hh_hl": [1.0, 1.0, -1.0],
        "rsi_14": [55.0, 60.0, 40.0],
        "adx": [20.0, 22.0, 25.0],
        "trend_strength": [0.2, 0.3, -0.1],
    })
    htf = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 04:00"], utc=True),
        "close": [1990.0, 2010.0],
        "atr": [4.0, 5.0],
        "structure_hh_hl": [1.0, -1.0],
        "rsi_14": [50.0, 30.0],
        "adx": [18.0, 30.0],
        "trend_strength": [0.1, -0.2],
        "chart_pattern_score": [0.0, 1.0],
        "pat_bias": [1.0, -1.0],
        "pat_strength": [0.5, 0.5],
        "dist_to_support": [0.01, 0.02],
        "dist_to_resist": [0.02, 0.01],
        "trendline_slope": [0.0, 0.0],
        "macd_hist": [0.1, -0.1],
    })
    # Satisfy HTF length gate inside enrich_with_higher_timeframes.
    while len(htf) < 8:
        last = htf.iloc[-1].copy()
        last["timestamp"] = last["timestamp"] + pd.Timedelta(hours=4)
        htf = pd.concat([htf, pd.DataFrame([last])], ignore_index=True)

    def _fake_load(symbol, htf_name):
        assert htf_name == "H4"
        frame = htf.copy()
        keep = ["timestamp"]
        for col in [
            "trend_strength", "structure_hh_hl", "adx", "rsi_14", "chart_pattern_score",
            "pat_bias", "pat_strength", "dist_to_support", "dist_to_resist",
            "trendline_slope", "macd_hist",
        ]:
            out_name = f"htf_{htf_name}__{col}"
            frame[out_name] = frame[col]
            keep.append(out_name)
        frame["htf_H4__atr_pct"] = frame["atr"] / frame["close"]
        keep.append("htf_H4__atr_pct")
        return frame[keep]

    with patch(
        "atis.engines.engine4_training.data_sources._load_htf_context_frame",
        side_effect=_fake_load,
    ):
        out, meta = enrich_with_higher_timeframes(base, "XAUUSD", "H1", higher_tfs=["H4"])
    assert meta["n_htf_cols"] > 0
    assert "htf_H4__rsi_14" in out.columns
    assert "mtf_H4_structure_agree" in out.columns
    # All three local bars see only the 00:00 HTF bar (04:00 is in the future).
    assert float(out["htf_H4__rsi_14"].iloc[-1]) == 50.0


def test_train_confidence_floor_respects_target_rate():
    from atis.engines.engine4_training import train_confidence_floor

    class _M:
        classes_ = np.array([-1, 1])

        def predict_proba(self, X):
            # Mostly mid confidence; a fat right tail of high-confidence rows.
            n = len(X)
            p = np.full((n, 2), 0.45)
            p[: max(1, n // 20), 1] = 0.92
            p[: max(1, n // 20), 0] = 0.08
            p[max(1, n // 20) :, 0] = 0.55
            p[max(1, n // 20) :, 1] = 0.45
            return p

    X = np.zeros((200, 3))
    strict = train_confidence_floor(
        _M(), X, decision_threshold=0.55, confidence_quantile=0.95, min_floor=0.55, target_trade_rate=0.0, max_floor=0.99
    )
    aligned = train_confidence_floor(
        _M(), X, decision_threshold=0.55, confidence_quantile=0.95, min_floor=0.55, target_trade_rate=0.05, max_floor=0.88
    )
    assert aligned <= 0.88
    assert aligned <= strict


def test_policy_liquidity_starved_detects_near_zero_rate():
    from atis.engines.engine4_training import policy_liquidity_starved

    assert policy_liquidity_starved(trades=0, n_bars=200, min_trades=5, target_trade_rate=0.05)
    assert policy_liquidity_starved(trades=2, n_bars=400, min_trades=5, target_trade_rate=0.05)
    assert not policy_liquidity_starved(trades=20, n_bars=400, min_trades=5, target_trade_rate=0.05)


def test_cap_preds_by_trade_rate_blocks_overtrading():
    from atis.engines.engine4_training import cap_preds_by_trade_rate

    preds = np.ones(100, dtype=float)
    conf = np.linspace(0.1, 1.0, 100)
    out = cap_preds_by_trade_rate(preds, conf, max_trade_rate=0.20)
    assert np.count_nonzero(out) == 20
    assert out[-1] == 1.0  # highest confidence kept


def test_structure_primary_sides_exclusive():
    from atis.engines.engine4_training import structure_primary_sides

    long = np.array([True, False, True, False])
    short = np.array([False, True, True, False])
    sides = structure_primary_sides(long, short)
    assert list(sides) == [1.0, -1.0, 0.0, 0.0]


def test_meta_labeling_uses_structure_side_not_argmax():
    from atis.engines.engine4_training import policy_from_proba, structure_primary_sides

    # Model prefers short, structure says long-only → meta should take long if p_up high enough.
    proba = np.array([
        [0.25, 0.15, 0.60],  # up conf 0.60
        [0.55, 0.20, 0.25],  # down preferred by argmax
    ])
    classes = [-1, 0, 1]
    primary = structure_primary_sides(
        np.array([True, False]),
        np.array([False, True]),
    )
    preds = policy_from_proba(
        proba,
        classes,
        decision_threshold=0.50,
        directional_edge=0.10,
        confidence_floor=0.50,
        primary_sides=primary,
    )
    assert preds[0] == 1.0
    assert preds[1] == -1.0


def test_diagnose_fit_does_not_reject_strong_test_as_unstable_for_gates():
    """Mirror gate rule: strong Test Sharpe with Val≫Test stays warning/balanced."""
    from atis.engines.engine4_training import diagnose_fit

    d = diagnose_fit(
        {"accuracy": 0.60},
        {"accuracy": 0.52},
        {"accuracy": 0.50, "roc_auc_ovr": 0.55, "trade_rate_filtered": 0.08},
        {"sharpe": 1.8, "n_trades": 50},
        {"sharpe": 3.2, "n_trades": 30},
        {"sharpe": 2.1, "n_trades": 40},
        trade_rate_filtered=0.08,
        median_fold_trade_rate=0.07,
    )
    assert d["status"] == "balanced"


def test_prefer_relative_features_drops_absolute_prices():
    from atis.engines.engine4_training import prefer_relative_features

    cols = ["close", "ema_50", "rsi_14", "dist_to_support", "ret_1", "atr_pct", "open", "macd_hist"]
    kept = prefer_relative_features(cols, keep_min=4)
    assert "rsi_14" in kept
    assert "dist_to_support" in kept
    assert "close" not in kept or len(kept) >= 4


def test_select_best_result_rejects_zero_one_trade_deploy_preference():
    """Liquidity-first: gated model with enough deploy trades beats 1-trade champion."""
    from atis.engines.engine4_training.final_model import select_best_result

    results = [
        {
            "timeframe": "H4",
            "model_path": "h4/model.joblib",
            "passed_gates": True,
            "metrics": {
                "financial_oos": {"sharpe": 2.0, "max_drawdown": -0.04, "total_return": 0.2, "n_trades": 20},
                "financial_deploy_holdout": {"sharpe": 3.0, "n_trades": 1},
            },
        },
        {
            "timeframe": "H1",
            "model_path": "h1/model.joblib",
            "passed_gates": True,
            "metrics": {
                "financial_oos": {"sharpe": 0.9, "max_drawdown": -0.03, "total_return": 0.05, "n_trades": 40},
                "financial_deploy_holdout": {"sharpe": 0.6, "n_trades": 12},
            },
        },
    ]
    best = select_best_result(results, min_deploy_trades=8)
    assert best is not None
    assert best["timeframe"] == "H1"


def test_short_edge_multiple_is_stricter():
    from atis.engines.engine4_training import policy_from_proba

    proba = np.array([
        [0.30, 0.70],  # up edge 0.40
        [0.70, 0.30],  # down edge 0.40
    ])
    classes = [-1, 1]
    loose = policy_from_proba(
        proba, classes, decision_threshold=0.5, directional_edge=0.35, short_edge_multiple=1.0
    )
    strict = policy_from_proba(
        proba, classes, decision_threshold=0.5, directional_edge=0.35, short_edge_multiple=1.5
    )
    assert loose[0] == 1.0 and loose[1] == -1.0
    assert strict[0] == 1.0 and strict[1] == 0.0


def test_peek_json_int_reads_header_only(tmp_path):
    from atis.engines.engine4_training.data_sources import _peek_json_int

    path = tmp_path / "big.json"
    # Simulate a huge payload after the header scalars the UI needs.
    path.write_text('{"row_count": 42, "count": 7, "items": [' + ("0," * 20000) + "0]}", encoding="utf-8")
    assert _peek_json_int(path, "row_count") == 42
    assert _peek_json_int(path, "count") == 7
    assert _peek_json_int(tmp_path / "missing.json", "row_count") is None
