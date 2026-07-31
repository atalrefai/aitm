"""Enterprise e4-v14 unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_pipeline_v14():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert (
        "e4-v14" in PIPELINE_VERSION
        or "e4-v15" in PIPELINE_VERSION
        or "e4-v16" in PIPELINE_VERSION
        or "enterprise" in PIPELINE_VERSION
        or "intelligent-training" in PIPELINE_VERSION
        or "research-factory" in PIPELINE_VERSION
    )


def test_data_intelligence_scores():
    from atis.engines.engine4_training.data_intelligence import analyze_training_frame

    n = 2000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "close": 2000 + np.cumsum(rng.normal(0, 1, n)),
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "is_outlier": rng.random(n) < 0.01,
            "session": rng.choice(["asian", "london", "new_york"], n),
        }
    )
    y = pd.Series(rng.choice([-1, 0, 1], n, p=[0.2, 0.6, 0.2]))
    rep = analyze_training_frame(df, y, ["feat_a", "feat_b"], timeframe="H1", cfg={})
    assert 0 <= rep["score"] <= 100
    assert "label_share" in rep


def test_feature_intelligence_selects():
    from atis.engines.engine4_training.feature_intelligence import analyze_and_select_features

    rng = np.random.default_rng(1)
    n = 800
    signal = rng.normal(0, 1, n)
    y = (signal > 0).astype(int) * 2 - 1
    X = pd.DataFrame(
        {
            "good": signal + rng.normal(0, 0.1, n),
            "noise": rng.normal(0, 1, n),
            "dup": signal + rng.normal(0, 0.05, n),
            "weak": rng.normal(0, 0.01, n),
        }
    )
    cols, rep = analyze_and_select_features(X, y, max_features=3, seed=1)
    assert rep["enabled"]
    assert len(cols) <= 3
    assert "good" in cols or "dup" in cols


def test_model_zoo_runs():
    from atis.engines.engine4_training.model_zoo import compare_model_zoo, map_winner_to_baseline

    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(0, 1, (n, 6))
    y = rng.choice([-1, 0, 1], n)
    meta = compare_model_zoo(X, y, None, seed=2, cfg={"model_zoo_enabled": True}, max_models=5)
    assert meta["enabled"]
    assert meta["winner"]
    assert map_winner_to_baseline(meta["winner"]) in {"lightgbm", "rf", "logistic", "ensemble"}


def test_stress_and_monte_carlo():
    from atis.engines.engine4_training.stress_testing import monte_carlo_trade_paths, stress_scenarios
    from atis.engines.engine4_training import financial_metrics

    r = np.zeros(300)
    r[::10] = 0.002
    r[5::10] = -0.001
    st = stress_scenarios(r, financial_fn=lambda x: financial_metrics(x, hold_bars=6))
    assert "scenarios" in st
    assert "base" in st["scenarios"]
    mc = monte_carlo_trade_paths(r[r != 0], n_paths=80, seed=0)
    assert mc["enabled"]
    assert 0 <= mc["p_profit"] <= 1


def test_apply_pending_overrides(tmp_path):
    from atis.engines.engine4_training.enterprise_report import apply_pending_overrides

    p = tmp_path / "knowledge_loop.json"
    p.write_text(
        '{"pending_overrides": {"lgb_max_depth": 3, "latency_bars": 1, "horizon_bars_delta": -1}}',
        encoding="utf-8",
    )
    cfg = {"horizon_bars": 6, "lgb_max_depth": 5}
    applied = apply_pending_overrides(cfg, p)
    assert cfg["lgb_max_depth"] == 3
    assert cfg["latency_bars"] == 1
    assert cfg["horizon_bars"] == 5
    assert "lgb_max_depth" in applied


def test_live_readiness_and_critique():
    from atis.engines.engine4_training.readiness import compute_live_readiness
    from atis.engines.engine4_training.enterprise_report import (
        build_intelligent_critique,
        propose_config_overrides,
    )

    metrics = {
        "financial_oos": {
            "sharpe": 2.0,
            "sharpe_ci_low": 1.0,
            "expectancy": 0.001,
            "n_trades": 80,
            "max_drawdown": -0.05,
        },
        "classification": {"roc_auc_ovr": 0.7, "accuracy": 0.65},
        "fit_diagnosis": {"status": "balanced", "sharpe_gap_val_test": 0.2},
        "regime_validation": {"stable": True},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 0.9}, "pbo": {"pbo": 0.3}},
        "stress_testing": {"robust": True, "worst_sharpe": 0.5, "scenarios": {"base": {}}},
        "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.7},
        "gate_failures": [],
    }
    ready = compute_live_readiness(passed_gates=True, metrics=metrics, timeframe="M5")
    assert ready["score"] >= 55
    assert ready["verdict"] in {"live_ready", "paper_ready", "research_only"}
    crit = build_intelligent_critique(metrics, timeframe="M5", passed=True)
    assert crit["root_cause"]
    opt = propose_config_overrides(
        timeframe="H4",
        metrics={
            **metrics,
            "classification": {"roc_auc_ovr": 0.48, "accuracy": 0.49},
            "gate_failures": ["weak_expectancy"],
            "live_readiness": ready,
        },
        passed_gates=False,
    )
    assert opt["overrides"]
