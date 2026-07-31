# Appended by implementation — adaptive intelligence / DQ / fold liquidity tests

from __future__ import annotations

import numpy as np


def test_dq_gate_fails_insufficient_h4_sample():
    import pandas as pd
    from atis.engines.engine4_training.data_quality_gate import compute_data_quality_score

    n = 80
    df = pd.DataFrame(
        {
            "close": np.linspace(2000, 2010, n),
            "feat_a": np.random.randn(n),
            "is_outlier": np.zeros(n, dtype=bool),
        }
    )
    y = pd.Series(np.zeros(n, dtype=int))
    report = compute_data_quality_score(
        df,
        y,
        ["feat_a"],
        timeframe="H4",
        cfg={
            "dq_gate_enabled": True,
            "dq_gate_hard": True,
            "walk_forward_splits": 5,
            "fold_validation_ratio": 0.25,
        },
    )
    assert report["gate_pass"] is False
    assert report["skip_reason"]


def test_dq_gate_passes_liquid_m5_sample():
    import pandas as pd
    from atis.engines.engine4_training.data_quality_gate import compute_data_quality_score

    n = 5000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "close": 2000 + np.cumsum(rng.normal(0, 1, n)),
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "is_outlier": rng.random(n) < 0.01,
        }
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.22, 0.56, 0.22]))
    report = compute_data_quality_score(
        df,
        y,
        ["feat_a", "feat_b"],
        timeframe="M5",
        cfg={
            "dq_gate_enabled": True,
            "dq_gate_hard": True,
            "walk_forward_splits": 5,
            "fold_validation_ratio": 0.25,
        },
    )
    assert report["gate_pass"] is True
    assert report["score"] >= 55


def test_starved_folds_excluded_from_best_selection():
    from atis.engines.engine4_training.adaptive_learning import (
        fold_eligible_for_selection,
        select_best_liquid_fold,
    )

    cfg = {"min_val_trades_by_tf": {"H4": 8}}
    folds = [
        {"fold": 0, "val_sharpe": 4.5, "n_val_trades": 0, "n_validation": 40},
        {"fold": 1, "val_sharpe": 3.2, "n_val_trades": 6, "n_validation": 40},
        {"fold": 2, "val_sharpe": 2.8, "n_val_trades": 7, "n_validation": 40},
        {"fold": 3, "val_sharpe": 1.1, "n_val_trades": 7, "n_validation": 40},
        {"fold": 4, "val_sharpe": 0.9, "n_val_trades": 22, "n_validation": 40},
    ]
    assert not fold_eligible_for_selection(
        n_val_trades=6, val_sharpe=3.2, timeframe="H4", cfg=cfg, n_val_bars=40
    )
    best_i, _score, row = select_best_liquid_fold(folds, timeframe="H4", cfg=cfg)
    assert best_i == 4
    assert float(row["n_val_trades"]) >= 8


def test_policy_consensus_requires_liquid_folds():
    from atis.engines.engine4_training.adaptive_learning import policy_consensus_ok

    folds = [
        {
            "fold": i,
            "val_sharpe": 1.0,
            "n_val_trades": 3,
            "n_validation": 50,
            "policy": {"decision_threshold": 0.55, "directional_edge": 0.15},
        }
        for i in range(5)
    ]
    ok, meta = policy_consensus_ok(
        folds,
        timeframe="M5",
        cfg={"min_val_trades_by_tf": {"M5": 20}, "policy_min_agree_folds": 3},
    )
    assert ok is False
    assert meta["n_liquid_folds"] == 0


def test_intelligence_plan_targets_h4_labeling_or_liquidity():
    from atis.engines.engine4_training.intelligence import critique_training_run

    results = [
        {
            "timeframe": "M5",
            "passed_gates": True,
            "metrics": {
                "fit_diagnosis": {"status": "balanced", "sharpe_gap_val_test": 0.1},
                "classification": {"accuracy": 0.76, "roc_auc_ovr": 0.84},
                "financial_oos": {
                    "sharpe": 8.0,
                    "sharpe_ci_low": 2.0,
                    "max_drawdown": -0.04,
                    "n_trades": 80,
                },
                "financial_deploy_holdout": {"n_trades": 20, "sharpe": 3.0},
                "gate_failures": [],
            },
        },
        {
            "timeframe": "H4",
            "passed_gates": False,
            "metrics": {
                "fit_diagnosis": {"status": "unstable_generalization", "sharpe_gap_val_test": 2.5},
                "classification": {"accuracy": 0.496, "roc_auc_ovr": 0.444},
                "financial_oos": {
                    "sharpe": -1.0,
                    "sharpe_ci_low": -2.0,
                    "max_drawdown": -0.12,
                    "n_trades": 9,
                },
                "financial_deploy_holdout": {"n_trades": 6, "sharpe": -3.0},
                "gate_failures": ["val_fold_liquidity", "filter_driven_edge"],
                "data_quality": {"score": 48, "gate_pass": True},
            },
        },
        {
            "timeframe": "H1",
            "passed_gates": True,
            "metrics": {
                "fit_diagnosis": {
                    "status": "overfitting",
                    "sharpe_gap_train_val": 2.5,
                    "sharpe_gap_val_test": 1.8,
                },
                "classification": {"accuracy": 0.75, "roc_auc_ovr": 0.82},
                "financial_oos": {
                    "sharpe": 7.9,
                    "sharpe_ci_low": 1.2,
                    "max_drawdown": -0.05,
                    "n_trades": 60,
                },
                "financial_deploy_holdout": {"n_trades": 15, "sharpe": 2.0},
                "gate_failures": [],
            },
        },
    ]
    plan = critique_training_run(results, models_root=None, run_id="test")
    nxt = plan["next_experiment"]
    assert nxt["timeframe"] == "H4"
    assert nxt["root_cause"] in {"Labeling", "Liquidity/Sample", "Features"}


def test_final_model_prefers_balanced_over_overfitting():
    from atis.engines.engine4_training.final_model import select_best_result

    results = [
        {
            "timeframe": "H1",
            "model_path": "h1/model.joblib",
            "passed_gates": True,
            "metrics": {
                "fit_diagnosis": {"status": "overfitting"},
                "classification": {"roc_auc_ovr": 0.82},
                "financial_oos": {
                    "sharpe": 9.5,
                    "max_drawdown": -0.03,
                    "total_return": 0.3,
                    "n_trades": 80,
                },
                "financial_deploy_holdout": {"sharpe": 4.0, "n_trades": 20},
            },
        },
        {
            "timeframe": "M5",
            "model_path": "m5/model.joblib",
            "passed_gates": True,
            "metrics": {
                "fit_diagnosis": {"status": "balanced"},
                "classification": {"roc_auc_ovr": 0.84},
                "financial_oos": {
                    "sharpe": 8.4,
                    "max_drawdown": -0.04,
                    "total_return": 0.25,
                    "n_trades": 100,
                },
                "financial_deploy_holdout": {"sharpe": 3.5, "n_trades": 25},
            },
        },
    ]
    best = select_best_result(results, min_deploy_trades=8)
    assert best["timeframe"] == "M5"


def test_failed_h4_not_selected_as_champion_over_m5():
    from atis.engines.engine4_training.final_model import select_best_result

    results = [
        {
            "timeframe": "H4",
            "model_path": "h4/model.joblib",
            "passed_gates": False,
            "metrics": {
                "fit_diagnosis": {"status": "unstable_generalization"},
                "classification": {"roc_auc_ovr": 0.44},
                "financial_oos": {
                    "sharpe": 2.0,
                    "max_drawdown": -0.1,
                    "total_return": 0.01,
                    "n_trades": 7,
                },
                "financial_deploy_holdout": {"sharpe": 3.0, "n_trades": 1},
            },
        },
        {
            "timeframe": "M5",
            "model_path": "m5/model.joblib",
            "passed_gates": True,
            "metrics": {
                "fit_diagnosis": {"status": "balanced"},
                "classification": {"roc_auc_ovr": 0.84},
                "financial_oos": {
                    "sharpe": 8.0,
                    "max_drawdown": -0.04,
                    "total_return": 0.2,
                    "n_trades": 90,
                },
                "financial_deploy_holdout": {"sharpe": 3.0, "n_trades": 20},
            },
        },
    ]
    assert select_best_result(results, min_deploy_trades=8)["timeframe"] == "M5"


def test_retrain_interval_and_iterative_stop():
    from datetime import datetime, timedelta, timezone

    from atis.engines.engine4_training.adaptive_learning import (
        iterative_stop_decision,
        should_trigger_retrain,
    )

    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    old = (now - timedelta(days=10)).isoformat()
    trig, reason = should_trigger_retrain(
        last_train_utc=old, retrain_interval_days=7, now=now
    )
    assert trig and reason == "retrain_interval"
    stop, why = iterative_stop_decision(
        [{"sharpe_ci_low": 1.6, "fit_status": "balanced"}], kpi_ci_low=1.5
    )
    assert stop and why == "kpi_reached"


def test_dynamic_execution_costs_scale_with_vol():
    from atis.engines.engine4_training.adaptive_learning import dynamic_execution_costs

    sp1, sl1, _ = dynamic_execution_costs(
        2000.0,
        0.001,
        base_spread_pips=30,
        base_slippage_pips=5,
        commission_per_lot=7,
        pip_size=0.01,
    )
    sp2, sl2, _ = dynamic_execution_costs(
        2000.0,
        0.005,
        base_spread_pips=30,
        base_slippage_pips=5,
        commission_per_lot=7,
        pip_size=0.01,
    )
    assert sp2 >= sp1 and sl2 >= sl1


def test_nested_hp_search_runs_on_train_only():
    from atis.engines.engine4_training.adaptive_learning import nested_hyperparameter_search

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = rng.choice([-1, 1], size=200)
    cfg = {
        "nested_hp_search": True,
        "nested_hp_trials": 3,
        "nested_hp_compare_baselines": True,
    }
    _best_cfg, meta = nested_hyperparameter_search(
        X, y, None, base_cfg=cfg, timeframe="H1", seed=0, n_trials=3
    )
    assert meta.get("enabled") is True
    assert meta.get("trials", 0) >= 1
