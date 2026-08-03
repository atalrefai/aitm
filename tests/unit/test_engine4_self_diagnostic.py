"""Unit tests for Engine-4 Self-Diagnostic + anti-inflation layer (v17.2)."""

from __future__ import annotations

import json
from pathlib import Path


def test_pipeline_version_self_diagnostic():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert "v17" in PIPELINE_VERSION
    assert (
        "self-diagnostic" in PIPELINE_VERSION
        or "weakness-hardening" in PIPELINE_VERSION
        or "quality-first" in PIPELINE_VERSION
    )

def test_saturation_detection_pegged_to_cap():
    from atis.engines.engine4_training.promotion_v16 import fold_stability_report

    folds = [
        {"n_val_trades": 40, "policy": {"val_sharpe": 2.0}, "trade_rate": 0.12, "accuracy": 0.7, "test_sharpe": 1.0},
        {"n_val_trades": 35, "policy": {"val_sharpe": 2.1}, "trade_rate": 0.119, "accuracy": 0.71, "test_sharpe": 1.1},
        {"n_val_trades": 30, "policy": {"val_sharpe": 1.9}, "trade_rate": 0.118, "accuracy": 0.69, "test_sharpe": 0.9},
        {"n_val_trades": 30, "policy": {"val_sharpe": 2.2}, "trade_rate": 0.12, "accuracy": 0.72, "test_sharpe": 1.2},
        {"n_val_trades": 30, "policy": {"val_sharpe": 2.0}, "trade_rate": 0.121, "accuracy": 0.70, "test_sharpe": 1.0},
    ]
    st = fold_stability_report(folds, cfg={"max_fold_trade_rate": 0.12, "fail_on_fold_unstable": True})
    assert st["trade_rate_pegged"] is True


def test_inflated_sharpe_gates_m1_style():
    from atis.engines.engine4_training.promotion_v16 import inflated_sharpe_report

    # M1-style: uncapped 38, capped 9, weak trade_raw — must still fail
    infl = inflated_sharpe_report(
        {"sharpe": 9.06, "sharpe_uncapped": 38.35, "trade_sharpe_raw": 0.08},
        cfg={
            "fail_on_inflated_sharpe": True,
            "max_sharpe_uncapped": 15.0,
            "max_uncapped_to_capped_ratio": 2.5,
            "max_path_vs_trade_gap": 0.12,
            "min_trade_sharpe_raw": 0.05,
            "enforce_min_trade_sharpe_raw": True,
            "uncapped_rescue_min_trade_sharpe": 0.35,
        },
    )
    assert infl["inflated"] is True
    assert infl["gate_pass"] is False
    assert "uncapped_above_max" in infl["reasons"]

    # Path vs trade gap
    infl2 = inflated_sharpe_report(
        {"sharpe": 7.0, "sharpe_uncapped": 10.0, "trade_sharpe_raw": 0.02},
        cfg={"fail_on_inflated_sharpe": True, "max_sharpe_uncapped": 20.0, "max_path_vs_trade_gap": 0.12},
    )
    assert infl2["inflated"] is True
    assert infl2["path_vs_trade_gap"] is True

    # Sharpe↔AUC mismatch
    infl3 = inflated_sharpe_report(
        {"sharpe": 5.0, "sharpe_uncapped": 5.0, "trade_sharpe_raw": 0.3},
        cfg={"fail_on_inflated_sharpe": True, "max_sharpe_uncapped": 20.0},
        auc=0.51,
    )
    assert infl3["inflated"] is True
    assert infl3["sharpe_auc_mismatch"] is True


def test_inflated_sharpe_m5_trade_supported_rescue():
    """M5 fingerprint: uncapped≈22 but trade_raw≈0.71 and aligned Val/Test — must pass."""
    from atis.engines.engine4_training.promotion_v16 import inflated_sharpe_report

    infl = inflated_sharpe_report(
        {"sharpe": 11.26, "sharpe_uncapped": 22.49, "trade_sharpe_raw": 0.7095},
        cfg={
            "fail_on_inflated_sharpe": True,
            "max_sharpe_uncapped": 12.0,
            "max_uncapped_to_capped_ratio": 2.2,
            "uncapped_rescue_min_trade_sharpe": 0.35,
            "max_uncapped_ratio_with_trade_rescue": 4.0,
            "max_path_vs_trade_gap": 0.10,
        },
        auc=0.865,
    )
    assert infl["inflated"] is False
    assert infl["gate_pass"] is True
    assert infl["trade_supported_uncapped"] is True
    assert "uncapped_high_but_trade_supported" in infl["notes"]


def test_inflated_sharpe_m1_trade_supported_despite_high_ratio():
    """M1 fingerprint: uncapped≈38, ratio≈3.07, but trade_raw≈0.78 — mechanical HF, must pass."""
    from atis.engines.engine4_training.promotion_v16 import inflated_sharpe_report

    infl = inflated_sharpe_report(
        {"sharpe": 12.38, "sharpe_uncapped": 38.06, "trade_sharpe_raw": 0.7796},
        cfg={
            "fail_on_inflated_sharpe": True,
            "max_sharpe_uncapped": 12.0,
            "max_uncapped_to_capped_ratio": 2.2,
            "uncapped_rescue_min_trade_sharpe": 0.35,
            "max_uncapped_ratio_with_trade_rescue": 4.0,
            "max_path_vs_trade_gap": 0.10,
        },
        auc=0.871,
    )
    assert infl["inflated"] is False
    assert infl["gate_pass"] is True
    assert "ratio_high_but_trade_supported" in infl["notes"]


def test_early_fold_weakness_detection():
    from atis.engines.engine4_training.promotion_v16 import fold_stability_report

    folds = [
        {"accuracy": 0.52, "test_sharpe": -0.1, "val_sharpe": 6.0, "n_val_trades": 40, "trade_rate": 0.08},
        {"accuracy": 0.51, "test_sharpe": 0.0, "val_sharpe": 6.0, "n_val_trades": 40, "trade_rate": 0.08},
        {"accuracy": 0.75, "test_sharpe": 5.0, "val_sharpe": 7.0, "n_val_trades": 40, "trade_rate": 0.08},
        {"accuracy": 0.76, "test_sharpe": 5.5, "val_sharpe": 7.2, "n_val_trades": 40, "trade_rate": 0.08},
        {"accuracy": 0.74, "test_sharpe": 6.0, "val_sharpe": 7.1, "n_val_trades": 40, "trade_rate": 0.08},
    ]
    st = fold_stability_report(folds, cfg={"early_fold_frac": 0.4, "early_fold_min_acc": 0.58})
    assert st["early_folds_weak"] is True


def test_overfit_sharpe_gap_hard_fail():
    from atis.engines.engine4_training import should_fail_overfit_sharpe_gap_hard

    # M30-style: Train 13.8 → Val 9.8 gap≈4, Test still strong → hard may exempt
    # Soft overfit still flagged via diagnose_fit; hard requires collapse
    assert should_fail_overfit_sharpe_gap_hard(
        sharpe_gap_tv=4.0,
        overfit_sharpe_gap=2.0,
        train_sharpe=13.8,
        val_sharpe=9.8,
        test_sharpe=7.0,
        sharpe_gap_vt=2.7,
        n_test_trades=600,
        min_sharpe=0.25,
        min_trades=15,
        val_test_gap_hard=3.5,
    ) is False  # strong test exempts hard

    # Real collapse: Val weak + Test collapsed
    assert should_fail_overfit_sharpe_gap_hard(
        sharpe_gap_tv=5.0,
        overfit_sharpe_gap=2.0,
        train_sharpe=10.0,
        val_sharpe=3.0,
        test_sharpe=0.2,
        sharpe_gap_vt=2.8,
        n_test_trades=40,
        min_sharpe=0.25,
        min_trades=15,
        val_test_gap_hard=3.5,
    ) is True


def test_filter_driven_edge_in_diagnosis():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    metrics = {
        "gate_failures": ["filter_driven_edge", "h4_no_edge"],
        "classification": {"roc_auc_ovr": 0.51, "accuracy": 0.50, "trade_rate_filtered": 0.07},
        "financial_oos": {"sharpe": 1.0, "sharpe_uncapped": 1.0, "trade_sharpe_raw": 0.2, "n_trades": 58, "expectancy": 0.0006},
        "financial_train": {"sharpe": -0.9},
        "financial_validation": {"sharpe": 2.3},
        "fit_diagnosis": {
            "status": "balanced",
            "sharpe_gap_train_val": -3.2,
            "sharpe_gap_val_test": 1.3,
            "filter_driven_edge_risk": True,
        },
        "fold_stability": {"trade_rate_pegged": False, "early_folds_weak": True, "stable": True},
        "sharpe_inflation": {"inflated": False},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 0.0}, "pbo": {"pbo": 0.25}},
        "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.6},
        "data_quality": {"quality_flags": {"excessive_gaps": True}},
    }
    diag = build_self_diagnosis(metrics, timeframe="H4", passed_gates=False, cfg={})
    assert diag["primary_root_cause"] in {"Labels", "Features", "ValidationDesign", "DataQuality"}
    assert diag["safe_for_live"]["safe_for_live"] is False
    assert diag["metric_honesty_score"] <= 100


def test_self_diagnosis_m1_inflation_and_saturation():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    metrics = {
        "gate_failures": ["trade_rate_saturated", "inflated_sharpe"],
        "classification": {"roc_auc_ovr": 0.84, "accuracy": 0.76, "trade_rate_filtered": 0.116},
        "financial_oos": {
            "sharpe": 9.06,
            "sharpe_uncapped": 38.35,
            "trade_sharpe_raw": 0.08,
            "n_trades": 967,
            "expectancy": 0.0017,
        },
        "financial_train": {"sharpe": 9.48},
        "financial_validation": {"sharpe": 7.39},
        "fit_diagnosis": {"status": "balanced", "sharpe_gap_train_val": 2.09, "sharpe_gap_val_test": 1.67},
        "fold_stability": {
            "trade_rate_pegged": True,
            "early_folds_weak": False,
            "stable": True,
            "early_fold_stats": {"n_early": 2, "mean_accuracy": 0.7, "mean_test_sharpe": 1.0, "frac_negative_test": 0.0},
        },
        "sharpe_inflation": {
            "inflated": True,
            "sharpe": 9.06,
            "sharpe_uncapped": 38.35,
            "trade_sharpe_raw": 0.08,
            "uncapped_ratio": 4.23,
            "path_vs_trade_gap": True,
        },
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
        "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.99},
        "stress_testing": {"robust": True, "worst_sharpe": 5.0},
    }
    diag = build_self_diagnosis(
        metrics,
        timeframe="M1",
        passed_gates=False,
        cfg={"max_fold_trade_rate": 0.12, "max_sharpe_uncapped": 15.0},
    )
    assert diag["primary_root_cause"] == "MetricInflation"
    assert diag["metric_honesty_score"] < 50
    assert diag["safe_for_live"]["safe_for_live"] is False
    assert "trade_rate_saturated" in diag["safe_for_live"]["constraints_failed"] or (
        "inflated_sharpe" in diag["safe_for_live"]["constraints_failed"]
    )
    assert diag["suggested_config_diff"]
    assert diag["next_actions"][0]["code"] == "demote_uncapped_path_sharpe"


def test_self_diagnosis_m30_overfit():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    metrics = {
        "gate_failures": ["overfit_sharpe_gap", "trade_rate_saturated"],
        "classification": {"roc_auc_ovr": 0.84, "accuracy": 0.76, "trade_rate_filtered": 0.15},
        "financial_oos": {
            "sharpe": 7.08,
            "sharpe_uncapped": 8.24,
            "trade_sharpe_raw": 0.25,
            "n_trades": 611,
            "expectancy": 0.0024,
        },
        "financial_train": {"sharpe": 13.81},
        "financial_validation": {"sharpe": 9.79},
        "fit_diagnosis": {
            "status": "overfitting",
            "sharpe_gap_train_val": 4.02,
            "sharpe_gap_val_test": 2.71,
            "accuracy_gap_train_val": 0.16,
        },
        "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
        "sharpe_inflation": {"inflated": False},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
        "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.8},
    }
    diag = build_self_diagnosis(metrics, timeframe="M30", passed_gates=False, cfg={"max_fold_trade_rate": 0.12})
    assert diag["primary_root_cause"] == "Model/HP"
    assert diag["generalization_score"] < 70
    assert diag["next_actions"][0]["code"] == "regularize_capacity"
    knobs = diag["next_actions"][0]["knobs"]
    assert knobs.get("prefer_nested_on_capacity_conflict") is True
    assert "rf_max_depth" in knobs


def test_self_diagnosis_m1_rf_overfit_emits_rf_knobs():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    metrics = {
        "gate_failures": ["overfit_sharpe_gap"],
        "classification": {"roc_auc_ovr": 0.847, "accuracy": 0.761, "trade_rate_filtered": 0.06},
        "financial_oos": {
            "sharpe": 10.68,
            "sharpe_uncapped": 31.47,
            "trade_sharpe_raw": 0.67,
            "n_trades": 469,
            "expectancy": 0.0023,
        },
        "financial_train": {"sharpe": 17.47},
        "financial_validation": {"sharpe": 13.47},
        "fit_diagnosis": {
            "status": "overfitting",
            "sharpe_gap_train_val": 4.0,
            "sharpe_gap_val_test": 2.79,
            "accuracy_gap_train_val": 0.14,
        },
        "model_zoo": {"winner": "random_forest"},
        "nested_hp": {"best_family": "logistic", "best_score": -0.6},
        "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
        "sharpe_inflation": {"inflated": False},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.0}},
        "monte_carlo": {"enabled": True, "stable": True, "p_profit": 1.0},
    }
    diag = build_self_diagnosis(metrics, timeframe="M1", passed_gates=False, cfg={})
    assert diag["primary_root_cause"] == "Model/HP"
    knobs = diag["suggested_config_diff"]
    assert knobs["prefer_nested_on_capacity_conflict"] is True
    assert knobs["rf_max_depth"] <= 4
    assert knobs["rf_min_samples_leaf"] >= 25


def test_readiness_deductions_never_100_with_warnings():
    from atis.engines.engine4_training.readiness import compute_live_readiness

    metrics = {
        "financial_oos": {
            "sharpe": 5.6,
            "sharpe_ci_low": 4.0,
            "expectancy": 0.0017,
            "n_trades": 328,
            "sharpe_uncapped": 5.6,
            "trade_sharpe_raw": 0.3,
            "sharpe_inflation": {"inflated": False},
        },
        "classification": {"roc_auc_ovr": 0.82},
        "fit_diagnosis": {
            "status": "balanced",
            "sharpe_gap_train_val": -0.68,
            "sharpe_gap_val_test": 3.24,  # H1 20260803 smell
        },
        "regime_validation": {"stable": True},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.25}},
        "fold_stability": {"early_folds_weak": False, "trade_rate_pegged": False, "stable": True},
        "stress_testing": {"robust": True, "worst_sharpe": 2.0},
        "monte_carlo": {"stable": True, "p_profit": 0.7, "enabled": True},
        "session_validation": {"sessions": {"london": {"sharpe": 1.0}, "ny": {"sharpe": 1.2}}},
        "data_intelligence": {"ready": True},
        "model_zoo": {"enabled": True, "winner": "logistic"},
        "feature_explainability": {"shap": {"enabled": True, "method": "tree"}},
    }
    ready = compute_live_readiness(passed_gates=True, metrics=metrics, timeframe="H1")
    assert ready["score"] <= 88.0
    assert ready["verdict"] != "live_ready"
    assert "val_test_gap" in ready["material_warnings"] or "val_test_sharpe_gap" in ready["deductions"]


def test_readiness_blocks_saturation_and_inflation():
    from atis.engines.engine4_training.readiness import compute_live_readiness

    metrics = {
        "financial_oos": {
            "sharpe": 9.0,
            "sharpe_ci_low": 7.0,
            "expectancy": 0.001,
            "n_trades": 900,
            "sharpe_uncapped": 38.0,
            "trade_sharpe_raw": 0.05,
            "sharpe_inflation": {"inflated": True},
        },
        "classification": {"roc_auc_ovr": 0.84},
        "fit_diagnosis": {"status": "balanced", "sharpe_gap_train_val": 1.0, "sharpe_gap_val_test": 1.0},
        "regime_validation": {"stable": True},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
        "fold_stability": {"early_folds_weak": False, "trade_rate_pegged": True, "stable": True},
        "stress_testing": {"robust": True},
        "monte_carlo": {"stable": True, "p_profit": 0.99, "enabled": True},
        "sharpe_inflation": {"inflated": True},
    }
    ready = compute_live_readiness(passed_gates=True, metrics=metrics, timeframe="M1")
    assert ready["verdict"] != "live_ready"
    assert "inflated_sharpe" in ready["deductions"] or "inflated_sharpe" in ready["material_warnings"]
    assert "trade_rate_saturated" in ready["deductions"] or "trade_rate_saturated" in ready["material_warnings"]


def test_diagnosis_json_roundtrip(tmp_path: Path):
    from atis.engines.engine4_training.self_diagnostic import (
        build_self_diagnosis,
        write_diagnosis_json,
    )

    metrics = {
        "gate_failures": ["inflated_sharpe"],
        "classification": {"roc_auc_ovr": 0.8, "accuracy": 0.7, "trade_rate_filtered": 0.1},
        "financial_oos": {"sharpe": 8.0, "sharpe_uncapped": 25.0, "trade_sharpe_raw": 0.05, "n_trades": 100, "expectancy": 0.001},
        "fit_diagnosis": {"status": "balanced", "sharpe_gap_train_val": 1.0, "sharpe_gap_val_test": 1.0},
        "fold_stability": {"trade_rate_pegged": False, "early_folds_weak": False, "stable": True},
        "sharpe_inflation": {"inflated": True, "sharpe_uncapped": 25.0, "uncapped_ratio": 3.1, "trade_sharpe_raw": 0.05},
        "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
        "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.99},
    }
    diag = build_self_diagnosis(metrics, timeframe="M5", passed_gates=False, cfg={"max_sharpe_uncapped": 15})
    path = tmp_path / "diagnosis.json"
    write_diagnosis_json(path, diag)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["primary_root_cause"] == "MetricInflation"
    assert "narrative_ar" in loaded
    assert "suggested_config_diff" in loaded


def test_research_factory_uses_diagnosis(tmp_path: Path):
    from atis.engines.engine4_training.research_factory import append_experiment

    metrics = {
        "financial_oos": {"sharpe": 9.0, "expectancy": 0.001, "sharpe_ci_low": 0.5, "n_trades": 40},
        "classification": {"roc_auc_ovr": 0.8},
        "fit_diagnosis": {"status": "balanced"},
        "pipeline_version": "e4-v17.2",
        "self_diagnosis": {
            "primary_root_cause": "MetricInflation",
            "metric_honesty_score": 30,
            "generalization_score": 70,
            "suggested_config_diff": {"max_sharpe_uncapped": 15.0, "rank_by_trade_sharpe": True},
            "next_actions": [
                {
                    "code": "demote_uncapped_path_sharpe",
                    "hypothesis_ar": "إخضاع شارب المسار",
                    "knobs": {"max_sharpe_uncapped": 15.0},
                    "expected_effect": "reject inflation",
                    "risk": "low",
                }
            ],
            "safe_for_live": {"verdict": "not_safe"},
        },
    }
    out = append_experiment(
        tmp_path,
        symbol="XAUUSD",
        timeframe="M1",
        version="vtest",
        metrics=metrics,
        cfg={"use_promotion_validation_mode": True, "max_fold_trade_rate": 0.12},
        passed_gates=False,
    )
    assert out["hypothesis"]["code"] == "demote_uncapped_path_sharpe"
    assert out["next_hypothesis"]["code"] == "demote_uncapped_path_sharpe"
    assert out["next_hypothesis"].get("timeframe") == "M1"
    next_file = tmp_path / "intelligence" / "next_hypothesis.json"
    assert next_file.exists()
    raw = json.loads(next_file.read_text(encoding="utf-8"))
    assert raw.get("timeframe") == "M1"
