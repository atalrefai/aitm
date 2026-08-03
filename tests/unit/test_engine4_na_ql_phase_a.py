"""Unit tests for Engine-4 NA-QL Phase A (quality compound + consistency + unify)."""

from __future__ import annotations

import numpy as np


def test_pipeline_version_phase_a():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert "v17.4" in PIPELINE_VERSION or "na-ql" in PIPELINE_VERSION


def test_quality_compound_penalizes_saturation_and_inflation():
    from atis.engines.engine4_training.financial_hpo import quality_compound_score

    good = quality_compound_score(
        win_rate=0.58,
        f1=0.45,
        expectancy=0.0015,
        trade_rate=0.07,
        trade_sharpe_raw=0.4,
        target_trade_rate=0.08,
        max_trade_rate=0.12,
        inflated=False,
        trade_rate_pegged=False,
    )
    bad = quality_compound_score(
        win_rate=0.58,
        f1=0.45,
        expectancy=0.0015,
        trade_rate=0.12,
        trade_sharpe_raw=0.4,
        target_trade_rate=0.08,
        max_trade_rate=0.12,
        inflated=True,
        trade_rate_pegged=True,
    )
    assert good["score"] > bad["score"]
    assert "trade_rate_pegged" in bad["penalties"]
    assert "metric_inflation" in bad["penalties"]


def test_financial_proxy_includes_f1_and_cap_penalty():
    from atis.engines.engine4_training.financial_hpo import financial_proxy_score

    rng = np.random.default_rng(0)
    y = rng.choice([-1, 0, 1], size=200, p=[0.2, 0.6, 0.2])
    # High-quality sparse preds
    pred_q = np.where(y != 0, y, 0)
    mask = rng.random(200) > 0.85
    pred_q = np.where(mask, pred_q, 0)
    # Spam preds near cap
    pred_spam = rng.choice([-1, 1], size=200)

    sq = financial_proxy_score(y, pred_q, target_trade_rate=0.10, max_trade_rate=0.12)
    ss = financial_proxy_score(y, pred_spam, target_trade_rate=0.10, max_trade_rate=0.12)
    assert sq > ss


def test_consistency_score_penalizes_dispersion():
    from atis.engines.engine4_training.promotion_v16 import consistency_score_report

    stable_folds = [
        {"win_rate": 0.55, "f1_macro": 0.42, "expectancy": 0.001, "test_sharpe": 1.0},
        {"win_rate": 0.56, "f1_macro": 0.43, "expectancy": 0.0011, "test_sharpe": 1.1},
        {"win_rate": 0.54, "f1_macro": 0.41, "expectancy": 0.0009, "test_sharpe": 0.9},
    ]
    wild_folds = [
        {"win_rate": 0.70, "f1_macro": 0.60, "expectancy": 0.005, "test_sharpe": 4.0},
        {"win_rate": 0.40, "f1_macro": 0.20, "expectancy": -0.004, "test_sharpe": -2.0},
        {"win_rate": 0.55, "f1_macro": 0.40, "expectancy": 0.001, "test_sharpe": 0.5},
    ]
    regimes_ok = {
        "regimes": {
            "trending": {"win_rate": 0.55, "expectancy": 0.001, "skipped": False},
            "ranging": {"win_rate": 0.53, "expectancy": 0.0008, "skipped": False},
            "high_volatility": {"win_rate": 0.52, "expectancy": 0.0005, "skipped": False},
        },
        "stable": True,
    }
    s = consistency_score_report(
        fold_metrics=stable_folds,
        regime_validation=regimes_ok,
        financial_oos={"win_rate": 0.55},
        classification={"f1_macro": 0.42},
    )
    w = consistency_score_report(
        fold_metrics=wild_folds,
        regime_validation={**regimes_ok, "stable": False},
        financial_oos={"win_rate": 0.50},
        classification={"f1_macro": 0.30},
    )
    assert s["score"] > w["score"]
    assert s["enabled"] is True


def test_regime_balanced_holdouts_gate():
    from atis.engines.engine4_training.promotion_v16 import regime_balanced_holdout_report

    weak = {
        "regimes": {
            "trending": {"win_rate": 0.60, "expectancy": 0.002, "skipped": False},
            "ranging": {"win_rate": 0.40, "expectancy": -0.001, "skipped": False},
            "high_volatility": {"win_rate": 0.42, "expectancy": -0.002, "skipped": False},
        }
    }
    off = regime_balanced_holdout_report(weak, cfg={"regime_balanced_holdouts": False})
    assert off["gate_pass"] is True
    on = regime_balanced_holdout_report(
        weak,
        cfg={"regime_balanced_holdouts": True, "regime_balanced_min_regimes": 3},
    )
    assert on["gate_pass"] is False
    assert on["n_ok"] < 3


def test_research_factory_prefers_diagnosis_hypothesis():
    from atis.engines.engine4_training.research_factory import infer_hypothesis

    metrics = {
        "self_diagnosis": {
            "primary_root_cause": "TradePolicy",
            "suggested_config_diff": {"max_fold_trade_rate": 0.06, "quality_first_trade_policy": True},
            "unified_hypothesis": {
                "code": "desaturate_trade_policy",
                "single_change": "max_fold_trade_rate=0.06",
                "ar": "خفض ضغط التداول",
                "knobs": {"max_fold_trade_rate": 0.06},
            },
            "next_actions": [{"code": "desaturate_trade_policy", "knobs": {"max_fold_trade_rate": 0.06}}],
        }
    }
    hyp = infer_hypothesis({"use_ensemble": True, "fail_on_high_pbo": True}, metrics)
    assert hyp["code"] == "desaturate_trade_policy"
    assert hyp.get("source") == "self_diagnosis"


def test_propose_overrides_exclusive_when_diagnosis_present():
    from atis.engines.engine4_training.enterprise_report import propose_config_overrides

    metrics = {
        "gate_failures": ["trade_rate_saturated", "overfit_sharpe_gap"],
        "fit_diagnosis": {"status": "overfitting"},
        "classification": {"roc_auc_ovr": 0.51},
        "financial_oos": {"expectancy": -0.001, "sharpe": 0.1},
        "fold_stability": {"trade_rate_pegged": True},
        "self_diagnosis": {
            "primary_root_cause": "TradePolicy",
            "suggested_config_diff": {
                "quality_first_trade_policy": True,
                "max_fold_trade_rate": 0.06,
            },
            "next_actions": [{"code": "desaturate_trade_policy", "hypothesis_ar": "خفض"}],
            "unified_hypothesis": {"code": "desaturate_trade_policy"},
        },
    }
    out = propose_config_overrides(timeframe="M15", metrics=metrics, passed_gates=False)
    assert out["unified_from_diagnosis"] is True
    assert out["overrides"]["max_fold_trade_rate"] == 0.06
    # Must not pile H4/near-chance barrier knobs on top when diagnosis owns the diff
    assert "barrier_atr_multiplier" not in out["overrides"] or "barrier_atr_multiplier" in metrics[
        "self_diagnosis"
    ]["suggested_config_diff"]


def test_smart_recommendations_primary_from_diagnosis():
    from atis.engines.engine4_training.smart_recommendations import build_smart_recommendations

    metrics = {
        "passed_gates": False,
        "gate_failures": ["trade_rate_saturated"],
        "fit_diagnosis": {"status": "balanced"},
        "financial_oos": {"expectancy": 0.001},
        "classification": {"roc_auc_ovr": 0.7},
        "self_diagnosis": {
            "suggested_config_diff": {"max_fold_trade_rate": 0.06},
            "unified_hypothesis": {
                "code": "desaturate_trade_policy",
                "ar": "خفض ضغط التداول",
                "knobs": {"max_fold_trade_rate": 0.06},
            },
            "next_actions": [{"code": "desaturate_trade_policy", "knobs": {"max_fold_trade_rate": 0.06}}],
        },
        "live_readiness": {"score": 40},
    }
    rec = build_smart_recommendations(metrics, timeframe="M15")
    assert rec["primary_code"] == "desaturate_trade_policy"
    assert rec["unified_from_diagnosis"] is True
    assert rec["primary_knobs"].get("max_fold_trade_rate") == 0.06


def test_readiness_material_warning_on_low_consistency():
    from atis.engines.engine4_training.readiness import compute_live_readiness

    metrics = {
        "financial_oos": {"sharpe": 2.0, "sharpe_ci_low": 0.5, "expectancy": 0.001, "n_trades": 80, "trade_sharpe_raw": 0.3},
        "classification": {"roc_auc_ovr": 0.7},
        "fit_diagnosis": {"status": "balanced", "sharpe_gap_train_val": 0.5, "sharpe_gap_val_test": 0.5},
        "fold_stability": {"stable": True, "trade_rate_pegged": False, "early_folds_weak": False},
        "consistency": {"enabled": True, "score": 40.0},
        "self_diagnosis": {"metric_honesty_score": 80, "generalization_score": 80},
    }
    ready = compute_live_readiness(passed_gates=True, metrics=metrics, timeframe="M15")
    assert "low_consistency" in ready["material_warnings"]
    assert ready["verdict"] != "live_ready"
