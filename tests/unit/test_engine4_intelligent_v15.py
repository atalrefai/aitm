"""Unit tests for Engine 4 v15 intelligent training modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def test_pipeline_version_v15():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert (
        "v15" in PIPELINE_VERSION
        or "v16" in PIPELINE_VERSION
        or "v17" in PIPELINE_VERSION
    )
    assert (
        "intelligent-training" in PIPELINE_VERSION
        or "research-factory" in PIPELINE_VERSION
        or "priority-hardening" in PIPELINE_VERSION
        or "weakness-hardening" in PIPELINE_VERSION
    )


def test_label_quality_report():
    from atis.engines.engine4_training.label_quality import analyze_label_quality

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25]))
    w = np.clip(rng.random(n), 0.3, 1.0)
    rep = analyze_label_quality(X, y, label_weights=w, timeframe="H1", cfg={}, seed=0)
    assert rep["enabled"] is True
    assert 0 <= rep["score"] <= 100
    assert "flags" in rep
    assert "summary_ar" in rep


def test_feature_explainability_permutation_and_stability():
    from atis.engines.engine4_training.feature_explainability import (
        build_explainability_report,
        feature_stability_across_folds,
    )

    rng = np.random.default_rng(1)
    n, p = 300, 8
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int) * 2 - 1
    names = [f"f{i}" for i in range(p)]
    model = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=1)
    model.fit(X, y)
    folds = [names[:5], names[1:6], names[:4] + [names[6]]]
    stab = feature_stability_across_folds(folds, min_frac=0.5)
    assert stab["enabled"] is True
    assert "mean_jaccard" in stab
    rep = build_explainability_report(
        model,
        X,
        y,
        names,
        fold_top_features=folds,
        seed=1,
        cfg={"shap_enabled": False, "permutation_importance_enabled": True, "permutation_n_repeats": 2},
    )
    assert rep["enabled"] is True
    assert rep["permutation"]["enabled"] is True
    assert len(rep["consensus_top"]) > 0


def test_champion_challenger_promote_first():
    from atis.engines.engine4_training.champion_challenger import compare_challenger_to_champion
    from pathlib import Path

    metrics = {
        "passed_gates": True,
        "financial_oos": {"sharpe": 1.2, "expectancy": 0.001, "max_drawdown": 0.1, "n_trades": 40},
        "classification": {"roc_auc_ovr": 0.58},
        "fit_diagnosis": {"status": "balanced"},
        "live_readiness": {"score": 80},
    }
    cmp = compare_challenger_to_champion(
        metrics,
        models_root=Path("models_does_not_exist_xyz"),
        symbol="XAUUSD",
        timeframe="H1",
        cfg={},
    )
    assert cmp["promote"] is True
    assert cmp["decision"] == "promote_as_first_champion"


def test_smart_recommendations_overfit():
    from atis.engines.engine4_training.smart_recommendations import build_smart_recommendations

    recs = build_smart_recommendations(
        {
            "passed_gates": False,
            "gate_failures": ["overfit"],
            "fit_diagnosis": {"status": "overfitting"},
            "financial_oos": {"sharpe": -0.2, "expectancy": -0.001},
            "classification": {"roc_auc_ovr": 0.51},
            "live_readiness": {"score": 40},
        },
        timeframe="H1",
    )
    assert recs["enabled"] is True
    assert recs["n_recommendations"] >= 1
    codes = {i["code"] for i in recs["items"]}
    assert "regularize" in codes


def test_nested_hp_across_outer_folds():
    from atis.engines.engine4_training.adaptive_learning import nested_hp_across_outer_folds

    rng = np.random.default_rng(2)
    payloads = []
    for fold in range(2):
        n = 220
        X = rng.normal(size=(n, 6))
        y = (X[:, 0] > 0).astype(int) * 2 - 1
        y[y == 0] = -1
        # mix some flats as 0
        y[::7] = 0
        payloads.append({"X": X, "y": y, "w": np.ones(n), "fold": fold})
    cfg = {
        "nested_hp_search": True,
        "nested_hp_trials": 3,
        "nested_hp_compare_baselines": False,
        "lgb_estimators": 40,
        "lgb_learning_rate": 0.05,
        "lgb_max_depth": 3,
        "lgb_num_leaves": 8,
        "lgb_min_child_samples": 20,
        "lgb_subsample": 0.8,
        "lgb_colsample": 0.8,
        "lgb_reg_alpha": 0.1,
        "lgb_reg_lambda": 1.0,
    }
    best_cfg, meta = nested_hp_across_outer_folds(
        payloads, base_cfg=cfg, timeframe="H1", seed=2, n_trials=3, max_folds=2
    )
    assert meta.get("enabled") is True
    assert meta.get("mode") == "nested_across_outer_folds"
    assert "best_family" in meta
    assert isinstance(best_cfg, dict)


def test_enterprise_report_includes_v15_sections(tmp_path):
    from atis.engines.engine4_training.enterprise_report import write_enterprise_report

    path = tmp_path / "dossier.md"
    write_enterprise_report(
        path,
        {
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "pipeline_version": "e4-v15.0-intelligent-training-engine-20260731",
            "version": "test",
            "passed_gates": True,
            "financial_oos": {"sharpe": 1.0, "expectancy": 0.001},
            "classification": {"accuracy": 0.55, "roc_auc_ovr": 0.56},
            "fit_diagnosis": {"status": "balanced"},
            "label_quality": {"score": 80, "summary_ar": "ok", "noise": {"noise_rate": 0.4}},
            "feature_explainability": {
                "shap": {"enabled": False},
                "permutation": {"enabled": True},
                "stability": {"summary_ar": "stable"},
                "consensus_top": [{"feature": "f1", "score": 0.2}],
                "warnings": [],
            },
            "champion_challenger": {
                "decision": "promote_as_first_champion",
                "promote": True,
                "summary_ar": "ترقية",
            },
            "smart_recommendations": {
                "primary_code": "monitor",
                "executive_ar": "راقب",
                "items": [{"priority": 4, "code": "monitor", "ar": "راقب"}],
            },
            "nested_hp": {"mode": "nested_across_outer_folds", "best_family": "lightgbm", "best_score": 0.05},
            "live_readiness": {"score": 70, "verdict": "paper_ready", "verdict_ar": "ورقي"},
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "Label Quality" in text
    assert "Champion vs Challenger" in text
    assert "Smart Recommendations" in text
    assert "Explainability" in text
