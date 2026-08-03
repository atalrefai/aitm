"""Unit tests for Engine 4 v16 research-factory upgrades."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_pipeline_version_v16():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert "v16" in PIPELINE_VERSION or "v17" in PIPELINE_VERSION
    assert (
        "research-factory" in PIPELINE_VERSION
        or "priority-hardening" in PIPELINE_VERSION
        or "weakness-hardening" in PIPELINE_VERSION
        or "self-diagnostic" in PIPELINE_VERSION
    )

def test_resolve_zoo_vs_nested_ensemble():
    from atis.engines.engine4_training.financial_hpo import resolve_zoo_vs_nested

    out = resolve_zoo_vs_nested(
        nested_meta={"best_family": "logistic", "best_score": 0.2},
        zoo_meta={"winner": "hist_gbm", "ranking": [{"family": "hist_gbm", "score": 0.25}]},
        current_model_name="lightgbm",
        cfg={"prefer_ensemble_on_conflict": True},
    )
    assert out["conflict"] is True
    assert out["selected_baseline"] == "ensemble"


def test_resolve_zoo_vs_nested_single_family_default():
    from atis.engines.engine4_training.financial_hpo import resolve_zoo_vs_nested

    # Comparable scores (both positive acc-edge style) → zoo may win.
    out = resolve_zoo_vs_nested(
        nested_meta={"best_family": "logistic", "best_score": 0.2},
        zoo_meta={"winner": "hist_gbm", "ranking": [{"family": "hist_gbm", "score": 0.25}]},
        current_model_name="lightgbm",
        cfg={"prefer_ensemble_on_conflict": False},
    )
    assert out["conflict"] is True
    assert out["selected_baseline"] != "ensemble"
    assert out["reason"] == "prefer_zoo_financial_proxy"


def test_resolve_zoo_prefers_nested_when_scores_incomparable():
    """M1 smell: nested expectancy_cost (~-0.6) vs zoo acc-edge (~0.26) must not auto-pick RF."""
    from atis.engines.engine4_training.financial_hpo import resolve_zoo_vs_nested

    out = resolve_zoo_vs_nested(
        nested_meta={"best_family": "logistic", "best_score": -0.600971},
        zoo_meta={
            "winner": "random_forest",
            "ranking": [{"family": "random_forest", "score": 0.26}],
        },
        current_model_name="rf",
        cfg={"prefer_ensemble_on_conflict": False},
    )
    assert out["conflict"] is True
    assert out["scores_comparable"] is False
    assert out["selected_baseline"] == "logistic"
    assert out["reason"] == "prefer_nested_incomparable_scores"


def test_resolve_zoo_prefers_nested_under_regularize():
    from atis.engines.engine4_training.financial_hpo import resolve_zoo_vs_nested

    out = resolve_zoo_vs_nested(
        nested_meta={"best_family": "logistic", "best_score": 0.22},
        zoo_meta={
            "winner": "random_forest",
            "ranking": [{"family": "random_forest", "score": 0.28}],
        },
        current_model_name="rf",
        cfg={
            "prefer_ensemble_on_conflict": False,
            "force_regularize_hp": True,
            "prefer_simpler_within_epsilon": True,
        },
    )
    assert out["selected_baseline"] == "logistic"
    assert out["reason"] == "prefer_nested_simpler_under_regularize"


def test_trade_level_and_expectancy_cost():
    from atis.engines.engine4_training.financial_hpo import (
        expectancy_covers_cost,
        trade_level_sharpe,
    )

    rets = np.array([0.001, -0.0005, 0.002, 0.0015, -0.0004, 0.0008])
    tl = trade_level_sharpe(rets)
    assert tl["n_trades"] == 6
    assert "trade_sharpe_raw" in tl
    ok, meta = expectancy_covers_cost(
        0.002,
        spread_pips=30,
        slippage_pips=5,
        pip_size=0.01,
        close_price=2000.0,
        cost_multiple=1.0,
    )
    assert "covers" in meta


def test_fold_stability_and_holdouts():
    from atis.engines.engine4_training.promotion_v16 import (
        crisis_recent_holdout_slices,
        fold_stability_report,
    )

    folds = [
        {"n_val_trades": 40, "policy": {"val_sharpe": 2.0}, "trade_rate": 0.1},
        {"n_val_trades": 35, "policy": {"val_sharpe": 2.2}, "trade_rate": 0.11},
        {"n_val_trades": 30, "policy": {"val_sharpe": 1.8}, "trade_rate": 0.09},
    ]
    stab = fold_stability_report(folds, cfg={"fail_on_fold_unstable": True})
    assert stab["stable"] is True
    slices = crisis_recent_holdout_slices(1000)
    assert len(slices["recent"]) > 0
    assert len(slices["crisis"]) > 0


def test_barrier_sweep_and_clean(tmp_path=None):
    from atis.engines.engine4_training.barrier_optimization import (
        clean_label_weights,
        sweep_barrier_params,
    )

    rng = np.random.default_rng(0)
    n = 400
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "open": close,
            "atr": np.abs(rng.normal(2, 0.3, n)),
        }
    )
    rep = sweep_barrier_params(
        df,
        timeframe="H1",
        cfg={
            "barrier_sweep_enabled": True,
            "barrier_atr_multiplier": 1.5,
            "horizon_by_timeframe": {"H1": 6},
            "barrier_sweep_atr_grid": [1.4, 1.5, 1.7],
            "barrier_sweep_horizon_grid": [5, 6],
            "barrier_sweep_min_gain": 0.5,  # high → likely keep base
        },
    )
    assert rep["enabled"] is True
    y = pd.Series(rng.choice([-1, 0, 1], size=n))
    w = np.ones(n)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    w2, meta = clean_label_weights(y, w, X, cfg={"label_cleaning_enabled": True}, seed=0)
    assert len(w2) == n
    assert meta.get("enabled") is True


def test_research_factory_append(tmp_path):
    from atis.engines.engine4_training.research_factory import append_experiment, load_board

    out = append_experiment(
        tmp_path,
        symbol="XAUUSD",
        timeframe="H1",
        version="vtest",
        metrics={
            "passed_gates": True,
            "financial_oos": {"sharpe": 1.2, "expectancy": 0.001, "sharpe_ci_low": 0.5, "n_trades": 40},
            "classification": {"roc_auc_ovr": 0.55},
            "fit_diagnosis": {"status": "balanced"},
            "pipeline_version": "e4-v16",
        },
        cfg={"use_ensemble": False},
        passed_gates=True,
    )
    assert out["hypothesis"]["code"]
    board = load_board(tmp_path / "intelligence" / "research_factory.json")
    assert len(board["experiments"]) == 1


def test_shadow_challenger_register(tmp_path):
    from atis.engines.engine4_training.shadow_challenger import register_shadow_challenger

    out = register_shadow_challenger(
        tmp_path,
        symbol="XAUUSD",
        timeframe="M5",
        version="v1",
        model_path=str(tmp_path / "model.joblib"),
        metrics={
            "passed_gates": True,
            "financial_oos": {"sharpe": 1.0, "expectancy": 0.001},
            "live_readiness": {"score": 80},
        },
        comparison={"promote": False, "decision": "keep_champion"},
    )
    assert out["registered"] is True


def test_h4_quarantined_from_confirm():
    from atis.engines.engine4_training.multi_tf_decision import confirm_tfs_for_primary

    tfs = confirm_tfs_for_primary(
        "M5",
        {"confirm_timeframes": ["M15", "H1", "H4"], "quarantine_h4_confirm": True},
    )
    assert "H4" not in tfs
    assert "H1" in tfs


def test_stress_extra_scenarios():
    from atis.engines.engine4_training.stress_testing import stress_scenarios

    r = np.zeros(100)
    r[::5] = 0.001
    r[1::7] = -0.0008

    def _fm(x):
        x = x[x != 0]
        if len(x) < 2:
            return {"sharpe": 0.0, "expectancy": 0.0, "max_drawdown": 0.0, "n_trades": 0}
        return {
            "sharpe": float(np.mean(x) / (np.std(x) + 1e-9)),
            "expectancy": float(np.mean(x)),
            "max_drawdown": -0.01,
            "n_trades": float(len(x)),
        }

    out = stress_scenarios(
        r, financial_fn=_fm, latency_extra=2, gap_shock=0.001, seed=1
    )
    assert "latency_extra" in out["scenarios"]
    assert "gap_shock" in out["scenarios"]


def test_confidence_position_size_bounds():
    from atis.engines.engine4_training.promotion_v16 import confidence_position_size

    low = confidence_position_size(0.4, atr_pct=0.01, min_size=0.25, max_size=1.5)
    high = confidence_position_size(0.95, atr_pct=0.001, min_size=0.25, max_size=1.5)
    assert 0.25 <= low <= 1.5
    assert 0.25 <= high <= 1.5
    assert high >= low


def test_financial_proxy_penalizes_spam_trades():
    from atis.engines.engine4_training.financial_hpo import financial_proxy_score

    y = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1] * 5)
    good = y.copy()
    good[1::5] = 0  # ~80% trade rate tempered
    spam_wrong = -np.ones_like(y)
    assert financial_proxy_score(y, good) > financial_proxy_score(y, spam_wrong)


def test_research_suggest_next_hypothesis():
    from atis.engines.engine4_training.research_factory import suggest_next_hypothesis

    hyp = suggest_next_hypothesis(
        [],
        {"use_promotion_validation_mode": False, "max_fold_trade_rate": 0.22},
        stop=False,
        stop_reason="continue",
    )
    assert hyp is not None
    assert hyp["code"] == "cpcv_promotion"


def test_shadow_retrain_advisory_helpers(tmp_path):
    from atis.engines.engine4_training.shadow_challenger import (
        read_retrain_advisory,
        write_retrain_request,
    )

    adv = read_retrain_advisory(tmp_path)
    assert adv.get("exists") is False
    intel = tmp_path / "intelligence"
    intel.mkdir(parents=True)
    (intel / "retrain_advisory.json").write_text(
        '{"would_trigger_now": true, "schedule_reason": "drift"}',
        encoding="utf-8",
    )
    adv2 = read_retrain_advisory(tmp_path)
    assert adv2.get("would_trigger_now") is True
    req = write_retrain_request(tmp_path, reason="drift", source="test")
    assert req["status"] == "pending"
    assert (intel / "retrain_request.json").exists()


def test_inflated_sharpe_and_early_folds():
    from atis.engines.engine4_training.promotion_v16 import (
        fold_stability_report,
        inflated_sharpe_report,
    )

    infl = inflated_sharpe_report(
        {"sharpe": 7.0, "sharpe_uncapped": 40.0, "trade_sharpe_raw": 0.05},
        cfg={"fail_on_inflated_sharpe": True, "max_sharpe_uncapped": 20.0},
    )
    assert infl["inflated"] is True
    assert infl["gate_pass"] is False

    folds = [
        {"accuracy": 0.52, "test_sharpe": -0.1, "val_sharpe": 6.0, "n_val_trades": 40, "trade_rate": 0.12},
        {"accuracy": 0.51, "test_sharpe": 0.0, "val_sharpe": 6.0, "n_val_trades": 40, "trade_rate": 0.12},
        {"accuracy": 0.75, "test_sharpe": 5.0, "val_sharpe": 7.0, "n_val_trades": 40, "trade_rate": 0.12},
        {"accuracy": 0.76, "test_sharpe": 5.5, "val_sharpe": 7.2, "n_val_trades": 40, "trade_rate": 0.12},
        {"accuracy": 0.74, "test_sharpe": 6.0, "val_sharpe": 7.1, "n_val_trades": 40, "trade_rate": 0.12},
    ]
    st = fold_stability_report(folds, cfg={"early_fold_frac": 0.4, "early_fold_min_acc": 0.58})
    assert st["early_folds_weak"] is True


def test_shap_fallback_without_tree():
    from sklearn.linear_model import LogisticRegression
    from atis.engines.engine4_training.feature_explainability import compute_shap_importance

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    names = [f"f{i}" for i in range(6)]
    out = compute_shap_importance(model, X, names, max_rows=80, seed=1)
    assert out.get("enabled") is True
    assert out.get("top")
