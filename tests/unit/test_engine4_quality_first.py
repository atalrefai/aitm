"""Quality-first trade policy (e4-v17.3) — anti-peg + win-rate oriented scoring."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from atis.engines.engine4_training import PIPELINE_VERSION, train_confidence_floor
from atis.engines.engine4_training.financial_hpo import financial_proxy_score
from atis.engines.engine4_training.final_model import publish_final_model


def test_effective_max_trade_rate_leaves_peg_headroom():
    from atis.engines.engine4_training import effective_max_trade_rate, cap_preds_by_trade_rate

    hard = 0.05
    ops = effective_max_trade_rate(hard, quality_first=True, headroom_frac=0.82, target_trade_rate=0.035)
    assert ops < hard * 0.90  # below fold_stability peg threshold
    assert ops <= 0.035 + 1e-9

    rng = np.random.default_rng(0)
    n = 1000
    preds = np.ones(n)
    conf = rng.random(n)
    capped = cap_preds_by_trade_rate(preds, conf, max_trade_rate=hard, quality_first=True, headroom_frac=0.82)
    rate = float(np.mean(capped != 0))
    assert rate < hard * 0.90
    assert rate <= hard * 0.82 + 0.002


def test_desaturate_steps_below_005_when_already_pegged():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    diag = build_self_diagnosis(
        {
            "fit_diagnosis": {"status": "balanced", "sharpe_gap_train_val": 0.3, "sharpe_gap_val_test": 0.6},
            "financial_oos": {
                "sharpe": 13.25,
                "sharpe_uncapped": 14.32,
                "trade_sharpe_raw": 0.83,
                "n_trades": 508,
                "expectancy": 0.0049,
                "trade_rate": 0.055,
            },
            "classification": {"accuracy": 0.797, "roc_auc_ovr": 0.869},
            "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {"inflated": False},
            "advanced": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.25}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 1.0},
            "stress_testing": {"robust": True},
            "max_fold_trade_rate": 0.05,
        },
        timeframe="M15",
        passed_gates=False,
        gate_failures=["trade_rate_saturated"],
        cfg={
            "max_fold_trade_rate": 0.05,
            "max_fold_trade_rate_by_tf": {"M15": 0.05},
            "min_trade_confidence": 0.58,
            "confidence_quantile": 0.93,
            "directional_edge": 0.18,
        },
    )
    assert diag["primary_root_cause"] == "TradePolicy"
    knobs = diag["suggested_config_diff"]
    # Collapsed 0.05 below M15 healthy 0.08 → restore baseline + headroom (not noop cut).
    assert knobs["max_fold_trade_rate"] == 0.08
    assert knobs["target_trade_rate"] < knobs["max_fold_trade_rate"] * 0.85
    assert knobs["min_trade_confidence"] >= 0.60
    assert knobs["confidence_quantile"] >= 0.93
    assert float(knobs.get("quality_first_cap_headroom_frac", 1.0)) <= 0.82


def test_desaturate_restores_m30_poisoned_floor():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    diag = build_self_diagnosis(
        {
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": -0.22,
                "sharpe_gap_val_test": 3.55,
            },
            "financial_oos": {
                "sharpe": 9.07,
                "sharpe_uncapped": 9.07,
                "trade_sharpe_raw": 0.70,
                "n_trades": 296,
                "expectancy": 0.0035,
                "trade_rate": 0.046,
            },
            "classification": {"accuracy": 0.712, "roc_auc_ovr": 0.827},
            "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {"inflated": False},
            "advanced": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.25}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 1.0},
            "max_fold_trade_rate": 0.05,
        },
        timeframe="M30",
        passed_gates=False,
        gate_failures=["val_test_gap_hard", "trade_rate_saturated"],
        cfg={
            "max_fold_trade_rate": 0.05,
            "max_fold_trade_rate_by_tf": {"M30": 0.05},
            "min_trade_confidence": 0.58,
            "confidence_quantile": 0.93,
        },
    )
    assert diag["primary_root_cause"] == "TradePolicy"
    knobs = diag["suggested_config_diff"]
    assert knobs["max_fold_trade_rate"] == 0.07
    assert knobs["target_trade_rate"] <= 0.045
    assert "Restore" in (diag["next_actions"][0].get("hypothesis") or "")

def test_pipeline_version_quality_first():
    assert "e4-v17." in PIPELINE_VERSION
    assert "quality-first" in PIPELINE_VERSION
    assert "v17.3" in PIPELINE_VERSION or "v17.4" in PIPELINE_VERSION


def test_early_folds_auc_rescue():
    from atis.engines.engine4_training.promotion_v16 import fold_stability_report

    # M15 fingerprint: early Acc≈0.55 but AUC≈0.74 and positive mean test Sharpe
    folds = [
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 10.0},
            "trade_rate": 0.07,
            "accuracy": 0.55,
            "roc_auc_ovr": 0.74,
            "test_sharpe": 2.5,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 9.5},
            "trade_rate": 0.07,
            "accuracy": 0.53,
            "roc_auc_ovr": 0.73,
            "test_sharpe": 1.8,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 11.0},
            "trade_rate": 0.07,
            "accuracy": 0.57,
            "roc_auc_ovr": 0.75,
            "test_sharpe": 2.0,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 10.5},
            "trade_rate": 0.07,
            "accuracy": 0.76,
            "roc_auc_ovr": 0.84,
            "test_sharpe": 8.0,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 10.2},
            "trade_rate": 0.07,
            "accuracy": 0.75,
            "roc_auc_ovr": 0.83,
            "test_sharpe": 7.5,
        },
    ]
    st = fold_stability_report(
        folds,
        cfg={
            "max_fold_trade_rate": 0.09,
            "fail_on_fold_unstable": True,
            "early_fold_frac": 0.40,
            "early_fold_min_acc": 0.58,
            "early_fold_min_auc": 0.60,
            "early_fold_min_mean_test_sharpe": 0.0,
            "early_fold_max_neg_frac": 0.34,
        },
    )
    assert st["early_folds_weak"] is False
    assert st["early_fold_stats"]["auc_rescue"] is True


def test_early_folds_still_weak_when_auc_also_low():
    from atis.engines.engine4_training.promotion_v16 import fold_stability_report

    folds = [
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 2.0},
            "trade_rate": 0.07,
            "accuracy": 0.52,
            "roc_auc_ovr": 0.51,
            "test_sharpe": 0.2,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 2.1},
            "trade_rate": 0.07,
            "accuracy": 0.50,
            "roc_auc_ovr": 0.50,
            "test_sharpe": -0.1,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 8.0},
            "trade_rate": 0.07,
            "accuracy": 0.75,
            "roc_auc_ovr": 0.82,
            "test_sharpe": 6.0,
        },
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 8.0},
            "trade_rate": 0.07,
            "accuracy": 0.76,
            "roc_auc_ovr": 0.83,
            "test_sharpe": 6.5,
        },
    ]
    st = fold_stability_report(
        folds,
        cfg={
            "max_fold_trade_rate": 0.09,
            "early_fold_frac": 0.40,
            "early_fold_min_acc": 0.58,
            "early_fold_min_auc": 0.60,
        },
    )
    assert st["early_folds_weak"] is True


def test_effective_max_trade_rate_leaves_headroom_under_peg():
    from atis.engines.engine4_training import (
        cap_preds_by_trade_rate,
        effective_max_trade_rate,
    )
    from atis.engines.engine4_training.promotion_v16 import fold_stability_report

    # M5 fingerprint: gate cap 0.07 → ops must stay below peg thr 0.063
    ops = effective_max_trade_rate(0.07, quality_first=True, headroom_frac=0.82, target_trade_rate=0.045)
    assert ops < 0.07 * 0.90
    assert ops <= 0.07 * 0.82 + 1e-9

    n = 1000
    preds = np.ones(n, dtype=float)
    conf = np.linspace(0.5, 1.0, n)
    capped = cap_preds_by_trade_rate(preds, conf, max_trade_rate=ops)
    rate = float(np.mean(capped != 0))
    assert rate <= ops + 0.002
    assert rate < 0.07 * 0.90

    folds = [
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 8.0},
            "trade_rate": rate,
            "accuracy": 0.75,
            "test_sharpe": 6.0,
        }
        for _ in range(5)
    ]
    st = fold_stability_report(folds, cfg={"max_fold_trade_rate": 0.07, "fail_on_fold_unstable": True})
    assert st["trade_rate_pegged"] is False


def test_fill_to_hard_cap_still_pegs_without_headroom():
    from atis.engines.engine4_training import cap_preds_by_trade_rate
    from atis.engines.engine4_training.promotion_v16 import fold_stability_report

    n = 1000
    preds = np.ones(n, dtype=float)
    conf = np.linspace(0.5, 1.0, n)
    capped = cap_preds_by_trade_rate(preds, conf, max_trade_rate=0.07)  # quality_first=False default
    rate = float(np.mean(capped != 0))
    folds = [
        {
            "n_val_trades": 40,
            "policy": {"val_sharpe": 8.0},
            "trade_rate": rate,
            "accuracy": 0.75,
            "test_sharpe": 6.0,
        }
        for _ in range(5)
    ]
    st = fold_stability_report(folds, cfg={"max_fold_trade_rate": 0.07, "fail_on_fold_unstable": True})
    assert st["trade_rate_pegged"] is True

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    # Separable enough for stable proba max distribution
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)
    model = LogisticRegression(max_iter=400, random_state=0).fit(X, y)

    legacy = train_confidence_floor(
        model,
        X,
        decision_threshold=0.54,
        confidence_quantile=0.85,
        min_floor=0.52,
        target_trade_rate=0.09,
        max_floor=0.88,
        max_trade_rate=0.12,
        quality_first=False,
    )
    qf = train_confidence_floor(
        model,
        X,
        decision_threshold=0.56,
        confidence_quantile=0.90,
        min_floor=0.56,
        target_trade_rate=0.06,
        max_floor=0.88,
        max_trade_rate=0.09,
        quality_first=True,
    )
    assert qf >= legacy - 1e-9
    assert qf >= 0.56


def test_financial_proxy_prefers_higher_win_rate_not_pegged_rate():
    y = np.array([1, 1, -1, -1, 1, -1, 1, -1, 1, -1] * 8)
    # Pegged busy policy with mediocre accuracy
    busy = y.copy()
    busy[::3] = 0
    busy[1::5] = -busy[1::5]
    # Sparse high-precision policy
    sparse = np.zeros_like(y)
    hits = np.flatnonzero(y != 0)[::4]
    sparse[hits] = y[hits]

    busy_score = financial_proxy_score(y, busy, target_trade_rate=0.06)
    sparse_score = financial_proxy_score(y, sparse, target_trade_rate=0.06)
    assert sparse_score > busy_score


def test_final_model_mode_respects_paper_ready(tmp_path, monkeypatch):
    import atis.engines.engine4_training.final_model as fm

    model_dir = tmp_path / "H1" / "v1"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.joblib"
    model_path.write_bytes(b"fake")

    monkeypatch.setattr(fm, "get_path", lambda _name: tmp_path)
    results = [
        {
            "timeframe": "H1",
            "version": "v1",
            "symbol": "XAUUSD",
            "passed_gates": True,
            "model_path": str(model_path),
            "metrics": {
                "financial_oos": {"sharpe": 5.0, "max_drawdown": -0.02, "total_return": 0.2, "n_trades": 80},
                "financial_validation": {"sharpe": 6.0},
                "financial_deploy_holdout": {"sharpe": 4.5, "n_trades": 40},
                "classification": {"roc_auc_ovr": 0.82, "accuracy": 0.75},
                "fit_diagnosis": {"status": "balanced"},
                "live_readiness": {"verdict": "paper_ready", "score": 88},
            },
        }
    ]
    out = publish_final_model(results, symbol="XAUUSD", allow_paper_final=False)
    assert out.get("passed_gates") is True
    assert out.get("mode") == "paper_only"
    assert out.get("readiness_verdict") == "paper_ready"


def test_high_pbo_self_diag_prefers_selection_bias():
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    diag = build_self_diagnosis(
        {
            "fit_diagnosis": {
                "status": "overfitting",
                "sharpe_gap_train_val": 1.78,
                "sharpe_gap_val_test": 2.34,
            },
            "financial_oos": {
                "sharpe": 7.82,
                "sharpe_uncapped": 7.82,
                "trade_sharpe_raw": 0.50,
                "n_trades": 437,
                "expectancy": 0.0026,
                "trade_rate": 0.076,
            },
            "classification": {"accuracy": 0.70, "roc_auc_ovr": 0.82},
            "advanced": {"pbo": {"pbo": 0.75, "n_paths": 7, "reliable": 1}},
            "fold_stability": {"early_folds_weak": False, "trade_rate_pegged": False},
            "sharpe_inflation": {"inflated": False},
        },
        timeframe="M30",
        passed_gates=False,
        gate_failures=["high_pbo"],
        cfg={"max_fold_trade_rate": 0.08, "top_features": 52, "max_pbo": 0.55},
    )
    assert diag["primary_root_cause"] == "Model/HP"
    assert diag["next_actions"][0]["code"] == "reduce_selection_bias"


def test_pbo_soft_warn_without_corroboration():
    """Coarse PBO with flat Val≈Test should warn, not hard-fail."""
    from atis.engines.engine4_training.advanced_metrics import (
        pbo_hard_fail_decision,
        probability_of_backtest_overfitting,
    )

    fit_diag = {"status": "balanced", "sharpe_gap_val_test": 0.4}
    # Rank-flip heavy: top-IS paths land in bottom-OOS half → PBO≈0.75
    # but mean OOS still close to mean IS → soft_warn, not material.
    pbo_report = probability_of_backtest_overfitting(
        [10.0, 9.0, 8.0, 7.0, 1.0, 2.0, 3.0],
        [1.5, 2.0, 2.5, 3.0, 9.5, 8.5, 7.5],
    )
    assert pbo_report["pbo"] >= 0.55
    assert pbo_report["oos_retention"] >= 0.75
    assert pbo_report["material"] < 0.5
    assert pbo_report["soft_warn"] > 0.5

    cfg = {
        "fail_on_high_pbo": True,
        "max_pbo": 0.55,
        "pbo_require_corroboration": True,
        "pbo_min_paths_for_hard_fail": 6,
        "pbo_full_trust_paths": 10,
        "pbo_oos_collapse_frac": 0.65,
        "pbo_corroborate_val_test_gap": 2.0,
    }
    soft = pbo_hard_fail_decision(pbo_report, fit_diag, cfg=cfg)
    assert soft["hard_fail"] is False
    assert soft["soft_warn"] is True

    # Corroborated overfit + OOS collapse → hard-fail
    collapsed = probability_of_backtest_overfitting(
        [10.0, 9.0, 8.0, 7.0, 1.0, 2.0, 3.0],
        [0.4, 0.5, 0.3, 0.2, 1.2, 1.0, 0.8],
    )
    assert collapsed["material"] > 0.5
    hard = pbo_hard_fail_decision(
        collapsed,
        {"status": "overfitting", "sharpe_gap_val_test": 2.34},
        cfg=cfg,
    )
    assert hard["hard_fail"] is True


def test_effective_max_trade_rate_stays_under_peg_threshold():
    """H1 fingerprint: filling hard cap=0.05 pegs (≥0.90×cap). Ops must leave headroom."""
    from atis.engines.engine4_training import (
        cap_preds_by_trade_rate,
        effective_max_trade_rate,
    )

    # Pre-fix gate (rejected run): ops must stay under peg even if target is low.
    gate = 0.05
    peg = gate * 0.90
    ops = effective_max_trade_rate(
        gate, quality_first=True, headroom_frac=0.82, target_trade_rate=0.025
    )
    assert ops < peg
    assert ops <= gate * 0.82 + 1e-9

    # Post-desaturate H1 gate.
    gate2 = 0.035
    peg2 = gate2 * 0.90
    ops2 = effective_max_trade_rate(
        gate2, quality_first=True, headroom_frac=0.82, target_trade_rate=0.025
    )
    assert ops2 < peg2
    assert ops2 <= 0.025 + 1e-9

    n = 1000
    conf = np.linspace(0.5, 0.99, n)
    preds = np.ones(n)  # would fill any cap
    capped = cap_preds_by_trade_rate(preds, conf, max_trade_rate=ops2, quality_first=False)
    rate = float(np.mean(capped != 0))
    assert rate <= ops2 + 0.002
    assert rate < peg2


def test_desaturate_h1_cap_not_stuck_at_005():
    """Rejected H1 run: rate_cap already 0.05 must still propose a lower ceiling."""
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    diag = build_self_diagnosis(
        {
            "gate_failures": ["trade_rate_saturated"],
            "classification": {"roc_auc_ovr": 0.84, "accuracy": 0.75, "trade_rate_filtered": 0.054},
            "financial_oos": {
                "sharpe": 5.99,
                "sharpe_uncapped": 5.99,
                "trade_sharpe_raw": 0.65,
                "n_trades": 155,
                "expectancy": 0.0032,
            },
            "financial_train": {"sharpe": 6.9},
            "financial_validation": {"sharpe": 7.2},
            "financial_deploy_holdout": {"sharpe": 3.8, "n_trades": 43, "expectancy": 0.003},
            "fit_diagnosis": {"status": "balanced", "sharpe_gap_train_val": -0.37, "sharpe_gap_val_test": 1.25},
            "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {"inflated": False},
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 1.0},
            "stress_testing": {"robust": True, "worst_sharpe": 2.0},
        },
        timeframe="H1",
        passed_gates=False,
        cfg={
            "max_fold_trade_rate": 0.05,
            "max_fold_trade_rate_by_tf": {"H1": 0.05},
            "min_trade_confidence": 0.58,
            "confidence_quantile": 0.93,
            "directional_edge": 0.18,
        },
    )
    assert diag["primary_root_cause"] == "TradePolicy"
    knobs = diag["suggested_config_diff"]
    assert float(knobs["max_fold_trade_rate"]) < 0.05
    assert float(knobs["target_trade_rate"]) < float(knobs["max_fold_trade_rate"])
    assert knobs["quality_first_trade_policy"] is True


def test_m5_like_regime_sharpes_stable():
    """All-regime Sharpes 8.5–12.8 (M5) must be stable despite abs spread > 4."""
    from atis.engines.engine4_training.validation_protocols import evaluate_by_regime
    import numpy as np

    n = 400
    rets = np.zeros(n)
    masks = {
        "trending": np.zeros(n, dtype=bool),
        "ranging": np.zeros(n, dtype=bool),
        "high_volatility": np.zeros(n, dtype=bool),
        "low_volatility": np.zeros(n, dtype=bool),
    }
    canned = {
        "trending": 10.47,
        "ranging": 12.78,
        "high_volatility": 10.56,
        "low_volatility": 8.54,
    }
    for i, name in enumerate(masks):
        lo, hi = i * 100, (i + 1) * 100
        masks[name][lo:hi] = True
        rets[lo:hi:3] = 0.002

    def _fin(r):
        nz = np.flatnonzero(np.asarray(r) != 0)
        block = int(nz[0] // 100) if len(nz) else 0
        name = list(masks.keys())[block]
        sh = canned[name]
        return {
            "sharpe": sh,
            "sortino": sh * 2,
            "max_drawdown": -0.01,
            "profit_factor": 10.0,
            "expectancy": 0.003,
            "win_rate": 0.8,
        }

    out = evaluate_by_regime(rets, masks, financial_fn=_fin, min_bars=10)
    assert abs(out["sharpe_regime_spread"] - (12.78 - 8.54)) < 0.01
    assert out["stable"] is True
    assert out.get("sharpe_regime_rel_spread", 1.0) < 0.55


def test_regime_unstable_on_absolute_dispersion_when_floor_weak():
    from atis.engines.engine4_training.validation_protocols import evaluate_by_regime
    import numpy as np

    n = 200
    rets = np.zeros(n)
    masks = {
        "strong": np.zeros(n, dtype=bool),
        "weak": np.zeros(n, dtype=bool),
    }
    masks["strong"][:100] = True
    masks["weak"][100:] = True
    rets[:100:4] = 0.001
    rets[100::4] = 0.001
    sharpes = {"strong": 5.0, "weak": 0.5}

    def _fin(r):
        nz = np.flatnonzero(np.asarray(r) != 0)
        name = "strong" if len(nz) and nz[0] < 100 else "weak"
        sh = sharpes[name]
        return {
            "sharpe": sh,
            "sortino": sh,
            "max_drawdown": -0.05,
            "profit_factor": 1.2,
            "expectancy": 0.0001,
            "win_rate": 0.55,
        }

    out = evaluate_by_regime(rets, masks, financial_fn=_fin, min_bars=10)
    assert out["stable"] is False
    assert "large_sharpe_dispersion_across_regimes" in out["notes"]


def test_m30_val_cap_does_not_invent_train_val_overfit():
    """Capping Val for VT honesty must not create a false overfit_sharpe_gap (M30 100629)."""
    from atis.engines.engine4_training import diagnose_fit

    train_fin = {"sharpe": 12.30, "n_trades": 306}
    # Display/gate Val is honesty-capped; pre-cap fold median remains higher.
    val_fin = {
        "sharpe": 10.63,
        "sharpe_median_fold": 12.71,
        "sharpe_for_train_gap": 12.71,
        "sharpe_honesty": "median_fold_val_capped_by_test",
        "n_trades": 151,
    }
    test_fin = {"sharpe": 8.96, "n_trades": 294}
    cls = {"accuracy": 0.71, "roc_auc_ovr": 0.827, "trade_rate_filtered": 0.045}
    diag = diagnose_fit(cls, cls, cls, train_fin, val_fin, test_fin)
    assert diag["sharpe_gap_train_val"] < 0.5  # Train 12.3 vs raw Val 12.7
    assert diag["sharpe_gap_val_test"] < 2.0  # capped Val 10.63 vs Test 8.96
    assert diag["status"] != "overfitting"
    assert diag["status"] == "balanced"
    assert any("honesty-capped" in n for n in diag["notes"])

    # Without pre-cap fields, the same numbers falsely look like Train≫Val overfit.
    diag_old = diagnose_fit(
        cls,
        cls,
        cls,
        train_fin,
        {"sharpe": 10.63, "n_trades": 151},
        test_fin,
    )
    assert diag_old["status"] == "overfitting"
    assert diag_old["sharpe_gap_train_val"] > 1.5


def test_m30_val_test_gap_caps_fold_val_by_test():
    """M30 095842 fingerprint: tune-slice Val 12.7 vs fold Test 1–3 must not hard-fail."""
    import numpy as np
    from atis.engines.engine4_training import liquid_fold_val_sharpes
    from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis

    folds = [
        {"n_val_trades": 37, "val_sharpe": 12.71, "test_sharpe": 3.10, "policy": {"val_sharpe": 12.71}},
        {"n_val_trades": 37, "val_sharpe": 12.71, "test_sharpe": 1.60, "policy": {"val_sharpe": 12.71}},
        {"n_val_trades": 47, "val_sharpe": 12.51, "test_sharpe": 10.18, "policy": {"val_sharpe": 12.51}},
        {"n_val_trades": 47, "val_sharpe": 12.51, "test_sharpe": -0.19, "policy": {"val_sharpe": 12.51}},
        {"n_val_trades": 40, "val_sharpe": 13.51, "test_sharpe": 12.30, "policy": {"val_sharpe": 13.51}},
        {"n_val_trades": 40, "val_sharpe": 13.51, "test_sharpe": 8.16, "policy": {"val_sharpe": 13.51}},
        {"n_val_trades": 27, "val_sharpe": 14.01, "test_sharpe": 8.13, "policy": {"val_sharpe": 14.01}},
    ]
    raw, capped = liquid_fold_val_sharpes(folds, min_val_trades=14, cap_by_fold_test=True, test_slack=2.5)
    assert abs(float(np.median(raw)) - 12.71) < 0.05
    assert float(np.median(capped)) < 11.0
    test_sh, deploy_sh = 8.96, 7.63
    honest_oos = max(test_sh, deploy_sh)
    assert float(np.median(capped)) - honest_oos < 3.5

    diag = build_self_diagnosis(
        {
            "gate_failures": ["val_test_gap_hard"],
            "classification": {"roc_auc_ovr": 0.827, "accuracy": 0.712, "trade_rate_filtered": 0.045},
            "financial_oos": {
                "sharpe": 8.96,
                "sharpe_uncapped": 8.96,
                "trade_sharpe_raw": 0.70,
                "n_trades": 294,
                "expectancy": 0.0035,
            },
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": -0.55,
                "sharpe_gap_val_test": 3.90,
            },
            "fold_stability": {"trade_rate_pegged": False, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {"inflated": False},
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.25}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 1.0},
        },
        timeframe="M30",
        passed_gates=False,
        cfg={"policy_min_agree_folds": 3},
    )
    assert diag["primary_root_cause"] == "RegimeShift"
    knobs = diag["suggested_config_diff"]
    assert knobs.get("honest_val_cap_by_fold_test") is True
    assert float(knobs.get("honest_val_fold_test_slack", 99)) <= 2.5
    assert knobs.get("tune_policy_mode") == "each"

