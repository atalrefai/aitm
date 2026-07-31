"""Unit tests for Trading Intelligence Engine (e4-v13) modules."""

from __future__ import annotations

import numpy as np


def test_pipeline_version_v13():
    from atis.engines.engine4_training import PIPELINE_VERSION

    assert (
        "e4-v13" in PIPELINE_VERSION
        or "e4-v14" in PIPELINE_VERSION
        or "trading-intelligence" in PIPELINE_VERSION
        or "enterprise" in PIPELINE_VERSION
    )


def test_rolling_and_expanding_splits_causal():
    from atis.engines.engine4_training.validation_protocols import (
        build_validation_splits,
        rolling_window_splits,
    )

    n = 500
    exp = build_validation_splits(n, mode="expanding", n_splits=4, train_ratio=0.7, embargo=6, purge=6)
    roll = rolling_window_splits(n, 4, train_size=180, test_size=40, embargo=6, purge=6)
    assert exp and roll
    for tr, te in exp + roll:
        assert tr[-1] < te[0]
        assert len(tr) >= 20 and len(te) >= 5


def test_regime_masks_and_eval():
    from atis.engines.engine4_training.validation_protocols import (
        classify_market_regimes,
        evaluate_by_regime,
    )
    from atis.engines.engine4_training import financial_metrics

    rng = np.random.default_rng(0)
    n = 800
    close = 2000 + np.cumsum(rng.normal(0, 1.5, n))
    atr = np.abs(rng.normal(0.002, 0.0005, n))
    masks = classify_market_regimes(close, atr, trend_window=40)
    assert set(masks) >= {"trending", "ranging", "high_volatility", "low_volatility"}
    rets = rng.normal(0.0002, 0.01, n)
    rets[::7] = 0.0
    out = evaluate_by_regime(rets, masks, financial_fn=lambda r: financial_metrics(r, hold_bars=6))
    assert "regimes" in out
    assert "stable" in out


def test_expectancy_and_dsr():
    from atis.engines.engine4_training.advanced_metrics import (
        trade_expectancy,
        deflated_sharpe_ratio,
        probability_of_backtest_overfitting,
        enrich_financial_metrics,
    )

    x = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.004])
    exp = trade_expectancy(x)
    assert exp["expectancy"] != 0.0
    assert exp["payoff_ratio"] > 0
    dsr = deflated_sharpe_ratio(1.2, n_trials=10, n_obs=80)
    assert 0.0 <= dsr["deflated_sharpe"] <= 1.0
    pbo = probability_of_backtest_overfitting([2, 1.5, 1.0, 0.5, 0.2], [0.2, 0.5, 1.0, 0.1, -0.2])
    assert "pbo" in pbo
    enriched = enrich_financial_metrics({"sharpe": 1.0, "sortino": 1.2, "max_drawdown": -0.1}, x)
    assert "expectancy" in enriched
    assert "risk_adjusted_return" in enriched


def test_execution_latency_reduces_or_shifts_fills():
    from atis.engines.engine4_training.execution_realism import simulate_trade_returns

    close = np.linspace(2000, 2100, 200)
    preds = np.zeros(200)
    preds[10] = 1
    preds[50] = -1
    preds[90] = 1
    rets0, st0 = simulate_trade_returns(
        close,
        preds,
        hold_bars=5,
        spread_pips=30,
        slippage_pips=5,
        commission_per_lot=7,
        pip_size=0.01,
        latency_bars=0,
        dynamic_costs=False,
    )
    rets1, st1 = simulate_trade_returns(
        close,
        preds,
        hold_bars=5,
        spread_pips=30,
        slippage_pips=5,
        commission_per_lot=7,
        pip_size=0.01,
        latency_bars=2,
        execution_delay_bars=1,
        dynamic_costs=False,
    )
    assert st0["trades"] >= 1
    assert st1["latency_bars"] == 2.0
    assert st1["execution_delay_bars"] == 1.0
    # Different fill timing → different attributed returns path
    assert not np.allclose(rets0, rets1)


def test_financial_metrics_includes_expectancy():
    from atis.engines.engine4_training import financial_metrics

    r = np.zeros(100)
    r[::10] = 0.002
    r[5::10] = -0.001
    m = financial_metrics(r, periods_per_year=252 * 24, hold_bars=6, ann_cap="daily")
    assert "expectancy" in m
    assert "sortino" in m
    assert "profit_factor" in m
    assert "risk_adjusted_return" in m


def test_ensemble_builds():
    from atis.engines.engine4_training.ensemble_models import build_soft_voting_ensemble, blend_probas
    from atis.engines.engine4_training import build_model

    model = build_soft_voting_ensemble(seed=0, cfg={"lgb_estimators": 20, "lgb_max_depth": 3})
    assert model is not None
    m2 = build_model("ensemble", seed=0, cfg={"lgb_estimators": 20})
    assert m2 is not None
    a = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    b = np.array([[0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
    c = blend_probas([a, b], weights=[0.5, 0.5])
    assert c.shape == a.shape
    assert np.allclose(c.sum(axis=1), 1.0)


def test_knowledge_loop_records_episode(tmp_path):
    from atis.engines.engine4_training.knowledge_loop import (
        record_training_episode,
        load_knowledge,
        knowledge_store_path,
        population_stability_index,
    )

    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.5, 1.2, 200)
    assert population_stability_index(a, b) >= 0.0
    metrics = {
        "pipeline_version": "e4-v13-test",
        "timeframe": "H1",
        "financial_oos": {"sharpe": 0.8, "expectancy": 0.001, "sortino": 1.0, "max_drawdown": -0.05, "n_trades": 40},
        "fit_diagnosis": {"status": "balanced"},
        "classification": {"roc_auc_ovr": 0.6, "accuracy": 0.55},
        "gate_failures": [],
        "regime_validation": {"stable": True},
    }
    store = record_training_episode(
        tmp_path,
        symbol="XAUUSD",
        timeframe="H1",
        version="vtest",
        metrics=metrics,
        passed_gates=True,
        feature_psi=0.05,
    )
    path = knowledge_store_path(tmp_path, "XAUUSD", "H1")
    assert path.exists()
    loaded = load_knowledge(path)
    assert len(loaded["episodes"]) >= 1
    assert loaded["last_advisory"]["retrain_suggested"] is False
