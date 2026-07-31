"""Tests for FinalModel selection and publishing."""

from __future__ import annotations

from atis.engines.engine4_training.final_model import select_best_result


def test_select_best_result_prefers_higher_sharpe():
    results = [
        {
            "timeframe": "M15",
            "model_path": "a/model.joblib",
            "metrics": {"financial_oos": {"sharpe": -3.9, "max_drawdown": -0.29, "total_return": -0.28, "n_trades": 500}},
        },
        {
            "timeframe": "H1",
            "model_path": "b/model.joblib",
            "metrics": {"financial_oos": {"sharpe": -0.39, "max_drawdown": -0.08, "total_return": -0.01, "n_trades": 160}},
        },
        {
            "timeframe": "H4",
            "error": "boom",
            "model_path": "c/model.joblib",
            "metrics": {"financial_oos": {"sharpe": 9.0}},
        },
    ]
    best = select_best_result(results, allow_paper_final=True)
    assert best is not None
    assert best["timeframe"] == "H1"


def test_select_best_result_prefers_gated_model():
    results = [
        {
            "timeframe": "H1",
            "model_path": "a/model.joblib",
            "passed_gates": False,
            "metrics": {"financial_oos": {"sharpe": 0.9, "max_drawdown": -0.05, "total_return": 0.02, "n_trades": 20}},
        },
        {
            "timeframe": "M30",
            "model_path": "b/model.joblib",
            "passed_gates": True,
            "metrics": {"financial_oos": {"sharpe": 0.5, "max_drawdown": -0.02, "total_return": 0.03, "n_trades": 24}},
        },
    ]
    best = select_best_result(results)
    assert best is not None
    assert best["timeframe"] == "M30"


def test_select_best_result_prefers_liquid_tf_on_similar_losses():
    results = [
        {
            "timeframe": "M15",
            "model_path": "a/model.joblib",
            "metrics": {"financial_oos": {"sharpe": -1.14, "max_drawdown": -0.26, "total_return": -0.22, "n_trades": 900}},
        },
        {
            "timeframe": "H1",
            "model_path": "b/model.joblib",
            "metrics": {"financial_oos": {"sharpe": -1.20, "max_drawdown": -0.10, "total_return": -0.08, "n_trades": 80}},
        },
    ]
    best = select_best_result(results, allow_paper_final=True)
    assert best is not None
    assert best["timeframe"] == "H1"


def test_select_best_result_ignores_one_trade_deploy_sharpe():
    """Report case: H4 deploy sharpe=1.63 on 1 trade must not beat liquid M30."""
    results = [
        {
            "timeframe": "H4",
            "model_path": "h4/model.joblib",
            "passed_gates": True,
            "metrics": {
                "financial_oos": {"sharpe": 2.05, "max_drawdown": -0.043, "total_return": 0.276, "n_trades": 12},
                "financial_deploy_holdout": {"sharpe": 1.6273, "n_trades": 1},
            },
        },
        {
            "timeframe": "M30",
            "model_path": "m30/model.joblib",
            "passed_gates": True,
            "metrics": {
                "financial_oos": {"sharpe": 0.837, "max_drawdown": -0.0086, "total_return": 0.0234, "n_trades": 40},
                "financial_deploy_holdout": {"sharpe": 0.4, "n_trades": 12},
            },
        },
    ]
    best = select_best_result(results, min_deploy_trades=8)
    assert best is not None
    assert best["timeframe"] == "M30"
