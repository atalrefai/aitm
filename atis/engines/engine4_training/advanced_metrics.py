"""Trading-centric metrics beyond accuracy.

References:
- Sharpe / Sortino — Lo (2002); Sortino & van der Meer (1991).
- Profit factor / expectancy — Van Tharp position-sizing literature; retail
  quant practice aligned with institutional PnL attribution.
- Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).
- Probability of Backtest Overfitting (PBO) — Bailey et al. (2014), AFML Ch.11.
- Risk-adjusted return — excess return / |max DD| (Calmar-like) and Sortino.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy required)."""
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer/Moro-style rational)."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    # Ackley approximation
    t = np.sqrt(-2.0 * np.log(min(p, 1.0 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return float(-z if p < 0.5 else z)


def trade_expectancy(trade_returns: np.ndarray) -> dict[str, float]:
    """Classic expectancy: E[R] = win_rate * avg_win - loss_rate * avg_loss."""
    x = np.asarray(trade_returns, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return {
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "kelly_fraction_approx": 0.0,
        }
    wins = x[x > 0]
    losses = x[x < 0]
    wr = float(len(wins) / len(x))
    lr = 1.0 - wr
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) else 0.0
    exp = wr * avg_win - lr * avg_loss
    payoff = (avg_win / avg_loss) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)
    # Approximate Kelly for binary edge: f* = wr - (1-wr)/payoff
    kelly = 0.0
    if payoff > 0:
        kelly = float(np.clip(wr - (1.0 - wr) / payoff, -1.0, 1.0))
    return {
        "expectancy": float(exp),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(payoff),
        "kelly_fraction_approx": kelly,
    }


def risk_adjusted_return(
    *,
    total_return: float,
    max_drawdown: float,
    sharpe: float,
    sortino: float,
) -> dict[str, float]:
    """Composite risk-adjusted scores used in hedge-fund style scorecards."""
    dd = abs(float(max_drawdown))
    rar_dd = float(total_return) / dd if dd > 1e-12 else 0.0
    # Soft blend of Sharpe and Sortino (Sortino emphasizes downside)
    blended = 0.55 * float(sharpe) + 0.45 * float(sortino)
    return {
        "return_over_drawdown": rar_dd,
        "risk_adjusted_score": float(blended),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
    }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> dict[str, float]:
    """DSR ≈ Prob(SR* > SR_benchmark | selection bias from n_trials).

    Bailey & López de Prado (2014). Conservative when n_trials or non-normality rise.
    """
    sr = float(observed_sharpe)
    n = max(int(n_obs), 2)
    trials = max(int(n_trials), 1)
    # Expected max Sharpe under null of SR=0 across independent trials (Euler-Maclaurin approx)
    eulers = 0.5772156649
    z = _norm_ppf(1.0 - 1.0 / trials) if trials > 1 else 0.0
    sr_max_expected = float(
        ((1.0 - eulers) * z + eulers * _norm_ppf(1.0 - 1.0 / (trials * np.e)))
        if trials > 1
        else 0.0
    )
    # Variance of Sharpe estimator under non-normality (Lo / Mertens)
    sr_var = (
        1.0
        - skew * sr
        + ((kurtosis - 1.0) / 4.0) * (sr**2)
    ) / (n - 1)
    sr_var = max(float(sr_var), 1e-12)
    sr_std = float(np.sqrt(sr_var))
    # Deflated: how much observed exceeds expected max under multiple testing
    bench = max(float(sharpe_benchmark), sr_max_expected)
    dsr = float(_norm_cdf((sr - bench) / sr_std))
    return {
        "deflated_sharpe": dsr,
        "observed_sharpe": sr,
        "expected_max_sharpe_null": sr_max_expected,
        "sharpe_std": sr_std,
        "n_trials": float(trials),
        "n_obs": float(n),
    }


def probability_of_backtest_overfitting(
    path_is_sharpes: list[float],
    path_oos_sharpes: list[float],
) -> dict[str, float]:
    """PBO lite: fraction of paths where relative rank IS ≫ OOS (AFML Ch.11).

    For each combinatorial path, if the in-sample Sharpe rank exceeds OOS rank
    in the upper half while OOS is weak, count as overfit token.
    """
    is_s = np.asarray(path_is_sharpes, dtype=float)
    oos_s = np.asarray(path_oos_sharpes, dtype=float)
    m = min(len(is_s), len(oos_s))
    if m < 4:
        return {"pbo": 0.5, "n_paths": float(m), "reliable": 0.0}
    is_s, oos_s = is_s[:m], oos_s[:m]
    # Rank correlation failure: high IS with low OOS
    is_rank = np.argsort(np.argsort(is_s)).astype(float) + 1.0
    oos_rank = np.argsort(np.argsort(oos_s)).astype(float) + 1.0
    # PBO: P(OOS rank < median | IS rank > median)
    high_is = is_rank >= np.median(is_rank)
    if not np.any(high_is):
        return {"pbo": 0.5, "n_paths": float(m), "reliable": 0.0}
    fail = np.sum(high_is & (oos_rank < np.median(oos_rank))) / float(np.sum(high_is))
    return {
        "pbo": float(fail),
        "n_paths": float(m),
        "mean_is_sharpe": float(np.mean(is_s)),
        "mean_oos_sharpe": float(np.mean(oos_s)),
        "reliable": 1.0,
    }


def enrich_financial_metrics(base: dict[str, float], returns: np.ndarray) -> dict[str, float]:
    """Attach expectancy + risk-adjusted fields onto an existing financial dict."""
    out = dict(base)
    traded = np.asarray(returns, dtype=float)
    traded = traded[np.isfinite(traded) & (traded != 0)]
    exp = trade_expectancy(traded)
    out.update(exp)
    rar = risk_adjusted_return(
        total_return=float(out.get("sum_trade_returns", out.get("total_return", 0.0)) or 0.0),
        max_drawdown=float(out.get("max_drawdown", 0.0) or 0.0),
        sharpe=float(out.get("sharpe", 0.0) or 0.0),
        sortino=float(out.get("sortino", 0.0) or 0.0),
    )
    out["risk_adjusted_return"] = rar["return_over_drawdown"]
    out["risk_adjusted_score"] = rar["risk_adjusted_score"]
    return out


def metrics_rationale() -> dict[str, str]:
    return {
        "expectancy": "Average $ edge per trade; required for position sizing (Van Tharp).",
        "sortino": "Downside-deviation risk; preferred when returns asymmetric.",
        "profit_factor": "Gross profit / gross loss; robustness of payoff structure.",
        "deflated_sharpe": "Corrects Sharpe for multiple testing / selection bias (DSR).",
        "pbo": "Probability that relative IS ranking fails OOS (backtest overfitting).",
        "risk_adjusted_return": "Return per unit of drawdown — capacity / pain awareness.",
    }
