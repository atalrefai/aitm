"""Unit tests for model-driven near-entry dynamic SL/TP exits."""

from __future__ import annotations

import pandas as pd
import pytest

from atis.engines.engine5_live_trading.dynamic_exits import (
    aggregate_prediction_exits,
    compute_dynamic_sl_tp,
)


def _frame(
    *,
    close: float = 4000.0,
    atr: float = 20.0,
    support: float | None = 3975.0,
    resist: float | None = 4040.0,
    trend_strength: float = 25.0,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    if highs is not None and lows is not None:
        n = len(highs)
        return pd.DataFrame(
            {
                "close": [close] * n,
                "high": highs,
                "low": lows,
                "atr": [atr] * n,
                "trend_strength": [trend_strength] * n,
                "adx": [28.0] * n,
                "support_level": [support if support is not None else close - atr] * n,
                "resist_level": [resist if resist is not None else close + atr] * n,
            }
        )
    row = {
        "close": close,
        "atr": atr,
        "trend_strength": trend_strength,
        "adx": 28.0,
    }
    if support is not None:
        row["support_level"] = support
    if resist is not None:
        row["resist_level"] = resist
    return pd.DataFrame([row])


def test_tp_uses_expected_return_for_buy() -> None:
    featured = _frame(support=3970.0, resist=4200.0)  # resist far → ignored
    exits = compute_dynamic_sl_tp(
        price=4000.0,
        side="buy",
        atr_value=20.0,
        confidence=0.80,
        featured=featured,
        expected_return=0.01,  # +1% → 40 points raw
        risk_score=0.2,
        cfg={"structure_tp_snap": False, "min_rr": 1.0},
    )
    assert exits.tp > 4000.0
    assert exits.sl < 4000.0
    assert "model_expected_return" in exits.method
    # Reach fraction < 1, so TP move < full 40; proximity clamp ≤ 2.4×ATR
    assert exits.tp_distance < 40.0
    assert exits.tp_distance <= 2.4 * 20.0 + 1e-9
    assert exits.tp_distance > 10.0


def test_tp_uses_expected_return_for_sell() -> None:
    exits = compute_dynamic_sl_tp(
        price=4000.0,
        side="sell",
        atr_value=20.0,
        confidence=0.75,
        featured=_frame(support=3800.0, resist=4030.0),
        expected_return=-0.008,
        risk_score=0.3,
        cfg={"structure_tp_snap": False, "min_rr": 1.0},
    )
    assert exits.tp < 4000.0
    assert exits.sl > 4000.0
    assert "model_expected_return" in exits.method


def test_misaligned_expected_return_falls_back_to_atr() -> None:
    exits = compute_dynamic_sl_tp(
        price=4000.0,
        side="buy",
        atr_value=20.0,
        confidence=0.70,
        featured=_frame(),
        expected_return=-0.01,  # wrong sign for buy
        risk_score=0.4,
        cfg={"structure_tp_snap": False, "min_rr": 1.0},
    )
    assert "atr_confidence_fallback" in exits.method
    assert "expected_return_misaligned" in exits.method


def test_structure_sl_preferred_when_valid() -> None:
    # Support ~1 ATR below → structural invalidation
    featured = _frame(support=3978.0, resist=4025.0)
    exits = compute_dynamic_sl_tp(
        price=4000.0,
        side="buy",
        atr_value=20.0,
        confidence=0.70,
        featured=featured,
        expected_return=0.012,
        risk_score=0.25,
        cfg={"structure_tp_snap": False, "min_rr": 1.0, "structure_buffer_atr": 0.15},
    )
    assert "structure_sl" in exits.method
    # SL should sit near / beyond support - buffer
    assert exits.sl < 3978.0


def test_distant_structure_is_ignored() -> None:
    # Far historical support must not become TP on a sell.
    featured = _frame(support=3900.0, resist=4100.0)  # 5×ATR away
    exits = compute_dynamic_sl_tp(
        price=4000.0,
        side="sell",
        atr_value=20.0,
        confidence=0.70,
        featured=featured,
        expected_return=-0.008,
        risk_score=0.3,
        cfg={"min_rr": 1.0, "structure_reach_max_atr": 2.2, "structure_tp_max_atr": 2.0},
    )
    assert exits.meta["support_level"] is None
    assert exits.meta["resist_level"] is None
    assert "structure_tp" not in exits.method
    assert exits.tp_distance <= 2.4 * 20.0 + 1e-9
    assert exits.sl_distance <= 1.8 * 20.0 + 1e-9


def test_local_swing_preferred_over_distant_chart_level() -> None:
    # Chart support is far; recent lows provide a near-entry structure stop/target.
    featured = _frame(
        support=3800.0,
        resist=4200.0,
        highs=[4010, 4008, 4012, 4006, 4005],
        lows=[3990, 3988, 3992, 3985, 3987],
    )
    exits = compute_dynamic_sl_tp(
        price=4000.0,
        side="sell",
        atr_value=20.0,
        confidence=0.75,
        featured=featured,
        expected_return=-0.006,
        risk_score=0.3,
        cfg={
            "min_rr": 1.0,
            "local_swing_lookback": 5,
            "structure_reach_max_atr": 2.2,
            "structure_tp_max_atr": 2.0,
            "structure_buffer_atr": 0.1,
        },
    )
    assert exits.meta["local_support"] is not None
    assert exits.meta["local_support"] > 3980.0
    assert abs(exits.tp - exits.meta["local_support"]) < 5.0 or "structure_tp" in exits.method


def test_higher_confidence_widens_tp_vs_lower() -> None:
    featured = _frame(support=3975.0, resist=4025.0)
    low = compute_dynamic_sl_tp(
        price=4000.0,
        side="buy",
        atr_value=20.0,
        confidence=0.55,
        featured=featured,
        expected_return=0.015,
        risk_score=0.3,
        cfg={"structure_tp_snap": False, "min_rr": 1.0},
    )
    high = compute_dynamic_sl_tp(
        price=4000.0,
        side="buy",
        atr_value=20.0,
        confidence=0.95,
        featured=featured,
        expected_return=0.015,
        risk_score=0.3,
        cfg={"structure_tp_snap": False, "min_rr": 1.0},
    )
    assert high.tp_distance > low.tp_distance


def test_rr_gate_skips_when_impossible() -> None:
    with pytest.raises(ValueError, match="rr_below_min"):
        compute_dynamic_sl_tp(
            price=4000.0,
            side="buy",
            atr_value=20.0,
            confidence=0.52,
            featured=_frame(support=3975.0, resist=4025.0),
            expected_return=0.001,  # tiny target
            risk_score=0.9,  # wide stop pressure
            cfg={
                "structure_tp_snap": False,
                "min_rr": 3.0,
                "skip_if_rr_below_min": True,
                "tp_atr_clamp_min": 0.3,
                "tp_atr_clamp_max": 0.5,
                "sl_atr_clamp_min": 2.0,
                "sl_atr_clamp_max": 3.0,
            },
        )


def test_aggregate_prediction_exits_weights_agreeing_votes() -> None:
    votes = [
        {"tf": "M15", "pred": 1, "conf": 0.8, "expected_return": 0.01, "risk_score": 0.2},
        {"tf": "H1", "pred": 1, "conf": 0.4, "expected_return": 0.004, "risk_score": 0.4},
        {"tf": "H4", "pred": -1, "conf": 0.9, "expected_return": -0.02, "risk_score": 0.1},
    ]
    agg = aggregate_prediction_exits(votes, fused_pred=1)
    assert agg["exit_vote_count"] == 2
    # Closer to 0.01 than 0.004 because of higher weight on M15
    assert agg["expected_return"] > 0.007
    assert 0.2 < agg["risk_score"] < 0.35
