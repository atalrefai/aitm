"""Tests for Engine 5 multi-TF decision wiring and live risk/spread gates."""

from types import SimpleNamespace

from atis.engines.engine4_training.multi_tf_decision import (
    confirm_tfs_for_primary,
    multi_tf_decision,
)
from atis.engines.engine5_live_trading import (
    RiskManager,
    entries_allowed_for_spread,
    filter_atis_positions,
    position_size_lots,
)


def test_position_size_positive() -> None:
    lots = position_size_lots(10_000, 1.0, 1.1000, 1.0900)
    assert lots >= 0.01


def test_position_size_scales_with_risk() -> None:
    a = position_size_lots(10_000, 1.0, 1.10, 1.09)
    b = position_size_lots(10_000, 2.0, 1.10, 1.09)
    assert b >= a


def test_confirm_map_for_fast_timeframes() -> None:
    cfg = {
        "confirm_by_primary_tf": {
            "M1": ["M5", "M15", "H1"],
            "M5": ["M15", "M30", "H1"],
            "M15": ["M30", "H1", "H4"],
            "H1": ["H4"],
        },
        "confirm_timeframes": ["H4", "H1"],
    }
    assert confirm_tfs_for_primary("M1", cfg) == ["M5", "M15", "H1"]
    assert confirm_tfs_for_primary("M5", cfg) == ["M15", "M30", "H1"]
    assert confirm_tfs_for_primary("H1", cfg) == ["H4"]


def test_soft_veto_allows_agreeing_confirm() -> None:
    pred, dbg = multi_tf_decision(
        1,
        0.72,
        [{"tf": "M15", "pred": 1, "conf": 0.6}, {"tf": "H1", "pred": 0, "conf": 0.4}],
        mode="soft_veto",
        primary_tf="M5",
    )
    assert pred == 1
    assert dbg["reason"] in {"soft_agree", "soft_no_opposition", "soft_strong_primary"}


def test_filter_atis_positions_by_magic_and_symbol() -> None:
    positions = [
        SimpleNamespace(magic=260729, symbol="XAUUSD"),
        SimpleNamespace(magic=260729, symbol="XAUUSDm"),
        SimpleNamespace(magic=999, symbol="XAUUSD"),
        SimpleNamespace(magic=260729, symbol="EURUSD"),
    ]
    atis = filter_atis_positions(positions, "XAUUSD", atis_only=True)
    assert len(atis) == 2
    assert all(p.magic == 260729 for p in atis)


def test_filter_atis_positions_by_timeframe_comment() -> None:
    positions = [
        SimpleNamespace(magic=260729, symbol="XAUUSD", comment="ATIS|H1"),
        SimpleNamespace(magic=260729, symbol="XAUUSD", comment="ATIS|M15|engulf"),
        SimpleNamespace(magic=260729, symbol="XAUUSD", comment="ATIS|H1|tag"),
        SimpleNamespace(magic=260729, symbol="XAUUSD", comment="OTHER|H1"),
    ]
    h1 = filter_atis_positions(positions, "XAUUSD", atis_only=True, timeframe="H1")
    m15 = filter_atis_positions(positions, "XAUUSD", atis_only=True, timeframe="M15")
    assert len(h1) == 2
    assert len(m15) == 1
    assert all("H1" in (p.comment or "") for p in h1)


def test_entries_allowed_blocks_wide_spread() -> None:
    cfg = {
        "use_live_spread_filter": True,
        "max_entry_spread_pips": 25.0,
        "tight_spread_pips": 12.0,
        "max_entries_per_cycle": 8,
        "scale_in_when_tight_spread": True,
    }
    n, reason = entries_allowed_for_spread(40.0, cfg)
    assert n == 0
    assert reason == "spread_too_wide"


def test_entries_allowed_full_batch_on_tight_spread() -> None:
    cfg = {
        "use_live_spread_filter": True,
        "max_entry_spread_pips": 25.0,
        "tight_spread_pips": 12.0,
        "max_entries_per_cycle": 8,
        "scale_in_when_tight_spread": True,
    }
    n, reason = entries_allowed_for_spread(10.0, cfg)
    assert n == 8
    assert reason == "ok_tight_full"


def test_entries_spread_filter_off_ignores_wide_spread() -> None:
    cfg = {
        "use_live_spread_filter": False,
        "max_entry_spread_pips": 12.0,
        "tight_spread_pips": 12.0,
        "max_entries_per_cycle": 8,
        "scale_in_when_tight_spread": True,
    }
    n, reason = entries_allowed_for_spread(99.0, cfg)
    assert n == 1
    assert reason == "spread_filter_off"


def test_entries_allowed_scales_between_tight_and_max() -> None:
    cfg = {
        "use_live_spread_filter": True,
        "max_entry_spread_pips": 25.0,
        "tight_spread_pips": 12.0,
        "max_entries_per_cycle": 8,
        "scale_in_when_tight_spread": True,
    }
    n, reason = entries_allowed_for_spread(18.5, cfg)
    assert 1 <= n <= 8
    assert reason == "ok_scaled"


def test_risk_blocks_max_exposure() -> None:
    risk = RiskManager()
    risk.max_open = 50
    risk.max_exposure_pct = 1.0
    risk.risk_per_trade = 0.35
    ok, reason = risk.allow_new_trade(
        10_000.0,
        open_positions=2,
        planned_risk_pct=0.35,
        open_risk_pct=0.70,
    )
    assert ok is False
    assert reason == "max_exposure_pct"
