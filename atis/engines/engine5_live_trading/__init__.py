"""Engine 5 — Live Trading (Paper / Demo only by default)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from atis.config import (
    PROJECT_ROOT,
    ensure_project_dirs,
    get_path,
    load_engine_config,
    load_timeframes,
    set_global_seed,
)
from atis.engines.engine4_training.deep_learning import HAS_TORCH, load_llmodel, predict_with_llmodel
from atis.engines.engine2_cleaning import clean_dataframe
from atis.engines.engine3_features import features_parquet_path
from atis.engines.engine5_live_trading.dynamic_exits import (
    aggregate_prediction_exits,
    compute_dynamic_sl_tp,
)
from atis.shared.data_registry import DataStateRegistry
from atis.shared.feature_engine import compute_features, load_indicators_config
from atis.shared.logging_utils import get_logger
from atis.shared.mt5_client import MT5Client, mt5_session, _mt5_module

logger = get_logger("atis.engine5")

ATIS_MAGIC = 260729


@dataclass
class RiskState:
    day_key: str = ""
    week_key: str = ""
    day_pnl_pct: float = 0.0
    week_pnl_pct: float = 0.0
    open_positions: int = 0
    kill_switch: bool = False


@dataclass
class TradeRecord:
    ticket: int | None
    symbol: str
    side: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    confidence: float
    reason: str
    ts: str
    mode: str
    pattern_keys: list[str] = field(default_factory=list)
    pattern_ids: list[str] = field(default_factory=list)
    pattern_summary: str = ""


@dataclass
class LiveLoopReport:
    started_at: str
    finished_at: str | None = None
    iterations: int = 0
    signals: int = 0
    orders_sent: int = 0
    blocked_by_risk: int = 0
    errors: list[str] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cfg() -> dict[str, Any]:
    return load_engine_config().get("engine5_live", {})


def load_champion_or_latest(
    symbol: str,
    timeframe: str,
    *,
    allow_ungated: bool = False,
    match_timeframe_only: bool = False,
) -> dict[str, Any]:
    """Load FinalModel (preferred), then champion, then newest version for paper mode.

    When ``match_timeframe_only`` is True (multi-TF analysis), FinalModel is used
    only if its training timeframe matches ``timeframe`` — never applied across TFs.
    """
    cfg = _cfg()
    timeframe = str(timeframe).upper()

    if bool(cfg.get("prefer_final_model", True)):
        final_dir = get_path("models") / "FinalModel"
        model_path = final_dir / "model.joblib"
        meta_path = final_dir / "FINAL_MODEL.json"
        if model_path.exists():
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            meta_tf = str(meta.get("timeframe") or "").upper()
            allow_any = bool(cfg.get("final_model_any_timeframe", True)) and not match_timeframe_only
            if not meta_tf or meta_tf == timeframe or allow_any:
                # Still refuse cross-TF when match_timeframe_only is set.
                if match_timeframe_only and meta_tf and meta_tf != timeframe:
                    pass
                else:
                    bundle = joblib.load(model_path)
                    gated = bool(meta.get("passed_gates", True))
                    bundle["version"] = meta.get("version", "FinalModel")
                    bundle["gated"] = gated
                    bundle["artifact_type"] = "final_model"
                    bundle["symbol"] = meta.get("symbol", symbol)
                    bundle["timeframe"] = meta.get("timeframe", timeframe)
                    if gated or allow_ungated:
                        logger.info(
                            "using_final_model",
                            version=bundle["version"],
                            timeframe=bundle.get("timeframe"),
                            requested_tf=timeframe,
                            gated=gated,
                        )
                        return bundle

    llmodel_path = get_path("models") / "LLModel"
    if bool(cfg.get("prefer_llmodel", True)) and llmodel_path.exists() and not match_timeframe_only:
        if not HAS_TORCH:
            raise RuntimeError("LLModel موجود لكن torch غير مثبت")
        bundle = load_llmodel(llmodel_path)
        bundle["version"] = "LLModel"
        bundle["gated"] = True
        bundle["artifact_type"] = "llmodel"
        return bundle

    base = get_path("models") / symbol / timeframe
    champ_path = base / "champion.json"
    if champ_path.exists():
        meta = json.loads(champ_path.read_text(encoding="utf-8"))
        bundle = joblib.load(meta["model_path"])
        bundle["version"] = meta.get("version")
        bundle["gated"] = True
        bundle["artifact_type"] = bundle.get("artifact_type") or "champion"
        bundle["timeframe"] = timeframe
        bundle["symbol"] = symbol
        return bundle

    if not allow_ungated:
        raise FileNotFoundError(
            f"No champion model for {symbol}/{timeframe}. "
            "Train a gated model or pass allow_ungated for paper mode."
        )

    if not base.exists():
        raise FileNotFoundError(f"No models directory for {symbol}/{timeframe}")
    versions = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    if not versions:
        raise FileNotFoundError(f"No model versions for {symbol}/{timeframe}")
    model_path = versions[0] / "model.joblib"
    bundle = joblib.load(model_path)
    bundle["version"] = versions[0].name
    bundle["gated"] = False
    bundle["artifact_type"] = "ungated"
    bundle["timeframe"] = timeframe
    bundle["symbol"] = symbol
    logger.warning("using_ungated_model", symbol=symbol, timeframe=timeframe, version=bundle["version"])
    return bundle


def load_model_for_timeframe(
    symbol: str,
    timeframe: str,
    *,
    allow_ungated: bool = True,
) -> dict[str, Any]:
    """Load the trained model that belongs to this timeframe (training/test TF)."""
    return load_champion_or_latest(
        symbol,
        timeframe,
        allow_ungated=allow_ungated,
        match_timeframe_only=True,
    )


def _timeframes_for_llmodel(bundle: dict[str, Any], fallback_timeframe: str) -> list[str]:
    if bundle.get("artifact_type") == "llmodel":
        return list(bundle.get("timeframes") or [fallback_timeframe])
    return [fallback_timeframe]


def _fetch_feature_frames_for_bundle(
    client: MT5Client,
    symbol: str,
    bundle: dict[str, Any],
    fallback_timeframe: str,
    bars: int = 320,
) -> dict[str, pd.DataFrame]:
    ind_cfg = load_indicators_config()
    frames: dict[str, pd.DataFrame] = {}
    requested = _timeframes_for_llmodel(bundle, fallback_timeframe)
    known_tfs = load_timeframes()
    for tf in requested:
        if tf not in known_tfs:
            raise ValueError(f"unknown_timeframe:{tf}")
        raw = fetch_recent_bars(client, symbol, tf, bars=bars)
        if raw.empty:
            raise ValueError(f"no_bars:{tf}")
        cleaned, _ = clean_dataframe(raw, tf)
        frames[tf] = compute_features(cleaned, ind_cfg)
    return frames


def position_size_lots(
    balance: float,
    risk_pct: float,
    entry: float,
    sl: float,
    *,
    contract_size: float = 100000.0,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
) -> float:
    """Risk a fixed % of equity based on SL distance."""
    risk_amount = balance * (risk_pct / 100.0)
    stop_dist = abs(entry - sl)
    if stop_dist <= 0 or entry <= 0:
        return min_lot
    # loss per 1.0 lot ≈ stop_dist * contract_size
    lots = risk_amount / (stop_dist * contract_size)
    lots = max(min_lot, np.floor(lots / lot_step) * lot_step)
    return float(round(lots, 2))


class RiskManager:
    def __init__(self) -> None:
        cfg = _cfg()
        self.risk_per_trade = float(cfg.get("risk_per_trade_pct", 1.0))
        self.daily_limit = float(cfg.get("daily_loss_limit_pct", 3.0))
        self.weekly_limit = float(cfg.get("weekly_loss_limit_pct", 7.0))
        self.max_open = int(cfg.get("max_open_positions", 3))
        self.max_exposure_pct = float(cfg.get("max_exposure_pct", 10.0))
        self.state = RiskState(kill_switch=bool(cfg.get("kill_switch", False)))
        self._start_equity: float | None = None
        self._day_start_equity: float | None = None
        self._week_start_equity: float | None = None

    def update_equity(self, equity: float) -> None:
        now = _utc_now()
        day = now.strftime("%Y-%m-%d")
        week = now.strftime("%Y-W%W")
        if self._start_equity is None:
            self._start_equity = equity
        if self.state.day_key != day:
            self.state.day_key = day
            self._day_start_equity = equity
            self.state.day_pnl_pct = 0.0
        if self.state.week_key != week:
            self.state.week_key = week
            self._week_start_equity = equity
            self.state.week_pnl_pct = 0.0
        if self._day_start_equity:
            self.state.day_pnl_pct = (equity / self._day_start_equity - 1.0) * 100.0
        if self._week_start_equity:
            self.state.week_pnl_pct = (equity / self._week_start_equity - 1.0) * 100.0

    def activate_kill_switch(self, reason: str) -> None:
        self.state.kill_switch = True
        logger.error("kill_switch_activated", reason=reason)
        # Persist
        path = PROJECT_ROOT / "logs" / "live" / "kill_switch.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"active": True, "reason": reason, "ts": _utc_now().isoformat()}, indent=2),
            encoding="utf-8",
        )

    def allow_new_trade(
        self,
        equity: float,
        open_positions: int,
        *,
        planned_risk_pct: float | None = None,
        open_risk_pct: float = 0.0,
    ) -> tuple[bool, str]:
        self.update_equity(equity)
        self.state.open_positions = open_positions
        if self.state.kill_switch or _cfg().get("kill_switch"):
            return False, "kill_switch"
        if self.state.day_pnl_pct <= -self.daily_limit:
            self.activate_kill_switch("daily_loss_limit")
            return False, "daily_loss_limit"
        if self.state.week_pnl_pct <= -self.weekly_limit:
            self.activate_kill_switch("weekly_loss_limit")
            return False, "weekly_loss_limit"
        if open_positions >= self.max_open:
            return False, "max_open_positions"
        add_risk = float(self.risk_per_trade if planned_risk_pct is None else planned_risk_pct)
        if open_risk_pct + add_risk > self.max_exposure_pct + 1e-9:
            return False, "max_exposure_pct"
        return True, "ok"


def _symbol_matches(position_symbol: str, wanted: str) -> bool:
    a = str(position_symbol or "").upper()
    b = str(wanted or "").upper()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a) or b in a or a in b


def _comment_timeframe(comment: str | None) -> str | None:
    """Parse timeframe from ATIS order comment ``ATIS|H1`` / ``ATIS|H1|tag``."""
    parts = str(comment or "").strip().split("|")
    if len(parts) < 2:
        return None
    if str(parts[0]).strip().upper() != "ATIS":
        return None
    tf = str(parts[1] or "").strip().upper()
    return tf or None


def filter_atis_positions(
    positions: list[Any] | None,
    symbol: str | None = None,
    *,
    magic: int = ATIS_MAGIC,
    atis_only: bool = True,
    timeframe: str | None = None,
) -> list[Any]:
    """Count only ATIS-managed positions (optional symbol / timeframe filter)."""
    want_tf = str(timeframe or "").strip().upper() or None
    out: list[Any] = []
    for pos in positions or []:
        try:
            pos_magic = int(getattr(pos, "magic", 0) or 0)
            pos_symbol = str(getattr(pos, "symbol", "") or "")
            pos_comment = str(getattr(pos, "comment", "") or "")
        except Exception:
            continue
        if atis_only and pos_magic != int(magic):
            continue
        if symbol and not _symbol_matches(pos_symbol, symbol):
            continue
        if want_tf:
            pos_tf = _comment_timeframe(pos_comment)
            if pos_tf != want_tf:
                continue
        out.append(pos)
    return out


def _pip_size_from_info(info: Any, fallback: float = 0.01) -> float:
    try:
        point = float(getattr(info, "point", 0) or 0)
        digits = int(getattr(info, "digits", 2) or 2)
    except Exception:
        return float(fallback)
    if point <= 0:
        return float(fallback)
    # Gold/FX: treat 1 pip as 10 points when digits are 3/5-style quotes.
    if digits in (3, 5):
        return point * 10.0
    return point


def read_live_spread(
    client: MT5Client,
    symbol: str,
    *,
    pip_size_fallback: float = 0.01,
) -> dict[str, float]:
    """Read broker bid/ask and convert spread to pips."""
    mt5 = _mt5_module()
    resolved = client.resolve_symbol(symbol)
    tick = mt5.symbol_info_tick(resolved)
    info = mt5.symbol_info(resolved)
    if tick is None or info is None:
        raise RuntimeError(f"No tick/info for {resolved}")
    bid = float(tick.bid)
    ask = float(tick.ask)
    pip = _pip_size_from_info(info, fallback=pip_size_fallback)
    raw = max(0.0, ask - bid)
    spread_pips = raw / max(pip, 1e-12)
    return {
        "bid": bid,
        "ask": ask,
        "spread": raw,
        "spread_pips": float(spread_pips),
        "pip_size": float(pip),
    }


def entries_allowed_for_spread(spread_pips: float, cfg: dict[str, Any] | None = None) -> tuple[int, str]:
    """
    How many new entries this cycle given live spread.
    - If use_live_spread_filter is false → ignore spread (model-only): 1 entry
    - Block when spread is wide
    - Scale up toward max_entries_per_cycle when spread is very tight
    """
    c = cfg if cfg is not None else _cfg()
    if not bool(c.get("use_live_spread_filter", True)):
        return 1, "spread_filter_off"
    max_spread = float(c.get("max_entry_spread_pips", 25.0))
    tight = float(c.get("tight_spread_pips", 12.0))
    max_entries = max(1, int(c.get("max_entries_per_cycle", 1)))
    scale = bool(c.get("scale_in_when_tight_spread", True))
    if spread_pips > max_spread:
        return 0, "spread_too_wide"
    if not scale or max_entries <= 1:
        return 1, "ok_single"
    if spread_pips <= tight:
        return max_entries, "ok_tight_full"
    # Linear scale between 1 and max_entries as spread moves from max→tight.
    span = max(max_spread - tight, 1e-9)
    frac = (max_spread - spread_pips) / span
    n = int(round(1 + frac * (max_entries - 1)))
    return max(1, min(max_entries, n)), "ok_scaled"


def _unit_cost_live(
    price: float,
    live: dict[str, Any],
    *,
    live_spread_pips: float | None = None,
) -> float:
    # Prefer live broker spread when available so tight markets unlock more opportunities.
    if live_spread_pips is not None and np.isfinite(live_spread_pips):
        spread = float(live_spread_pips)
    else:
        spread = float(live.get("spread_pips", 30.0))
    floor = float(live.get("spread_pips_floor", 0.0) or 0.0)
    spread = max(spread, floor)
    slip = float(live.get("slippage_pips", 5.0))
    comm = float(live.get("commission_per_lot", 7.0))
    pip = float(live.get("pip_size", 0.01))
    return ((spread + slip) * pip + comm / 100.0) / max(price, 1e-12)


def _predict_latest(
    bundle: dict[str, Any],
    features_df: pd.DataFrame,
    *,
    live_spread_pips: float | None = None,
) -> tuple[int, float, dict[str, Any]]:
    """
    Return (side, confidence, debug).
    side: -1 sell, 0 hold, +1 buy.

    Prefer Engine-4 trade_policy + calibrator + regime/trend filters when present
    so live decisions match Training/Validation/Testing evaluation.
    """
    cfg = _cfg()
    cols = bundle["feature_cols"]
    row = features_df.iloc[[-1]].copy()
    missing = [c for c in cols if c not in row.columns]
    if missing:
        # Soft-fill rare gaps so one absent HTF column does not abort the cycle.
        for c in missing:
            row[c] = 0.0
        debug_missing = missing
    else:
        debug_missing = []
    X = row[cols].values
    if np.isnan(X).any():
        return 0, 0.0, {"reason": "nan_features", "missing_filled": debug_missing}
    X_s = bundle["scaler"].transform(X)
    model = bundle["model"]
    raw_pred = int(model.predict(X_s)[0])
    debug: dict[str, Any] = {
        "raw_pred": raw_pred,
        "pipeline": bundle.get("pipeline_version"),
        "missing_filled": debug_missing,
    }
    if live_spread_pips is not None:
        debug["live_spread_pips"] = float(live_spread_pips)

    if not hasattr(model, "predict_proba"):
        return raw_pred, 0.55, debug

    proba = model.predict_proba(X_s)
    classes = [int(c) for c in model.classes_]
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        try:
            proba = calibrator(proba)
            debug["calibrated"] = True
        except Exception:
            debug["calibrated"] = False
    proba_1d = proba[0]
    pmap = {classes[i]: float(proba_1d[i]) for i in range(len(classes))}
    p_down = pmap.get(-1, 0.0)
    p_hold = pmap.get(0, 0.0)
    p_up = pmap.get(1, 0.0)
    debug.update({"p_down": p_down, "p_hold": p_hold, "p_up": p_up})

    policy = bundle.get("trade_policy") or {}
    live = bundle.get("live_policy") or {}
    use_e4_policy = bool(policy) and bool(cfg.get("use_training_trade_policy", True))

    if use_e4_policy:
        from atis.engines.engine4_training import (
            apply_trend_align,
            policy_from_proba,
            structure_primary_sides,
            trend_masks_from_frame,
        )

        atr_pct = None
        unit_costs = None
        regime_mask = None
        last = features_df.iloc[-1]
        price = float(last["close"]) if "close" in features_df.columns else 0.0
        if "atr" in features_df.columns and price > 0:
            atr_v = float(last["atr"])
            atr_pct = np.array([atr_v / max(price, 1e-12)], dtype=float)
            unit_costs = np.array(
                [_unit_cost_live(price, live, live_spread_pips=live_spread_pips)],
                dtype=float,
            )
            if bool(live.get("regime_filter", True)):
                lo = float(live.get("regime_atr_low", 0.0) or 0.0)
                hi_raw = live.get("regime_atr_high")
                hi = float(hi_raw) if hi_raw is not None else float("inf")
                regime_mask = np.array([(atr_pct[0] >= lo) and (atr_pct[0] <= hi)], dtype=bool)

        use_meta = bool(live.get("meta_labeling", False)) and bool(bundle.get("trend_align", True))
        allow_long, allow_short = trend_masks_from_frame(features_df.tail(80))
        primary = structure_primary_sides(allow_long[-1:], allow_short[-1:]) if use_meta else None
        preds = policy_from_proba(
            proba,
            classes,
            decision_threshold=float(policy.get("decision_threshold", cfg.get("confidence_threshold", 0.55))),
            directional_edge=float(policy.get("directional_edge", cfg.get("directional_edge", 0.12))),
            confidence_quantile=0.0,
            confidence_floor=float(policy.get("confidence_floor", policy.get("decision_threshold", 0.55))),
            atr_pct=atr_pct,
            unit_costs=unit_costs,
            cost_edge_multiple=float(live.get("cost_edge_multiple", 0.0) or 0.0),
            regime_mask=regime_mask,
            short_edge_multiple=float(live.get("short_edge_multiple", 1.0) or 1.0),
            primary_sides=primary,
        )
        if not use_meta and bool(bundle.get("trend_align", True)):
            preds = apply_trend_align(preds, allow_long[-1:], allow_short[-1:])
        side = int(preds[0])
        conf = float(max(p_up, p_down, p_hold))
        if side != 0:
            conf = float(p_up if side > 0 else p_down)
        debug["reason"] = "e4_trade_policy" if side != 0 else "e4_hold"
        debug["meta_labeling"] = use_meta
        debug["regime_ok"] = None if regime_mask is None else bool(regime_mask[0])
        return side, conf, debug

    # Legacy relative directional fallback (older artifacts without trade_policy).
    use_directional = bool(cfg.get("directional_decision", True))
    min_side = float(cfg.get("min_side_probability", 0.15))
    rel_thr = float(cfg.get("directional_rel_threshold", 0.52))

    if use_directional:
        denom = p_up + p_down
        if denom >= 1e-9:
            rel_up = p_up / denom
            rel_down = p_down / denom
            debug["rel_up"] = rel_up
            debug["rel_down"] = rel_down
            if rel_up >= rel_thr and p_up >= min_side:
                return 1, float(rel_up), {**debug, "reason": "rel_buy"}
            if rel_down >= rel_thr and p_down >= min_side:
                return -1, float(rel_down), {**debug, "reason": "rel_sell"}
        if raw_pred in (-1, 1) and pmap.get(raw_pred, 0) >= min_side:
            return raw_pred, float(pmap[raw_pred]), {**debug, "reason": "raw_side"}
        return 0, float(max(p_up, p_down, p_hold)), {**debug, "reason": "no_directional_edge"}

    conf = float(np.max(proba_1d))
    return raw_pred, conf, {**debug, "reason": "argmax"}


def _predict_bundle(
    bundle: dict[str, Any],
    features_df: pd.DataFrame | None = None,
    frames: dict[str, pd.DataFrame] | None = None,
    *,
    live_spread_pips: float | None = None,
) -> tuple[int, float, dict[str, Any]]:
    if bundle.get("artifact_type") == "llmodel":
        if not frames:
            raise ValueError("llmodel_requires_multitimeframe_frames")
        result = predict_with_llmodel(bundle, frames)
        debug = {
            "reason": "llmodel",
            "scenario_probabilities": result["scenario_probabilities"],
            "attention_by_timeframe": result["attention_by_timeframe"],
            "expected_return": result["expected_return"],
            "risk_score": result["risk_score"],
            "feature_rankings": result["feature_rankings"],
            "artifact_path": result["artifact_path"],
        }
        return int(result["pred"]), float(result["confidence"]), debug
    if features_df is None:
        raise ValueError("baseline_model_requires_features_df")
    return _predict_latest(bundle, features_df, live_spread_pips=live_spread_pips)


def _sl_tp_from_atr(
    price: float,
    side: str,
    atr_value: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float, float]:
    """Legacy fixed ATR×multiplier exits (used only when dynamic_exits.enabled is false)."""
    if side == "buy":
        sl = price - sl_mult * atr_value
        tp = price + tp_mult * atr_value
    else:
        sl = price + sl_mult * atr_value
        tp = price - tp_mult * atr_value
    return float(sl), float(tp)


def resolve_sl_tp(
    *,
    price: float,
    side: str,
    atr_value: float,
    confidence: float,
    featured: pd.DataFrame,
    prediction_debug: dict[str, Any] | None = None,
    sl_mult: float = 1.5,
    tp_mult: float = 2.5,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Prefer model-driven dynamic exits; fall back to fixed ATR multipliers."""
    cfg = cfg if cfg is not None else _cfg()
    dyn_cfg = dict(cfg.get("dynamic_exits") or {})
    dbg = prediction_debug or {}
    if bool(dyn_cfg.get("enabled", True)):
        exits = compute_dynamic_sl_tp(
            price=price,
            side=side,
            atr_value=atr_value,
            confidence=confidence,
            featured=featured,
            expected_return=_finite_or_none(dbg.get("expected_return")),
            risk_score=_finite_or_none(dbg.get("risk_score")),
            cfg=dyn_cfg,
        )
        return exits.sl, exits.tp, exits.as_dict()
    sl, tp = _sl_tp_from_atr(price, side, atr_value, sl_mult, tp_mult)
    return sl, tp, {
        "method": "fixed_atr_multipliers",
        "sl": sl,
        "tp": tp,
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def build_order_comment(
    timeframe: str,
    *,
    prefix: str = "ATIS",
    pattern_tag: str | None = None,
) -> str:
    """MT5 order comment — timeframe + optional pattern tag (max 31 chars)."""
    tf = str(timeframe or "").strip().upper() or "?"
    tag = (pattern_tag or "").strip().replace("pat_", "").replace("cmp_", "")[:10]
    raw = f"{prefix}|{tf}|{tag}" if tag else f"{prefix}|{tf}"
    return raw[:31]


def _sync_pattern_overlay(
    client: MT5Client,
    featured: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    """Detect patterns and publish async MT5 chart drawings (never raises)."""
    cfg = _cfg()
    overlay_cfg = cfg.get("pattern_overlay") or {}
    if not bool(overlay_cfg.get("enabled", True)):
        return {"enabled": False}
    try:
        from atis.shared.mt5_pattern_overlay import get_overlay_service

        service = get_overlay_service(cfg)
        broker_symbol = client.resolve_symbol(symbol)
        service.set_terminal_info_provider(lambda: client.terminal_info())
        overlays = service.sync_from_features(
            featured,
            symbol=symbol,
            timeframe=timeframe,
            broker_symbol=broker_symbol,
        )
        return {
            "enabled": True,
            "count": len(overlays),
            "keys": [o.key for o in overlays[:12]],
            "ids": [o.id for o in overlays[:12]],
        }
    except Exception as exc:
        logger.warning("pattern_overlay_sync_failed", symbol=symbol, error=str(exc))
        return {"enabled": True, "error": str(exc)}


def _link_trade_to_patterns(
    *,
    side: str,
    ticket: int | None,
    reason: str,
) -> dict[str, Any]:
    cfg = _cfg()
    if not bool((cfg.get("pattern_overlay") or {}).get("enabled", True)):
        return {}
    try:
        from atis.shared.mt5_pattern_overlay import get_overlay_service

        service = get_overlay_service(cfg)
        linked = service.link_trade(side=side, ticket=ticket, reason=reason)
        explain = service.explain_decision(side)
        return {
            "pattern_keys": [p.key for p in linked],
            "pattern_ids": [p.id for p in linked],
            "pattern_summary": explain.get("summary", ""),
            "explain": explain,
        }
    except Exception as exc:
        logger.warning("pattern_trade_link_failed", error=str(exc))
        return {"error": str(exc)}


def send_market_order(
    client: MT5Client,
    symbol: str,
    side: str,
    volume: float,
    sl: float,
    tp: float,
    comment: str = "ATIS",
) -> dict[str, Any]:
    """Send MT5 market order — SL/TP mandatory."""
    if sl <= 0 or tp <= 0:
        raise ValueError("Stop-Loss and Take-Profit are mandatory")
    mt5 = _mt5_module()
    resolved = client.resolve_symbol(symbol)
    tick = mt5.symbol_info_tick(resolved)
    info = mt5.symbol_info(resolved)
    if tick is None or info is None:
        raise RuntimeError(f"No tick/info for {resolved}")

    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == "buy" else tick.bid
    filling = mt5.ORDER_FILLING_IOC
    if info.filling_mode & 1:
        filling = mt5.ORDER_FILLING_FOK
    elif info.filling_mode & 2:
        filling = mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": resolved,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": 260729,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    return result._asdict()


def _position_filling_mode(info: Any, mt5: Any) -> Any:
    filling = mt5.ORDER_FILLING_IOC
    try:
        mode = int(getattr(info, "filling_mode", 0) or 0)
    except Exception:
        mode = 0
    if mode & 1:
        filling = mt5.ORDER_FILLING_FOK
    elif mode & 2:
        filling = mt5.ORDER_FILLING_IOC
    return filling


def serialize_position(pos: Any, mt5: Any | None = None) -> dict[str, Any]:
    """Normalize an MT5 position object into a JSON-safe dict."""
    mt5 = mt5 or _mt5_module()
    try:
        pos_type = int(getattr(pos, "type", -1))
    except Exception:
        pos_type = -1
    side = "buy" if pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0) else "sell"
    profit = float(getattr(pos, "profit", 0) or 0)
    swap = float(getattr(pos, "swap", 0) or 0)
    return {
        "ticket": int(getattr(pos, "ticket", 0) or 0),
        "symbol": str(getattr(pos, "symbol", "") or ""),
        "side": side,
        "type": pos_type,
        "volume": float(getattr(pos, "volume", 0) or 0),
        "price_open": float(getattr(pos, "price_open", 0) or 0),
        "price_current": float(getattr(pos, "price_current", 0) or 0),
        "sl": float(getattr(pos, "sl", 0) or 0),
        "tp": float(getattr(pos, "tp", 0) or 0),
        "profit": profit,
        "swap": swap,
        "net_profit": profit + swap,
        "magic": int(getattr(pos, "magic", 0) or 0),
        "comment": str(getattr(pos, "comment", "") or ""),
        "time": int(getattr(pos, "time", 0) or 0),
    }


def list_open_positions(
    client: MT5Client,
    symbol: str | None = None,
    *,
    atis_only: bool = True,
    magic: int = ATIS_MAGIC,
) -> list[dict[str, Any]]:
    """List currently open positions managed by ATIS (optionally all)."""
    mt5 = _mt5_module()
    raw = list(mt5.positions_get() or [])
    tracked = filter_atis_positions(raw, symbol, magic=magic, atis_only=atis_only)
    return [serialize_position(pos, mt5) for pos in tracked]


def close_position(
    client: MT5Client,
    ticket: int,
    *,
    comment: str = "ATIS close",
) -> dict[str, Any]:
    """Close a single open position by ticket via opposite market deal."""
    mt5 = _mt5_module()
    positions = list(mt5.positions_get(ticket=int(ticket)) or [])
    if not positions:
        raise ValueError(f"position_not_found:{ticket}")
    pos = positions[0]
    symbol = str(getattr(pos, "symbol", "") or "")
    volume = float(getattr(pos, "volume", 0) or 0)
    if volume <= 0:
        raise ValueError(f"invalid_volume:{ticket}")

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        # Try resolve via client cache for broker suffixes
        try:
            resolved = client.resolve_symbol(symbol)
            info = mt5.symbol_info(resolved)
            tick = mt5.symbol_info_tick(resolved)
            symbol = resolved
        except Exception:
            pass
    if info is None or tick is None:
        raise RuntimeError(f"No tick/info for {symbol}")

    pos_type = int(getattr(pos, "type", -1))
    is_buy = pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0)
    order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
    price = float(tick.bid if is_buy else tick.ask)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "position": int(ticket),
        "price": price,
        "deviation": 20,
        "magic": int(getattr(pos, "magic", ATIS_MAGIC) or ATIS_MAGIC),
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _position_filling_mode(info, mt5),
    }
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    payload = result._asdict()
    retcode = int(payload.get("retcode", -1))
    ok = retcode in (
        getattr(mt5, "TRADE_RETCODE_DONE", 10009),
        getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
    )
    return {
        "ok": ok,
        "ticket": int(ticket),
        "symbol": symbol,
        "side": "buy" if is_buy else "sell",
        "volume": volume,
        "profit": float(getattr(pos, "profit", 0) or 0),
        "retcode": retcode,
        "comment": str(payload.get("comment", "") or ""),
        "mt5": payload,
    }


def close_positions_filtered(
    client: MT5Client,
    *,
    mode: str = "all",
    symbol: str | None = None,
    atis_only: bool = True,
    magic: int = ATIS_MAGIC,
) -> dict[str, Any]:
    """Close open positions: all | winners | losers."""
    mode_key = str(mode or "all").strip().lower()
    if mode_key not in {"all", "winners", "losers"}:
        raise ValueError(f"invalid_close_mode:{mode}")

    positions = list_open_positions(client, symbol, atis_only=atis_only, magic=magic)
    selected: list[dict[str, Any]] = []
    for pos in positions:
        net = float(pos.get("net_profit", pos.get("profit", 0)) or 0)
        if mode_key == "winners" and net <= 0:
            continue
        if mode_key == "losers" and net >= 0:
            continue
        selected.append(pos)

    closed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for pos in selected:
        ticket = int(pos["ticket"])
        try:
            result = close_position(client, ticket, comment=f"ATIS {mode_key}")
            if result.get("ok"):
                closed.append(result)
            else:
                failed.append(result)
        except Exception as exc:
            failed.append({"ok": False, "ticket": ticket, "error": str(exc)})

    return {
        "mode": mode_key,
        "matched": len(selected),
        "closed": closed,
        "failed": failed,
        "closed_count": len(closed),
        "failed_count": len(failed),
        "ok": len(failed) == 0,
    }


def fetch_recent_bars(client: MT5Client, symbol: str, timeframe: str, bars: int = 300) -> pd.DataFrame:
    mt5 = _mt5_module()
    from atis.shared.mt5_client import timeframe_to_mt5

    resolved = client.resolve_symbol(symbol)
    rates = mt5.copy_rates_from_pos(resolved, timeframe_to_mt5(timeframe), 0, bars)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    return df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
            "symbol",
            "timeframe",
        ]
    ]


def _compute_base_features(client: MT5Client, symbol: str, timeframe: str, bars: int = 400) -> pd.DataFrame:
    raw = fetch_recent_bars(client, symbol, timeframe, bars=bars)
    if raw.empty:
        raise ValueError(f"no_bars:{timeframe}")
    cleaned, _ = clean_dataframe(raw, timeframe)
    return compute_features(cleaned, load_indicators_config())


def prepare_live_features(
    client: MT5Client,
    symbol: str,
    timeframe: str,
    *,
    bars: int = 400,
    htf_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match training feature pipeline: indicators → learning feats → HTF enrich."""
    from atis.engines.engine4_training import engineer_learning_features
    from atis.engines.engine4_training.data_sources import (
        enrich_with_higher_timeframes,
        higher_timeframes_for,
    )

    featured = _compute_base_features(client, symbol, timeframe, bars=bars)
    e4 = load_engine_config().get("engine4_training", {}) or {}
    if bool(e4.get("engineer_learning_features", True)):
        featured = engineer_learning_features(featured)

    enrich_meta: dict[str, Any] = {"enabled": False}
    if bool(e4.get("cross_tf_features", True)):
        cache = htf_cache if htf_cache is not None else {}
        htf_frames: dict[str, pd.DataFrame] = {}
        for htf in higher_timeframes_for(timeframe):
            if htf in cache:
                htf_frames[htf] = cache[htf]
                continue
            try:
                htf_feat = _compute_base_features(client, symbol, htf, bars=max(bars, 260))
                if bool(e4.get("engineer_learning_features", True)):
                    htf_feat = engineer_learning_features(htf_feat)
                cache[htf] = htf_feat
                htf_frames[htf] = htf_feat
            except Exception as exc:
                logger.warning("live_htf_fetch_failed", symbol=symbol, htf=htf, error=str(exc))
        featured, enrich_meta = enrich_with_higher_timeframes(
            featured,
            symbol,
            timeframe,
            htf_frames=htf_frames,
        )
    return featured, {"cross_tf": enrich_meta, "rows": int(len(featured))}


def analyze_timeframe_signal(
    client: MT5Client,
    symbol: str,
    timeframe: str,
    *,
    allow_ungated: bool = True,
    match_timeframe_only: bool = True,
    htf_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Fetch bars → clean → features (+HTF/learning) → trained-model prediction."""
    timeframe = str(timeframe).upper()
    bundle = load_champion_or_latest(
        symbol,
        timeframe,
        allow_ungated=allow_ungated,
        match_timeframe_only=match_timeframe_only,
    )
    live_spread_pips: float | None = None
    use_spread = bool(_cfg().get("use_live_spread_filter", True))
    if use_spread:
        try:
            live_spread_pips = float(read_live_spread(client, symbol)["spread_pips"])
        except Exception as exc:
            logger.warning("live_spread_read_failed", symbol=symbol, error=str(exc))

    if bundle.get("artifact_type") == "llmodel":
        frames = _fetch_feature_frames_for_bundle(client, symbol, bundle, timeframe, bars=360)
        featured = frames[timeframe] if timeframe in frames else frames[_timeframes_for_llmodel(bundle, timeframe)[0]]
        pred, conf, dbg = _predict_bundle(bundle, None, frames, live_spread_pips=live_spread_pips)
    else:
        featured, feat_meta = prepare_live_features(
            client, symbol, timeframe, bars=400, htf_cache=htf_cache
        )
        pred, conf, dbg = _predict_bundle(
            bundle, featured, None, live_spread_pips=live_spread_pips
        )
        dbg = {**(dbg or {}), "feature_pipeline": feat_meta}
    dbg = {
        **(dbg or {}),
        "use_live_spread_filter": use_spread,
        **({"live_spread_pips": live_spread_pips} if live_spread_pips is not None else {}),
    }
    model_tf = str(bundle.get("timeframe") or timeframe).upper()
    return {
        "tf": timeframe,
        "pred": int(pred),
        "conf": float(conf),
        "debug": dbg,
        "featured": featured,
        "bundle": bundle,
        "model_type": bundle.get("artifact_type", "baseline"),
        "model_version": bundle.get("version"),
        "model_timeframe": model_tf,
        "close": float(featured["close"].iloc[-1]),
        "atr": float(featured["atr"].iloc[-1]) if "atr" in featured.columns else float("nan"),
        "live_spread_pips": live_spread_pips,
    }


def _append_decision(decisions_log: Path, decision: dict[str, Any]) -> None:
    with decisions_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(decision, default=str) + "\n")


def _place_from_signal(
    *,
    report: LiveLoopReport,
    risk: RiskManager,
    client: MT5Client,
    symbol: str,
    timeframe: str,
    pred: int,
    conf: float,
    featured: pd.DataFrame,
    equity: float,
    positions: list[Any],
    dry_run: bool,
    mode: str,
    conf_thr: float,
    sl_mult: float,
    tp_mult: float,
    trades_log: Path,
    reason_suffix: str = "",
    prediction_debug: dict[str, Any] | None = None,
) -> None:
    # Gate 1 (mandatory): trained model / fused TF signal — never enter on spread alone.
    if pred == 0 or conf < conf_thr:
        report.errors.append(
            f"{symbol}:skip pred={pred} conf={conf:.3f} thr={conf_thr}{reason_suffix}"
        )
        return

    cfg = _cfg()
    atis_only = bool(cfg.get("count_atis_positions_only", True))
    use_spread = bool(cfg.get("use_live_spread_filter", True))

    # Gate 2 (optional): live bid-ask filter — only when enabled in Settings.
    spread_pips = float("nan")
    if use_spread:
        try:
            spread_info = read_live_spread(client, symbol)
            spread_pips = float(spread_info["spread_pips"])
        except Exception as exc:
            report.errors.append(f"{symbol}:spread_unavailable:{exc}")
            logger.warning("spread_gate_failed", symbol=symbol, error=str(exc))
            return
        n_entries, spread_reason = entries_allowed_for_spread(spread_pips, cfg)
        if n_entries <= 0:
            report.blocked_by_risk += 1
            logger.info(
                "trade_blocked",
                symbol=symbol,
                reason=spread_reason,
                spread_pips=spread_pips,
                max_entry_spread_pips=cfg.get("max_entry_spread_pips"),
            )
            report.errors.append(
                f"{symbol}:skip {spread_reason} spread_pips={spread_pips:.2f}{reason_suffix}"
            )
            return
    else:
        n_entries, spread_reason = 1, "spread_filter_off"

    report.signals += 1
    working = list(positions)
    placed = 0
    # Independent multi-TF: cap by this TF's own open book, and still respect global max.
    scope_by_tf = bool(cfg.get("multi_tf_independent", True)) and not bool(
        cfg.get("multi_tf_fusion", False)
    )
    max_per_tf = int(cfg.get("max_open_positions_per_tf", 0) or 0)
    for i in range(n_entries):
        tracked_all = filter_atis_positions(working, symbol, atis_only=atis_only)
        tracked_tf = (
            filter_atis_positions(working, symbol, atis_only=atis_only, timeframe=timeframe)
            if scope_by_tf
            else tracked_all
        )
        if max_per_tf > 0 and len(tracked_tf) >= max_per_tf:
            if placed == 0:
                report.blocked_by_risk += 1
            logger.info(
                "trade_blocked",
                symbol=symbol,
                reason="max_open_positions_per_tf",
                timeframe=timeframe,
                open_tf=len(tracked_tf),
                max_per_tf=max_per_tf,
            )
            break
        open_risk = float(len(tracked_all)) * float(risk.risk_per_trade)
        ok, reason = risk.allow_new_trade(
            equity,
            len(tracked_all),
            planned_risk_pct=risk.risk_per_trade,
            open_risk_pct=open_risk,
        )
        if not ok:
            if placed == 0:
                report.blocked_by_risk += 1
            logger.info(
                "trade_blocked",
                symbol=symbol,
                reason=reason,
                open_atis=len(tracked_all),
                open_tf=len(tracked_tf),
                timeframe=timeframe,
                attempted=i + 1,
                wanted=n_entries,
                spread_pips=spread_pips,
            )
            break

        side = "buy" if pred > 0 else "sell"
        atr_value = float(featured["atr"].iloc[-1])
        price = float(featured["close"].iloc[-1])
        if not np.isfinite(atr_value) or atr_value <= 0:
            report.errors.append(f"{symbol}:bad_atr")
            break
        try:
            sl, tp, exit_meta = resolve_sl_tp(
                price=price,
                side=side,
                atr_value=atr_value,
                confidence=float(conf),
                featured=featured,
                prediction_debug=prediction_debug,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                cfg=cfg,
            )
        except ValueError as exc:
            report.errors.append(f"{symbol}:exit_levels:{exc}")
            logger.info(
                "trade_blocked",
                symbol=symbol,
                reason=f"exit_levels:{exc}",
                pred=pred,
                conf=conf,
            )
            break
        volume = position_size_lots(equity, risk.risk_per_trade, price, sl)
        if bool(cfg.get("confidence_sizing_enabled", True)):
            try:
                from atis.engines.engine4_training.promotion_v16 import confidence_position_size

                atr_pct = float(atr_value) / max(float(price), 1e-9)
                size_m = confidence_position_size(
                    float(conf),
                    atr_pct=atr_pct,
                    base_size=float(cfg.get("confidence_sizing_base", 1.0)),
                    max_size=float(cfg.get("confidence_sizing_max", 1.5)),
                    min_size=float(cfg.get("confidence_sizing_min", 0.25)),
                )
                volume = float(max(0.01, round(volume * size_m, 2)))
            except Exception as exc:
                logger.warning("confidence_sizing_failed", error=str(exc))
        spread_txt = f"{spread_pips:.2f}" if np.isfinite(spread_pips) else "n/a"
        entry_tag = f";entry={i + 1}/{n_entries};spread_pips={spread_txt};{spread_reason}"
        exit_tag = (
            f";exit={exit_meta.get('method')};rr={float(exit_meta.get('reward_risk') or 0):.2f}"
        )

        # Explainable AI: which patterns support this side (before order).
        pattern_keys: list[str] = []
        pattern_ids: list[str] = []
        pattern_summary = ""
        try:
            from atis.shared.mt5_pattern_overlay import get_overlay_service

            if bool((cfg.get("pattern_overlay") or {}).get("enabled", True)):
                explain = get_overlay_service(cfg).explain_decision(side)
                pattern_keys = [p["key"] for p in explain.get("patterns") or []]
                pattern_ids = [p["id"] for p in explain.get("patterns") or []]
                pattern_summary = str(explain.get("summary") or "")
        except Exception as exc:
            logger.warning("pattern_explain_failed", error=str(exc))
        pattern_reason = f";patterns={','.join(pattern_keys[:4])}" if pattern_keys else ""

        if dry_run:
            link_info = _link_trade_to_patterns(
                side=side,
                ticket=None,
                reason=f"paper;pred={pred};tf={timeframe};conf={conf:.3f};{pattern_summary}",
            )
            if link_info.get("pattern_keys"):
                pattern_keys = list(link_info["pattern_keys"])
                pattern_ids = list(link_info.get("pattern_ids") or pattern_ids)
                pattern_summary = str(link_info.get("pattern_summary") or pattern_summary)
            rec = TradeRecord(
                ticket=None,
                symbol=symbol,
                side=side,
                volume=volume,
                entry_price=price,
                sl=sl,
                tp=tp,
                confidence=conf,
                reason=(
                    f"pred={pred};paper;tf={timeframe}{entry_tag}{exit_tag}"
                    f"{reason_suffix}{pattern_reason}"
                ),
                ts=_utc_now().isoformat(),
                mode="paper",
                pattern_keys=pattern_keys,
                pattern_ids=pattern_ids,
                pattern_summary=pattern_summary,
            )
            report.trades.append({**asdict(rec), "exit_meta": exit_meta})
            report.orders_sent += 1
            placed += 1
            with trades_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**asdict(rec), "exit_meta": exit_meta}, default=str) + "\n")
            logger.info("paper_trade", **asdict(rec), exit_meta=exit_meta)
            # Synthetic slot so subsequent scale-ins respect max_open in paper mode.
            working.append(
                type(
                    "P",
                    (),
                    {
                        "magic": ATIS_MAGIC,
                        "symbol": symbol,
                        "comment": build_order_comment(timeframe),
                    },
                )()
            )
            if bool(cfg.get("shadow_tracking_enabled", True)):
                try:
                    from atis.engines.engine4_training.shadow_challenger import record_shadow_decision

                    # Paper EV proxy until closed-trade PnL is available.
                    pnl_proxy = float(np.sign(pred)) * max(float(conf) - 0.5, 0.0) * 0.001
                    record_shadow_decision(
                        get_path("models"),
                        symbol=symbol,
                        timeframe=timeframe,
                        pnl=pnl_proxy,
                    )
                except Exception as exc:
                    logger.warning("shadow_record_failed", error=str(exc))
        else:
            pat_tag = pattern_keys[0] if pattern_keys else None
            order_comment = build_order_comment(timeframe, pattern_tag=pat_tag)
            result = send_market_order(
                client, symbol, side, volume, sl, tp, comment=order_comment
            )
            ticket = result.get("order") or result.get("deal")
            link_info = _link_trade_to_patterns(
                side=side,
                ticket=int(ticket) if ticket is not None else None,
                reason=(
                    f"demo;pred={pred};tf={timeframe};retcode={result.get('retcode')};"
                    f"{pattern_summary}"
                ),
            )
            if link_info.get("pattern_keys"):
                pattern_keys = list(link_info["pattern_keys"])
                pattern_ids = list(link_info.get("pattern_ids") or pattern_ids)
                pattern_summary = str(link_info.get("pattern_summary") or pattern_summary)
            rec = TradeRecord(
                ticket=ticket,
                symbol=symbol,
                side=side,
                volume=volume,
                entry_price=price,
                sl=sl,
                tp=tp,
                confidence=conf,
                reason=(
                    f"pred={pred};retcode={result.get('retcode')};tf={timeframe}"
                    f"{entry_tag}{exit_tag}{reason_suffix}{pattern_reason}"
                ),
                ts=_utc_now().isoformat(),
                mode="demo" if str(mode) != "live" else "live",
                pattern_keys=pattern_keys,
                pattern_ids=pattern_ids,
                pattern_summary=pattern_summary,
            )
            report.trades.append({**asdict(rec), "exit_meta": exit_meta})
            report.orders_sent += 1
            placed += 1
            with trades_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"trade": asdict(rec), "mt5": result, "exit_meta": exit_meta},
                        default=str,
                    )
                    + "\n"
                )
            if bool(cfg.get("shadow_tracking_enabled", True)):
                try:
                    from atis.engines.engine4_training.shadow_challenger import record_shadow_decision

                    pnl_proxy = float(np.sign(pred)) * max(float(conf) - 0.5, 0.0) * 0.001
                    record_shadow_decision(
                        get_path("models"),
                        symbol=symbol,
                        timeframe=timeframe,
                        pnl=pnl_proxy,
                    )
                except Exception as exc:
                    logger.warning("shadow_record_failed", error=str(exc))
            logger.info(
                "demo_order",
                ticket=rec.ticket,
                retcode=result.get("retcode"),
                entry=i + 1,
                of=n_entries,
                spread_pips=spread_pips,
                patterns=pattern_keys,
                exit_meta=exit_meta,
            )
            # Refresh broker positions after each fill so caps stay accurate.
            try:
                mt5 = _mt5_module()
                working = list(mt5.positions_get() or [])
            except Exception:
                working.append(
                    type(
                        "P",
                        (),
                        {
                            "magic": ATIS_MAGIC,
                            "symbol": symbol,
                            "comment": build_order_comment(timeframe),
                        },
                    )()
                )

        DataStateRegistry().audit(
            "engine5",
            "signal",
            symbol=symbol,
            timeframe=timeframe,
            detail_json=json.dumps(asdict(rec)),
        )

    if placed:
        positions[:] = working
        logger.info(
            "scale_in_complete",
            symbol=symbol,
            placed=placed,
            wanted=n_entries,
            spread_pips=spread_pips,
            spread_reason=spread_reason,
        )


def run_live_once(
    symbols: list[str],
    timeframe: str = "M5",
    *,
    dry_run: bool = True,
    allow_ungated: bool = True,
) -> LiveLoopReport:
    """Single inference+optional execution cycle (paper/demo) — gold only by default."""
    ensure_project_dirs()
    set_global_seed()
    cfg = _cfg()
    mode = str(cfg.get("mode", "paper"))
    if mode == "live":
        raise RuntimeError("Live real-money mode requires explicit user approval — refused.")

    full = load_engine_config()
    allowed = set(full.get("trading", {}).get("allowed_live_symbols", ["XAUUSD"]))
    if not symbols:
        symbols = list(cfg.get("symbols") or [full.get("trading", {}).get("primary_symbol", "XAUUSD")])
    bad = [s for s in symbols if s not in allowed]
    if bad:
        raise RuntimeError(f"Live trading restricted to {sorted(allowed)}; refused: {bad}")
    timeframe = timeframe or str(cfg.get("timeframe") or full.get("trading", {}).get("primary_timeframe", "M5"))
    exec_mode = "paper" if dry_run else "demo"

    report = LiveLoopReport(started_at=_utc_now().isoformat())
    # Drift / PSI advisory → optional retrain request for scheduler/UI.
    if bool(cfg.get("auto_retrain_request_enabled", True)):
        try:
            from atis.engines.engine4_training.shadow_challenger import (
                read_retrain_advisory,
                write_retrain_request,
            )

            adv = read_retrain_advisory(get_path("models"))
            if adv.get("auto_retrain_recommended") or adv.get("would_trigger_now"):
                req = write_retrain_request(
                    get_path("models"),
                    reason=str(adv.get("schedule_reason") or "drift_or_interval"),
                    source="engine5_live",
                    symbol=(symbols[0] if symbols else None),
                )
                report.errors.append(f"retrain_request:{req.get('reason')}")
                logger.info(
                    "retrain_request_written",
                    reason=req.get("reason"),
                    status=req.get("status"),
                )
        except Exception as exc:
            logger.warning("retrain_advisory_check_failed", error=str(exc))
    risk = RiskManager()
    conf_thr = float(cfg.get("confidence_threshold", 0.60))
    sl_mult = float(cfg.get("stop_loss_atr_multiplier", 1.5))
    tp_mult = float(cfg.get("take_profit_atr_multiplier", 2.5))
    trades_log = PROJECT_ROOT / "logs" / "live" / "trades_log.jsonl"
    trades_log.parent.mkdir(parents=True, exist_ok=True)
    decisions_log = PROJECT_ROOT / "logs" / "live" / "decisions_log.jsonl"

    with mt5_session() as client:
        account = client.account_summary()
        equity = float(account.get("equity") or account.get("balance") or 0)
        risk.update_equity(equity)
        mt5 = _mt5_module()
        positions = list(mt5.positions_get() or [])

        for symbol in symbols:
            report.iterations += 1
            try:
                # Prefer model trained on this TF (FinalModel only if TF matches).
                sig = analyze_timeframe_signal(
                    client,
                    symbol,
                    timeframe,
                    allow_ungated=allow_ungated,
                    match_timeframe_only=not bool(cfg.get("final_model_any_timeframe", False)),
                )
                pred, conf, dbg = int(sig["pred"]), float(sig["conf"]), dict(sig["debug"] or {})
                featured = sig["featured"]
                bundle = sig["bundle"]

                # Optional legacy confirm layer for single-TF runs.
                if (
                    pred != 0
                    and bool(cfg.get("multi_tf_confirm", False))
                    and not bool(cfg.get("multi_tf_fusion", True))
                    and bundle.get("artifact_type") != "llmodel"
                ):
                    from atis.engines.engine4_training.multi_tf_decision import (
                        confirm_tfs_for_primary,
                        multi_tf_decision,
                    )

                    confirm_tfs = confirm_tfs_for_primary(timeframe, cfg)
                    confirm_dbg: list[dict[str, Any]] = []
                    for ctf in confirm_tfs[:4]:
                        try:
                            csig = analyze_timeframe_signal(
                                client, symbol, ctf, allow_ungated=True, match_timeframe_only=True
                            )
                            confirm_dbg.append({"tf": ctf, "pred": int(csig["pred"]), "conf": float(csig["conf"])})
                        except Exception as exc:
                            confirm_dbg.append({"tf": ctf, "error": str(exc)})
                    pred, mtf_dbg = multi_tf_decision(
                        int(pred),
                        float(conf),
                        confirm_dbg,
                        mode=str(cfg.get("multi_tf_mode", "soft_veto")),
                        min_confirm_agree=int(cfg.get("min_confirm_agree", 1)),
                        veto_opposite_htf=bool(cfg.get("veto_opposite_htf", True)),
                        primary_tf=str(timeframe),
                    )
                    dbg["multi_tf_confirm"] = confirm_dbg
                    dbg["multi_tf_decision"] = mtf_dbg
                    if pred == 0:
                        dbg["reason"] = mtf_dbg.get("reason", "multi_tf_veto")

                decision = {
                    "ts": _utc_now().isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "pred": pred,
                    "confidence": conf,
                    "threshold": conf_thr,
                    "model_type": bundle.get("artifact_type", "baseline"),
                    "model_version": bundle.get("version"),
                    "model_timeframe": bundle.get("timeframe", timeframe),
                    "debug": dbg,
                    "close": float(featured["close"].iloc[-1]),
                    "live_spread_pips": sig.get("live_spread_pips"),
                }
                # Real-time pattern drawings on MT5 (async; does not block orders).
                overlay_meta = _sync_pattern_overlay(
                    client, featured, symbol=symbol, timeframe=timeframe
                )
                decision["pattern_overlay"] = {
                    k: overlay_meta[k]
                    for k in ("enabled", "count", "keys", "error")
                    if k in overlay_meta
                }
                _append_decision(decisions_log, decision)
                # Merge once: decision and dbg both may carry live_spread_pips (and similar keys).
                decision_fields = {k: v for k, v in decision.items() if k != "debug"}
                log_fields = {**(dbg or {}), **decision_fields}
                logger.info("live_decision", **log_fields)

                _place_from_signal(
                    report=report,
                    risk=risk,
                    client=client,
                    symbol=symbol,
                    timeframe=timeframe,
                    pred=pred,
                    conf=conf,
                    featured=featured,
                    equity=equity,
                    positions=positions,
                    dry_run=dry_run,
                    mode=exec_mode,
                    conf_thr=conf_thr,
                    sl_mult=sl_mult,
                    tp_mult=tp_mult,
                    trades_log=trades_log,
                    reason_suffix=f";reason={dbg.get('reason')}",
                    prediction_debug=dbg,
                )
            except Exception as exc:
                report.errors.append(f"{symbol}:{exc}")
                logger.exception("live_symbol_failed", symbol=symbol, error=str(exc))

    report.finished_at = _utc_now().isoformat()
    out = PROJECT_ROOT / "logs" / "live" / "live_run_report.json"
    out.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    return report


def run_live_multi_tf(
    symbols: list[str],
    timeframes: list[str],
    *,
    dry_run: bool = True,
    allow_ungated: bool = True,
) -> LiveLoopReport:
    """Run selected timeframes with their trained models.

    Default (independent): each TF analyzes its own bars/model and may trade
    on its own signal — no cross-TF fusion or veto.

    Optional legacy fusion: set ``multi_tf_fusion: true`` to merge votes into
    one decision before placing a single order.
    """
    ensure_project_dirs()
    set_global_seed()
    cfg = _cfg()
    mode = str(cfg.get("mode", "paper"))
    if mode == "live":
        raise RuntimeError("Live real-money mode requires explicit user approval — refused.")

    full = load_engine_config()
    allowed = set(full.get("trading", {}).get("allowed_live_symbols", ["XAUUSD"]))
    if not symbols:
        symbols = list(cfg.get("symbols") or [full.get("trading", {}).get("primary_symbol", "XAUUSD")])
    bad = [s for s in symbols if s not in allowed]
    if bad:
        raise RuntimeError(f"Live trading restricted to {sorted(allowed)}; refused: {bad}")

    tfs = [str(t).upper() for t in (timeframes or []) if str(t).strip()]
    if not tfs:
        tfs = [str(cfg.get("timeframe") or full.get("trading", {}).get("primary_timeframe", "H1"))]
    # De-dupe preserve order
    seen: set[str] = set()
    tfs = [t for t in tfs if not (t in seen or seen.add(t))]

    # Single TF → dedicated path (still uses trained model for that TF).
    if len(tfs) == 1:
        return run_live_once(symbols, tfs[0], dry_run=dry_run, allow_ungated=allow_ungated)

    independent = bool(cfg.get("multi_tf_independent", True)) and not bool(
        cfg.get("multi_tf_fusion", False)
    )
    if independent:
        return _run_live_multi_tf_independent(
            symbols,
            tfs,
            dry_run=dry_run,
            allow_ungated=allow_ungated,
        )
    return _run_live_multi_tf_fused(
        symbols,
        tfs,
        dry_run=dry_run,
        allow_ungated=allow_ungated,
    )


def _run_live_multi_tf_independent(
    symbols: list[str],
    tfs: list[str],
    *,
    dry_run: bool = True,
    allow_ungated: bool = True,
) -> LiveLoopReport:
    """Each TF: own model, own bars, own decision, own order attempt."""
    cfg = _cfg()
    report = LiveLoopReport(started_at=_utc_now().isoformat())
    risk = RiskManager()
    conf_thr = float(cfg.get("confidence_threshold", 0.60))
    sl_mult = float(cfg.get("stop_loss_atr_multiplier", 1.5))
    tp_mult = float(cfg.get("take_profit_atr_multiplier", 2.5))
    trades_log = PROJECT_ROOT / "logs" / "live" / "trades_log.jsonl"
    trades_log.parent.mkdir(parents=True, exist_ok=True)
    decisions_log = PROJECT_ROOT / "logs" / "live" / "decisions_log.jsonl"
    exec_mode = "paper" if dry_run else "demo"

    with mt5_session() as client:
        account = client.account_summary()
        equity = float(account.get("equity") or account.get("balance") or 0)
        risk.update_equity(equity)
        mt5 = _mt5_module()
        positions = list(mt5.positions_get() or [])

        for symbol in symbols:
            report.iterations += 1
            # No shared HTF cache across TFs — each timeframe stays isolated.
            for tf in tfs:
                try:
                    sig = analyze_timeframe_signal(
                        client,
                        symbol,
                        tf,
                        allow_ungated=allow_ungated,
                        match_timeframe_only=True,
                        htf_cache=None,
                    )
                    pred = int(sig["pred"])
                    conf = float(sig["conf"])
                    featured = sig["featured"]
                    dbg = dict(sig.get("debug") or {})
                    dbg["multi_tf_mode"] = "independent"
                    dbg["reason"] = dbg.get("reason") or "per_tf_independent"

                    decision = {
                        "ts": _utc_now().isoformat(),
                        "symbol": symbol,
                        "timeframe": tf,
                        "timeframes": [tf],
                        "pred": pred,
                        "confidence": conf,
                        "threshold": conf_thr,
                        "model_type": sig.get("model_type") or "per_tf",
                        "model_version": sig.get("model_version"),
                        "model_timeframe": sig.get("model_timeframe", tf),
                        "debug": dbg,
                        "close": float(featured["close"].iloc[-1]),
                        "live_spread_pips": sig.get("live_spread_pips"),
                    }
                    overlay_meta = _sync_pattern_overlay(
                        client, featured, symbol=symbol, timeframe=tf
                    )
                    decision["pattern_overlay"] = {
                        k: overlay_meta[k]
                        for k in ("enabled", "count", "keys", "error")
                        if k in overlay_meta
                    }
                    _append_decision(decisions_log, decision)
                    logger.info(
                        "live_tf_independent_decision",
                        symbol=symbol,
                        timeframe=tf,
                        pred=pred,
                        conf=conf,
                        model=sig.get("model_version"),
                        model_tf=sig.get("model_timeframe"),
                    )

                    _place_from_signal(
                        report=report,
                        risk=risk,
                        client=client,
                        symbol=symbol,
                        timeframe=tf,
                        pred=pred,
                        conf=conf,
                        featured=featured,
                        equity=equity,
                        positions=positions,
                        dry_run=dry_run,
                        mode=exec_mode,
                        conf_thr=conf_thr,
                        sl_mult=sl_mult,
                        tp_mult=tp_mult,
                        trades_log=trades_log,
                        reason_suffix=";mode=independent",
                        prediction_debug=dbg,
                    )
                    # Broker refresh only for live/demo fills; paper keeps synthetic slots.
                    if not dry_run:
                        try:
                            positions[:] = list(mt5.positions_get() or [])
                        except Exception:
                            pass
                except Exception as exc:
                    report.errors.append(f"{symbol}/{tf}:{exc}")
                    logger.warning(
                        "mtf_independent_tf_failed",
                        symbol=symbol,
                        tf=tf,
                        error=str(exc),
                    )

    report.finished_at = _utc_now().isoformat()
    out = PROJECT_ROOT / "logs" / "live" / "live_run_report.json"
    out.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    return report


def _run_live_multi_tf_fused(
    symbols: list[str],
    tfs: list[str],
    *,
    dry_run: bool = True,
    allow_ungated: bool = True,
) -> LiveLoopReport:
    """Legacy path: analyze all TFs, fuse votes, place one order."""
    from atis.engines.engine4_training.multi_tf_decision import fuse_multi_tf_votes

    cfg = _cfg()
    report = LiveLoopReport(started_at=_utc_now().isoformat())
    risk = RiskManager()
    conf_thr = float(cfg.get("confidence_threshold", 0.60))
    sl_mult = float(cfg.get("stop_loss_atr_multiplier", 1.5))
    tp_mult = float(cfg.get("take_profit_atr_multiplier", 2.5))
    trades_log = PROJECT_ROOT / "logs" / "live" / "trades_log.jsonl"
    trades_log.parent.mkdir(parents=True, exist_ok=True)
    decisions_log = PROJECT_ROOT / "logs" / "live" / "decisions_log.jsonl"
    min_agree = int(cfg.get("multi_tf_min_agree", cfg.get("min_confirm_agree", 2)))
    fusion_mode = str(cfg.get("multi_tf_fusion_mode", "weighted_consensus"))
    exec_mode = "paper" if dry_run else "demo"

    with mt5_session() as client:
        account = client.account_summary()
        equity = float(account.get("equity") or account.get("balance") or 0)
        risk.update_equity(equity)
        mt5 = _mt5_module()
        positions = list(mt5.positions_get() or [])

        for symbol in symbols:
            report.iterations += 1
            votes: list[dict[str, Any]] = []
            signals_by_tf: dict[str, dict[str, Any]] = {}
            htf_cache: dict[str, pd.DataFrame] = {}
            try:
                for tf in tfs:
                    try:
                        sig = analyze_timeframe_signal(
                            client,
                            symbol,
                            tf,
                            allow_ungated=allow_ungated,
                            match_timeframe_only=True,
                            htf_cache=htf_cache,
                        )
                        signals_by_tf[tf] = sig
                        sig_dbg = dict(sig.get("debug") or {})
                        votes.append({
                            "tf": tf,
                            "pred": int(sig["pred"]),
                            "conf": float(sig["conf"]),
                            "model_type": sig.get("model_type"),
                            "model_version": sig.get("model_version"),
                            "model_timeframe": sig.get("model_timeframe"),
                            "live_spread_pips": sig.get("live_spread_pips"),
                            "expected_return": sig_dbg.get("expected_return"),
                            "risk_score": sig_dbg.get("risk_score"),
                        })
                        logger.info(
                            "mtf_tf_signal",
                            symbol=symbol,
                            tf=tf,
                            pred=sig["pred"],
                            conf=sig["conf"],
                            model=sig.get("model_version"),
                            model_tf=sig.get("model_timeframe"),
                        )
                    except Exception as exc:
                        votes.append({"tf": tf, "error": str(exc)})
                        report.errors.append(f"{symbol}/{tf}:{exc}")
                        logger.warning("mtf_tf_failed", symbol=symbol, tf=tf, error=str(exc))

                pred, conf, fuse_dbg = fuse_multi_tf_votes(
                    votes,
                    mode=fusion_mode,
                    min_agree=min_agree,
                    min_avg_conf=conf_thr,
                    veto_opposite_htf=bool(cfg.get("veto_opposite_htf", True)),
                )
                exec_tf = str(fuse_dbg.get("execution_tf") or tfs[0]).upper()
                featured = None
                if exec_tf in signals_by_tf:
                    featured = signals_by_tf[exec_tf]["featured"]
                elif signals_by_tf:
                    first_key = next(iter(signals_by_tf))
                    featured = signals_by_tf[first_key]["featured"]
                    exec_tf = first_key

                exit_agg = aggregate_prediction_exits(votes, pred)
                decision_debug = {
                    "reason": fuse_dbg.get("reason"),
                    "multi_tf_fusion": fuse_dbg,
                    "votes": votes,
                    "scenario_probabilities": {
                        "buy": fuse_dbg.get("buy_votes", 0) / max(len([v for v in votes if "error" not in v]), 1),
                        "sell": fuse_dbg.get("sell_votes", 0) / max(len([v for v in votes if "error" not in v]), 1),
                        "hold": fuse_dbg.get("flat_votes", 0) / max(len([v for v in votes if "error" not in v]), 1),
                    },
                    "attention_by_timeframe": {
                        str(v.get("tf")): float(v.get("conf") or 0.0)
                        for v in votes if "error" not in v
                    },
                    **exit_agg,
                }
                decision = {
                    "ts": _utc_now().isoformat(),
                    "symbol": symbol,
                    "timeframe": exec_tf,
                    "timeframes": tfs,
                    "pred": pred,
                    "confidence": conf,
                    "threshold": conf_thr,
                    "model_type": "multi_tf_fusion",
                    "debug": decision_debug,
                    "close": float(featured["close"].iloc[-1]) if featured is not None else None,
                }
                if featured is not None:
                    overlay_meta = _sync_pattern_overlay(
                        client, featured, symbol=symbol, timeframe=exec_tf
                    )
                    decision["pattern_overlay"] = {
                        k: overlay_meta[k]
                        for k in ("enabled", "count", "keys", "error")
                        if k in overlay_meta
                    }
                _append_decision(decisions_log, decision)
                logger.info(
                    "live_mtf_decision",
                    symbol=symbol,
                    pred=pred,
                    conf=conf,
                    reason=fuse_dbg.get("reason"),
                    exec_tf=exec_tf,
                    buy=fuse_dbg.get("buy_votes"),
                    sell=fuse_dbg.get("sell_votes"),
                    expected_return=exit_agg.get("expected_return"),
                    risk_score=exit_agg.get("risk_score"),
                )

                if featured is None:
                    report.errors.append(f"{symbol}:no_featured_frame_after_fusion")
                    continue

                _place_from_signal(
                    report=report,
                    risk=risk,
                    client=client,
                    symbol=symbol,
                    timeframe=exec_tf,
                    pred=pred,
                    conf=conf,
                    featured=featured,
                    equity=equity,
                    positions=positions,
                    dry_run=dry_run,
                    mode=exec_mode,
                    conf_thr=conf_thr,
                    sl_mult=sl_mult,
                    tp_mult=tp_mult,
                    trades_log=trades_log,
                    reason_suffix=f";fusion={fuse_dbg.get('reason')}",
                    prediction_debug=decision_debug,
                )
            except Exception as exc:
                report.errors.append(f"{symbol}:{exc}")
                logger.exception("live_mtf_symbol_failed", symbol=symbol, error=str(exc))

    report.finished_at = _utc_now().isoformat()
    out = PROJECT_ROOT / "logs" / "live" / "live_run_report.json"
    out.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    return report


def run_live_loop(
    symbols: list[str],
    timeframe: str = "M5",
    *,
    interval_seconds: int = 60,
    max_iterations: int | None = None,
    dry_run: bool = True,
    allow_ungated: bool = True,
) -> None:
    """Polling loop — stops on kill switch or max_iterations."""
    n = 0
    while True:
        if _cfg().get("kill_switch"):
            logger.error("loop_stopped_kill_switch")
            break
        run_live_once(
            symbols,
            timeframe,
            dry_run=dry_run,
            allow_ungated=allow_ungated,
        )
        n += 1
        if max_iterations is not None and n >= max_iterations:
            break
        time.sleep(interval_seconds)
