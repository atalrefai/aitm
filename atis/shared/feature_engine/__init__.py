"""
Shared feature engine — single implementation for training AND live (Principle 1.4).

All indicators at time t use only data <= t (no look-ahead bias).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from atis.config import CONFIG_DIR


def load_indicators_config(path: Path | None = None) -> dict[str, Any]:
    p = path or (CONFIG_DIR / "indicators.yaml")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Primitive indicators (causal)
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window=period, min_periods=period).apply(_wma, raw=True)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger(
    close: pd.Series,
    period: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()
    md = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())),
        raw=True,
    )
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    return close.pct_change(periods=period) * 100


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_atr = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / tr_atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / tr_atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_af: float = 0.2,
) -> pd.Series:
    n = len(high)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=high.index)

    bull = True
    af = step
    ep = float(high.iloc[0])
    sar[0] = float(low.iloc[0])

    for i in range(1, n):
        prev = sar[i - 1]
        if bull:
            sar[i] = prev + af * (ep - prev)
            sar[i] = min(sar[i], float(low.iloc[i - 1]), float(low.iloc[max(0, i - 2)]))
            if float(low.iloc[i]) < sar[i]:
                bull = False
                sar[i] = ep
                ep = float(low.iloc[i])
                af = step
            else:
                if float(high.iloc[i]) > ep:
                    ep = float(high.iloc[i])
                    af = min(af + step, max_af)
        else:
            sar[i] = prev + af * (ep - prev)
            sar[i] = max(sar[i], float(high.iloc[i - 1]), float(high.iloc[max(0, i - 2)]))
            if float(high.iloc[i]) > sar[i]:
                bull = True
                sar[i] = ep
                ep = float(high.iloc[i])
                af = step
            else:
                if float(low.iloc[i]) < ep:
                    ep = float(low.iloc[i])
                    af = min(af + step, max_af)
    return pd.Series(sar, index=high.index)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> dict[str, pd.Series]:
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)  # plotted ahead — for features use unshifted mid
    # For ML features avoid look-ahead: use current displacement components without future shift
    span_a_now = (tenkan_sen + kijun_sen) / 2
    span_b_now = (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "span_a": span_a_now,
        "span_b": span_b_now,
        # Keep shifted cloud only as optional display — NOT used as causal feature
        "span_a_plot": span_a,
    }


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0)).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3.0
    cum_vol = volume.fillna(0).cumsum().replace(0, np.nan)
    return (tp * volume.fillna(0)).cumsum() / cum_vol


# ---------------------------------------------------------------------------
# Patterns & context
# ---------------------------------------------------------------------------

from atis.shared.feature_engine.patterns import (  # noqa: E402
    candlestick_patterns,
    compound_patterns,
    structural_patterns,
    swing_support_resistance,
)


def session_tag(timestamps: pd.Series) -> pd.Series:
    """Tag trading session by UTC hour: asia / europe / us."""
    hour = pd.to_datetime(timestamps, utc=True).dt.hour

    def _tag(h: int) -> str:
        if 0 <= h < 7:
            return "asia"
        if 7 <= h < 13:
            return "europe"
        if 13 <= h < 21:
            return "us"
        return "asia"

    return hour.map(_tag)


def volatility_regime(atr_series: pd.Series, lookback: int = 100) -> pd.Series:
    """Classify ATR percentile into calm / normal / violent."""
    pct = atr_series.rolling(lookback, min_periods=max(20, lookback // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )

    def _reg(p: float) -> str:
        if np.isnan(p):
            return "unknown"
        if p < 0.33:
            return "calm"
        if p < 0.66:
            return "normal"
        return "violent"

    return pct.map(_reg)


def trend_strength_score(close: pd.Series, adx_series: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """Composite 0–100-ish score from ADX and EMA alignment."""
    align = np.where(ema_fast > ema_slow, 1.0, -1.0)
    score = adx_series.fillna(0) * align
    return pd.Series(score, index=close.index)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Compute full feature set for one (symbol, timeframe) OHLCV frame.
    Input must be sorted by timestamp ascending. No future leakage.
    """
    if df.empty:
        return df.copy()

    cfg = config or load_indicators_config()
    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    close, high, low, open_ = out["close"], out["high"], out["low"], out["open"]
    volume = out["tick_volume"] if "tick_volume" in out.columns else out.get("real_volume", pd.Series(0, index=out.index))

    trend = cfg.get("trend", {})
    for p in trend.get("sma_periods", []):
        out[f"sma_{p}"] = sma(close, int(p))
    for p in trend.get("ema_periods", []):
        out[f"ema_{p}"] = ema(close, int(p))
    for p in trend.get("wma_periods", []):
        out[f"wma_{p}"] = wma(close, int(p))

    macd_cfg = trend.get("macd", {})
    m_line, m_sig, m_hist = macd(
        close,
        int(macd_cfg.get("fast", 12)),
        int(macd_cfg.get("slow", 26)),
        int(macd_cfg.get("signal", 9)),
    )
    out["macd"], out["macd_signal"], out["macd_hist"] = m_line, m_sig, m_hist
    out["adx"] = adx(high, low, close, int(trend.get("adx_period", 14)))

    ps = trend.get("parabolic_sar", {})
    out["psar"] = parabolic_sar(high, low, float(ps.get("step", 0.02)), float(ps.get("max_af", 0.2)))

    ich = trend.get("ichimoku", {})
    ich_out = ichimoku(
        high,
        low,
        int(ich.get("tenkan", 9)),
        int(ich.get("kijun", 26)),
        int(ich.get("senkou_b", 52)),
    )
    out["ichimoku_tenkan"] = ich_out["tenkan_sen"]
    out["ichimoku_kijun"] = ich_out["kijun_sen"]
    out["ichimoku_span_a"] = ich_out["span_a"]
    out["ichimoku_span_b"] = ich_out["span_b"]

    mom = cfg.get("momentum", {})
    for p in mom.get("rsi_periods", [14]):
        out[f"rsi_{p}"] = rsi(close, int(p))
    st = mom.get("stochastic", {})
    k, d = stochastic(high, low, close, int(st.get("k_period", 14)), int(st.get("d_period", 3)))
    out["stoch_k"], out["stoch_d"] = k, d
    out["cci"] = cci(high, low, close, int(mom.get("cci_period", 20)))
    out["williams_r"] = williams_r(high, low, close, int(mom.get("williams_r_period", 14)))
    out["roc"] = roc(close, int(mom.get("roc_period", 12)))

    vol = cfg.get("volatility", {})
    bb = vol.get("bollinger", {})
    bu, bm, bl = bollinger(close, int(bb.get("period", 20)), float(bb.get("std", 2.0)))
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bu, bm, bl
    out["atr"] = atr(high, low, close, int(vol.get("atr_period", 14)))
    std_p = int(vol.get("std_channel_period", 20))
    out["std_close"] = close.rolling(std_p, min_periods=std_p).std(ddof=0)
    kel = vol.get("keltner", {})
    kel_mid = ema(close, int(kel.get("period", 20)))
    kel_atr = atr(high, low, close, int(kel.get("period", 20)))
    mult = float(kel.get("atr_mult", 1.5))
    out["keltner_mid"] = kel_mid
    out["keltner_upper"] = kel_mid + mult * kel_atr
    out["keltner_lower"] = kel_mid - mult * kel_atr

    if cfg.get("volume", {}).get("enabled", True):
        out["obv"] = obv(close, volume)
        out["vwap"] = vwap(high, low, close, volume)
        out["vroc"] = volume.pct_change(12) * 100

    if cfg.get("patterns", {}).get("candlestick", True):
        pats = candlestick_patterns(out)
        out = pd.concat([out, pats], axis=1)

    if cfg.get("patterns", {}).get("structural", True):
        struct = swing_support_resistance(out)
        charts = structural_patterns(out)
        out = pd.concat([out, struct, charts], axis=1)

    if cfg.get("patterns", {}).get("compound", True):
        comps = compound_patterns(out)
        out = pd.concat([out, comps], axis=1)

    ctx = cfg.get("context", {})
    ctx_cols: dict[str, Any] = {}
    if ctx.get("session_tagging", True):
        ctx_cols["session"] = session_tag(out["timestamp"])
    if ctx.get("volatility_regime", True):
        ctx_cols["vol_regime"] = volatility_regime(out["atr"])
    if ctx.get("trend_strength", True):
        ema_f = out.get("ema_20", ema(close, 20))
        ema_s = out.get("ema_50", ema(close, 50))
        ctx_cols["trend_strength"] = trend_strength_score(close, out["adx"], ema_f, ema_s)
    ctx_cols["label"] = np.nan
    ctx_cols["label_meta"] = None
    out = pd.concat([out, pd.DataFrame(ctx_cols, index=out.index)], axis=1)

    return out
