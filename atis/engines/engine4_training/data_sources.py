"""Training data loading from JSON artifacts produced by prior stages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atis.config import get_path, load_engine_config
from atis.shared.data_json import load_timeframe_json
from atis.shared.data_registry import DataStateRegistry
from atis.shared.pattern_store import load_section, section_path, load_pattern_signal_matrix


# Higher→lower hierarchy for causal context injection (merge_asof backward).
_TF_RANK = {"M1": 1, "M5": 2, "M15": 3, "M30": 4, "H1": 5, "H4": 6, "D1": 7, "W1": 8, "MN1": 9}

# Scale-free / structural HTF columns that improve directional discrimination.
_HTF_CONTEXT_COLS = (
    "trend_strength",
    "structure_hh_hl",
    "adx",
    "rsi_14",
    "chart_pattern_score",
    "pat_bias",
    "pat_strength",
    "dist_to_support",
    "dist_to_resist",
    "trendline_slope",
    "macd_hist",
    "atr",
    "close",
)


def features_json_path(symbol: str, timeframe: str) -> Path:
    return get_path("data_features") / symbol / timeframe / "features.json"


def higher_timeframes_for(timeframe: str, available: list[str] | None = None) -> list[str]:
    """Return strictly higher timeframes (same symbol) for multi-TF context."""
    rank = _TF_RANK.get(str(timeframe).upper())
    if rank is None:
        return []
    pool = available
    if pool is None:
        cfg = load_engine_config().get("engine4_training", {}) or {}
        pool = list(cfg.get("default_timeframes") or list(_TF_RANK.keys()))
        extra = cfg.get("cross_tf_sources") or []
        pool = list(dict.fromkeys([*pool, *extra]))
    out = []
    for tf in pool:
        r = _TF_RANK.get(str(tf).upper())
        if r is not None and r > rank:
            out.append(str(tf).upper())
    out.sort(key=lambda t: _TF_RANK.get(t, 0))
    return out


def _peek_json_int(path: Path, key: str, *, max_bytes: int = 65536) -> int | None:
    """Read a top-level integer field without parsing the whole JSON file.

    features.json / discovery_log.json can be hundreds of MB; the UI and
    training_source_meta only need header scalars like row_count / count.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(max_bytes)
    except OSError:
        return None
    m = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', head)
    if not m:
        return None
    return int(m.group(1))


def training_source_meta(symbol: str, timeframe: str) -> dict[str, Any]:
    path = features_json_path(symbol, timeframe)
    exists = path.exists()
    registry = DataStateRegistry()
    registry_doc = registry.load_timeframe_doc(timeframe) or {}
    symbol_layers = ((registry_doc.get("symbols") or {}).get(symbol) or {})
    feat_layer = symbol_layers.get("features") or {}
    rows = int(feat_layer.get("row_count") or 0)
    if exists and rows <= 0:
        rows = int(_peek_json_int(path, "row_count") or 0)

    def _section_count(section: str) -> int:
        return int(_peek_json_int(section_path(symbol, timeframe, section), "count") or 0)

    return {
        "features_json_path": str(path),
        "features_json_exists": exists,
        "features_json_rows": rows,
        "registry_json_path": str(registry.timeframe_path(timeframe)),
        "pattern_paths": {
            section: str(section_path(symbol, timeframe, section))
            for section in ("candlesticks", "structural", "compounds", "knowledge", "discovery_log")
        },
        "registry": symbol_layers,
        "pattern_summary": {
            "knowledge_count": _section_count("knowledge"),
            "compound_count": _section_count("compounds"),
            "candle_count": _section_count("candlesticks"),
            "structural_count": _section_count("structural"),
            "discovery_count": _section_count("discovery_log"),
        },
    }


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values))) if values else 0.0


def _htf_context_from_frame(frame: pd.DataFrame, htf: str) -> pd.DataFrame | None:
    """Build causal HTF context columns from a features dataframe."""
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return None
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    keep = ["timestamp"]
    for col in _HTF_CONTEXT_COLS:
        if col not in work.columns:
            continue
        if col == "close":
            keep.append(col)
            continue
        if col == "atr" and "close" in work.columns:
            atr_pct = work["atr"].astype(float) / work["close"].astype(float).replace(0, np.nan)
            out_name = f"htf_{htf}__atr_pct"
            work[out_name] = atr_pct.replace([np.inf, -np.inf], np.nan)
            keep.append(out_name)
            continue
        out_name = f"htf_{htf}__{col}"
        work[out_name] = pd.to_numeric(work[col], errors="coerce")
        keep.append(out_name)
    cols = [c for c in keep if c in work.columns]
    out = work[cols].drop(columns=["close"], errors="ignore")
    return out.drop_duplicates(subset=["timestamp"], keep="last")


def _load_htf_context_frame(symbol: str, htf: str) -> pd.DataFrame | None:
    path = features_json_path(symbol, htf)
    if not path.exists():
        return None
    frame = load_timeframe_json(path)
    return _htf_context_from_frame(frame, htf)


def enrich_with_higher_timeframes(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    higher_tfs: list[str] | None = None,
    htf_frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Causal multi-TF enrichment: merge higher-TF context via merge_asof(backward).

    Adds htf_{TF}__* columns plus agreement features vs local structure/RSI/ADX.
    Never peeks into the future — only HTF bars with timestamp <= local bar.

    ``htf_frames``: optional precomputed feature frames keyed by TF (live path).
    When provided, preferred over disk features.json for that TF.
    """
    meta: dict[str, Any] = {"enabled": True, "sources": [], "n_htf_cols": 0}
    if "timestamp" not in df.columns:
        meta["enabled"] = False
        meta["reason"] = "no_timestamp"
        return df, meta

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.sort_values("timestamp").reset_index(drop=True)
    sources = higher_tfs if higher_tfs is not None else higher_timeframes_for(timeframe)
    added_cols: list[str] = []

    for htf in sources:
        htf_frame = None
        if htf_frames and htf in htf_frames:
            htf_frame = _htf_context_from_frame(htf_frames[htf], htf)
            src = "live"
        else:
            htf_frame = _load_htf_context_frame(symbol, htf)
            src = "disk"
        if htf_frame is None or len(htf_frame) < 5:
            continue
        before = set(work.columns)
        work = pd.merge_asof(
            work,
            htf_frame.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        new_cols = [c for c in work.columns if c not in before]
        added_cols.extend(new_cols)
        meta["sources"].append({
            "timeframe": htf,
            "rows": int(len(htf_frame)),
            "cols": new_cols,
            "source": src,
        })

        htf_struct = f"htf_{htf}__structure_hh_hl"
        htf_rsi = f"htf_{htf}__rsi_14"
        htf_adx = f"htf_{htf}__adx"
        htf_trend = f"htf_{htf}__trend_strength"
        if htf_struct in work.columns and "structure_hh_hl" in work.columns:
            local = work["structure_hh_hl"].astype(float)
            remote = work[htf_struct].astype(float)
            agree = f"mtf_{htf}_structure_agree"
            work[agree] = np.sign(local.fillna(0.0)) * np.sign(remote.fillna(0.0))
            added_cols.append(agree)
        if htf_rsi in work.columns and "rsi_14" in work.columns:
            delta = f"mtf_{htf}_rsi_delta"
            work[delta] = work["rsi_14"].astype(float) - work[htf_rsi].astype(float)
            added_cols.append(delta)
        if htf_adx in work.columns and "adx" in work.columns:
            ratio = f"mtf_{htf}_adx_ratio"
            work[ratio] = work["adx"].astype(float) / (work[htf_adx].astype(float).abs() + 1e-6)
            added_cols.append(ratio)
        if htf_trend in work.columns and "trend_strength" in work.columns:
            tdelta = f"mtf_{htf}_trend_delta"
            work[tdelta] = work["trend_strength"].astype(float) - work[htf_trend].astype(float)
            added_cols.append(tdelta)

    # Multi-TF consensus: mean structure agreement / RSI alignment across HTFs.
    agree_cols = [c for c in work.columns if str(c).endswith("_structure_agree")]
    if agree_cols:
        work["feat_mtf_structure_consensus"] = work[agree_cols].astype(float).mean(axis=1)
        added_cols.append("feat_mtf_structure_consensus")
    rsi_delta_cols = [c for c in work.columns if str(c).endswith("_rsi_delta")]
    if rsi_delta_cols:
        work["feat_mtf_rsi_dispersion"] = work[rsi_delta_cols].astype(float).abs().mean(axis=1)
        added_cols.append("feat_mtf_rsi_dispersion")
    trend_delta_cols = [c for c in work.columns if str(c).endswith("_trend_delta")]
    if trend_delta_cols:
        work["feat_mtf_trend_consensus"] = (
            np.sign(work[trend_delta_cols].astype(float)).mean(axis=1)
        )
        added_cols.append("feat_mtf_trend_consensus")

    meta["n_htf_cols"] = len(added_cols)
    meta["columns"] = added_cols
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].replace([np.inf, -np.inf], np.nan)
    return work, meta


def load_training_frame(symbol: str, timeframe: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the feature JSON plus registry/pattern JSON context for training."""
    path = features_json_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"features_json_missing:{symbol}:{timeframe}")

    df = load_timeframe_json(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    registry_doc = DataStateRegistry().load_timeframe_doc(timeframe) or {}
    symbol_layers = ((registry_doc.get("symbols") or {}).get(symbol) or {})
    raw_meta = symbol_layers.get("raw") or {}
    clean_meta = symbol_layers.get("clean") or {}
    feat_meta = symbol_layers.get("features") or {}

    sections = {
        "candlesticks": load_section(symbol, timeframe, "candlesticks") or {},
        "structural": load_section(symbol, timeframe, "structural") or {},
        "compounds": load_section(symbol, timeframe, "compounds") or {},
        "knowledge": load_section(symbol, timeframe, "knowledge") or {},
        "discovery_log": load_section(symbol, timeframe, "discovery_log") or {},
        "new_patterns": load_section(symbol, timeframe, "new_patterns") or {},
        "rankings": load_section(symbol, timeframe, "rankings") or {},
    }
    knowledge_items = list(sections["knowledge"].get("items") or [])
    compound_items = list(sections["compounds"].get("items") or [])
    candle_items = list(sections["candlesticks"].get("items") or [])
    structural_items = list(sections["structural"].get("items") or [])
    discovery_items = list(sections["discovery_log"].get("items") or [])
    new_items = list(sections["new_patterns"].get("items") or [])
    recommended = list((sections["rankings"] or {}).get("engine4_recommended") or [])
    approved_new = [x for x in new_items if x.get("approved")]

    success_vals = [float(x["success_rate"]) for x in knowledge_items if x.get("success_rate") is not None]
    conf_vals = [float(x["confidence"]) for x in knowledge_items if x.get("confidence") is not None]
    fwd_vals = [float(x["avg_forward_return"]) for x in knowledge_items if x.get("avg_forward_return") is not None]
    total_occ = float(sum(float(x.get("occurrences") or 0.0) for x in knowledge_items))
    bull_occ = float(
        sum(float(x.get("occurrences") or 0.0) for x in knowledge_items if str(x.get("bias")) == "bullish")
    )
    bear_occ = float(
        sum(float(x.get("occurrences") or 0.0) for x in knowledge_items if str(x.get("bias")) == "bearish")
    )

    context_cols = {
        "registry_raw_rows": float(raw_meta.get("row_count") or 0.0),
        "registry_clean_rows": float(clean_meta.get("row_count") or 0.0),
        "registry_feature_rows": float(feat_meta.get("row_count") or 0.0),
        "registry_raw_ok": 1.0 if raw_meta.get("last_run_status") == "success" else 0.0,
        "registry_clean_ok": 1.0 if clean_meta.get("last_run_status") == "success" else 0.0,
        "registry_feature_ok": 1.0 if feat_meta.get("last_run_status") == "success" else 0.0,
        "pattern_knowledge_count": float(len(knowledge_items)),
        "pattern_compound_count": float(len(compound_items)),
        "pattern_candle_count": float(len(candle_items)),
        "pattern_structural_count": float(len(structural_items)),
        "pattern_discovery_count": float(len(discovery_items)),
        "pattern_new_count": float(len(new_items)),
        "pattern_new_approved_count": float(len(approved_new)),
        "pattern_engine4_recommended_count": float(len(recommended)),
        "pattern_approved_ratio": float(
            sum(1 for x in knowledge_items if x.get("approved")) / max(len(knowledge_items), 1)
        ),
        "pattern_total_occurrences": total_occ,
        "pattern_success_rate_mean": _safe_mean(success_vals),
        "pattern_confidence_mean": _safe_mean(conf_vals),
        "pattern_forward_return_mean": _safe_mean(fwd_vals),
        "pattern_bullish_ratio": bull_occ / max(total_occ, 1.0),
        "pattern_bearish_ratio": bear_occ / max(total_occ, 1.0),
    }
    for col, value in context_cols.items():
        df[col] = float(value)

    # Merge bar-level promoted NewN / pattern signals (causal fire flags)
    sig = load_pattern_signal_matrix(symbol, timeframe)
    injected: list[str] = []
    if sig is not None and len(sig) and "timestamp" in sig.columns and "timestamp" in df.columns:
        left = df[["timestamp"]].copy()
        left["_i"] = np.arange(len(left))
        right = sig.copy()
        left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
        merged = left.merge(right, on="timestamp", how="left", suffixes=("", "_sig"))
        merged = merged.sort_values("_i")
        feature_cols = [c for c in right.columns if c != "timestamp"]
        for c in feature_cols:
            col_name = c if c.startswith(("New", "pat_", "cmp_", "disc_")) else f"pat_sig_{c}"
            # Prefer existing feature column; otherwise inject promoted signal
            if col_name not in df.columns:
                df[col_name] = merged[c].fillna(0).astype(float).to_numpy()
                injected.append(col_name)
            else:
                # Blend: keep feature engine value, OR-in discovery signal if missing fires
                base = df[col_name].fillna(0).astype(float).to_numpy()
                add = merged[c].fillna(0).astype(float).to_numpy()
                df[col_name] = np.maximum(base, add)
                injected.append(col_name)
        # Aggregate promoted fire count / bias score
        if injected:
            mat = df[injected].fillna(0).astype(float)
            df["pattern_promoted_fire_count"] = mat.sum(axis=1)
            # Signed score from recommended metadata bias when available
            bias_map = {
                str(x.get("pattern_key") or x.get("id") or ""): str(x.get("bias") or "neutral")
                for x in (recommended + approved_new + new_items)
            }
            score = np.zeros(len(df), dtype=float)
            for c in injected:
                b = bias_map.get(c, "neutral")
                w = 1.0 if b == "bullish" else (-1.0 if b == "bearish" else 0.25)
                score += df[c].fillna(0).astype(float).to_numpy() * w
            df["pattern_promoted_bias_score"] = score
    source_meta_injected = injected

    source_meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "features_json_path": str(path),
        "registry_json_path": str(DataStateRegistry().timeframe_path(timeframe)),
        "pattern_paths": {
            section: str(section_path(symbol, timeframe, section))
            for section in (
                "candlesticks",
                "structural",
                "compounds",
                "knowledge",
                "discovery_log",
                "new_patterns",
                "rankings",
                "relations",
                "validation_report",
            )
        },
        "row_count": int(len(df)),
        "registry": {
            "raw": raw_meta,
            "clean": clean_meta,
            "features": feat_meta,
        },
        "pattern_summary": {
            "knowledge_count": len(knowledge_items),
            "compound_count": len(compound_items),
            "candle_count": len(candle_items),
            "structural_count": len(structural_items),
            "discovery_count": len(discovery_items),
            "new_patterns_count": len(new_items),
            "new_approved_count": len(approved_new),
            "engine4_recommended_count": len(recommended),
            "injected_signal_columns": source_meta_injected,
            "avg_success_rate": _safe_mean(success_vals),
            "avg_confidence": _safe_mean(conf_vals),
            "avg_forward_return": _safe_mean(fwd_vals),
        },
    }

    e4 = load_engine_config().get("engine4_training", {}) or {}
    if bool(e4.get("cross_tf_features", True)):
        df, mtf_meta = enrich_with_higher_timeframes(df, symbol, timeframe)
        source_meta["cross_tf"] = mtf_meta
        source_meta["row_count"] = int(len(df))

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df, source_meta
