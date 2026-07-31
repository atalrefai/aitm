"""Advanced deep-learning training pipeline for the unified LLModel artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from atis.config import get_path, load_engine_config, load_timeframes
from atis.engines.engine4_training.data_sources import load_training_frame
from atis.shared.pattern_kb import PatternKnowledgeBase

try:  # pragma: no cover - dependency availability is environment-specific
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    HAS_TORCH = False


META_COLUMNS = {
    "timestamp",
    "symbol",
    "timeframe",
    "label",
    "label_meta",
}
PRICE_COLUMNS = {"open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
SESSION_MAP = {"asia": 0.0, "europe": 1.0, "us": 2.0}
VOL_MAP = {"unknown": 0.0, "calm": 1.0, "normal": 2.0, "violent": 3.0}


@dataclass
class PreparedSequenceData:
    symbol: str
    base_timeframe: str
    timeframes: list[str]
    sequence_length: int
    per_timeframe_features: dict[str, list[str]]
    context_features: list[str]
    timestamps: list[str]
    inputs: dict[str, np.ndarray]
    context: np.ndarray
    labels: np.ndarray
    future_returns: np.ndarray
    close: np.ndarray
    feature_index: dict[str, dict[str, int]]
    split_indices: dict[str, tuple[int, int]]


@dataclass
class LLModelArtifact:
    artifact_path: str
    metadata_path: str
    metrics_path: str
    report: dict[str, Any]


def _cfg() -> dict[str, Any]:
    base = load_engine_config().get("engine4_training", {})
    deep = base.get("deep_learning", {})
    return {"base": base, "deep": deep}


def _tf_minutes(tf: str) -> int:
    return int(load_timeframes()[tf]["minutes"])


def _tf_order(timeframes: list[str]) -> list[str]:
    return sorted(timeframes, key=_tf_minutes)


def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "session" in out.columns:
        out["session"] = out["session"].map(lambda x: SESSION_MAP.get(str(x), 0.0)).astype(float)
    if "vol_regime" in out.columns:
        out["vol_regime"] = out["vol_regime"].map(lambda x: VOL_MAP.get(str(x), 0.0)).astype(float)
    return out


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def _kb_context_for(symbol: str, timeframe: str) -> dict[str, float]:
    kb_rows = PatternKnowledgeBase().list_stats(symbol, timeframe, min_occurrences=1, limit=5000)
    if not kb_rows:
        return {
            "kb_occurrences_total": 0.0,
            "kb_success_rate_mean": 0.5,
            "kb_confidence_mean": 0.0,
            "kb_forward_return_mean": 0.0,
            "kb_bullish_edge": 0.0,
            "kb_bearish_edge": 0.0,
            "kb_patterns_active": 0.0,
        }
    success_vals = [float(r["success_rate"]) for r in kb_rows if r.get("success_rate") is not None]
    conf_vals = [float(r["confidence"]) for r in kb_rows if r.get("confidence") is not None]
    fwd_vals = [float(r["avg_forward_return"]) for r in kb_rows if r.get("avg_forward_return") is not None]
    occ = sum(float(r.get("occurrences") or 0.0) for r in kb_rows)
    bull = sum(float(r.get("occurrences") or 0.0) for r in kb_rows if str(r.get("bias")) == "bullish")
    bear = sum(float(r.get("occurrences") or 0.0) for r in kb_rows if str(r.get("bias")) == "bearish")
    return {
        "kb_occurrences_total": occ,
        "kb_success_rate_mean": float(np.mean(success_vals)) if success_vals else 0.5,
        "kb_confidence_mean": float(np.mean(conf_vals)) if conf_vals else 0.0,
        "kb_forward_return_mean": float(np.mean(fwd_vals)) if fwd_vals else 0.0,
        "kb_bullish_edge": bull / max(occ, 1.0),
        "kb_bearish_edge": bear / max(occ, 1.0),
        "kb_patterns_active": float(len(kb_rows)),
    }


def _frame_for_timeframe(symbol: str, timeframe: str, max_rows: int | None = None) -> pd.DataFrame:
    frame, _source_meta = load_training_frame(symbol, timeframe)
    frame = _encode_categorical(frame)
    if max_rows and len(frame) > max_rows:
        frame = frame.tail(max_rows).reset_index(drop=True)
    kb_ctx = _kb_context_for(symbol, timeframe)
    for key, value in kb_ctx.items():
        frame[key] = value
    return frame


def _prefixed_frame(frame: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, list[str]]:
    numeric = _numeric_feature_columns(frame)
    rename = {col: f"{timeframe}__{col}" for col in numeric}
    out = frame[["timestamp", *numeric]].rename(columns=rename)
    return out, list(rename.values())


def _label_from_return(forward_ret: float, hold_thr: float) -> int:
    if forward_ret > hold_thr:
        return 2
    if forward_ret < -hold_thr:
        return 0
    return 1


def prepare_multitimeframe_sequences(
    symbol: str,
    timeframes: list[str],
) -> PreparedSequenceData:
    cfg = _cfg()
    deep = cfg["deep"]
    base = cfg["base"]
    ordered = _tf_order(timeframes)
    base_tf = str(deep.get("base_timeframe") or ordered[0])
    if base_tf not in ordered:
        base_tf = ordered[0]
    seq_len = int(deep.get("sequence_length", 48))
    max_rows = int(deep.get("max_rows_per_timeframe", 12000))
    horizon = int(base.get("horizon_bars", 5))
    hold_thr = float(deep.get("hold_threshold", 0.0015))
    context_names = list(deep.get("context_features") or [])

    base_frame = _frame_for_timeframe(symbol, base_tf, max_rows=max_rows)
    merged = base_frame[["timestamp", "close"]].copy()
    per_tf_features: dict[str, list[str]] = {}

    for tf in ordered:
        tf_frame = _frame_for_timeframe(symbol, tf, max_rows=max_rows)
        prefixed, cols = _prefixed_frame(tf_frame, tf)
        per_tf_features[tf] = cols
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            prefixed.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    context_cols: list[str] = []
    for tf in ordered:
        for name in context_names:
            candidate = f"{tf}__{name}"
            if candidate in merged.columns and candidate not in context_cols:
                context_cols.append(candidate)
        for suffix in (
            "kb_occurrences_total",
            "kb_success_rate_mean",
            "kb_confidence_mean",
            "kb_forward_return_mean",
            "kb_bullish_edge",
            "kb_bearish_edge",
            "kb_patterns_active",
        ):
            candidate = f"{tf}__{suffix}"
            if candidate in merged.columns and candidate not in context_cols:
                context_cols.append(candidate)

    merged["future_return"] = merged["close"].shift(-horizon) / merged["close"] - 1.0
    merged["target"] = merged["future_return"].map(lambda x: _label_from_return(float(x), hold_thr) if pd.notna(x) else np.nan)

    required = ["future_return", "target"]
    for cols in per_tf_features.values():
        required.extend(cols)
    required.extend(context_cols)
    required = list(dict.fromkeys(required))

    merged = merged.dropna(subset=required).reset_index(drop=True)
    if len(merged) <= seq_len + 5:
        raise ValueError(f"insufficient_rows:{len(merged)}")

    feature_index = {tf: {name: i for i, name in enumerate(cols)} for tf, cols in per_tf_features.items()}
    inputs = {tf: [] for tf in ordered}
    context: list[np.ndarray] = []
    labels: list[int] = []
    future_returns: list[float] = []
    close_vals: list[float] = []
    timestamps: list[str] = []

    for end_idx in range(seq_len - 1, len(merged)):
        window = merged.iloc[end_idx - seq_len + 1 : end_idx + 1]
        for tf in ordered:
            arr = window[per_tf_features[tf]].to_numpy(dtype=np.float32)
            inputs[tf].append(arr)
        context.append(merged.iloc[end_idx][context_cols].to_numpy(dtype=np.float32))
        labels.append(int(merged.iloc[end_idx]["target"]))
        future_returns.append(float(merged.iloc[end_idx]["future_return"]))
        close_vals.append(float(merged.iloc[end_idx]["close"]))
        timestamps.append(str(merged.iloc[end_idx]["timestamp"]))

    arrays = {tf: np.stack(items).astype(np.float32) for tf, items in inputs.items()}
    ctx = np.stack(context).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    rets = np.asarray(future_returns, dtype=np.float32)
    close = np.asarray(close_vals, dtype=np.float32)

    total = len(y)
    train_end = max(seq_len + 1, int(total * float(deep.get("train_ratio", 0.7))))
    val_end = max(train_end + 1, int(total * (float(deep.get("train_ratio", 0.7)) + float(deep.get("val_ratio", 0.15)))))
    val_end = min(val_end, total - 1)

    return PreparedSequenceData(
        symbol=symbol,
        base_timeframe=base_tf,
        timeframes=ordered,
        sequence_length=seq_len,
        per_timeframe_features=per_tf_features,
        context_features=context_cols,
        timestamps=timestamps,
        inputs=arrays,
        context=ctx,
        labels=y,
        future_returns=rets,
        close=close,
        feature_index=feature_index,
        split_indices={
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total),
        },
    )


class MultiTimeframeDataset(Dataset):
    def __init__(self, prepared: PreparedSequenceData, start: int, end: int) -> None:
        self.prepared = prepared
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pos = self.start + idx
        item = {
            "context": self.prepared.context[pos],
            "label": self.prepared.labels[pos],
            "future_return": self.prepared.future_returns[pos],
        }
        for tf in self.prepared.timeframes:
            item[f"seq_{tf}"] = self.prepared.inputs[tf][pos]
        return item


class TimeframeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        _, h = self.gru(z)
        return h[-1]


class LLTradingModel(nn.Module):
    def __init__(
        self,
        timeframe_dims: dict[str, int],
        context_dim: int,
        hidden_dim: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.timeframes = list(timeframe_dims.keys())
        self.encoders = nn.ModuleDict(
            {tf: TimeframeEncoder(dim, hidden_dim, dropout) for tf, dim in timeframe_dims.items()}
        )
        self.timeframe_embeddings = nn.Parameter(torch.randn(len(self.timeframes), hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: dict[str, torch.Tensor], context: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = []
        for i, tf in enumerate(self.timeframes):
            tok = self.encoders[tf](sequences[tf]) + self.timeframe_embeddings[i]
            tokens.append(tok)
        token_tensor = torch.stack(tokens, dim=1)
        q = self.query.expand(token_tensor.size(0), -1, -1)
        attn_out, attn_weights = self.attn(q, token_tensor, token_tensor, need_weights=True)
        fused_tf = attn_out.squeeze(1)
        fused_ctx = self.context_net(context)
        fused = self.head(torch.cat([fused_tf, fused_ctx], dim=1))
        logits = self.classifier(fused)
        expected_return = self.return_head(fused).squeeze(1)
        risk = torch.sigmoid(self.risk_head(fused).squeeze(1))
        return {
            "logits": logits,
            "expected_return": expected_return,
            "risk": risk,
            "attention": attn_weights.squeeze(1),
        }


def _batch_to_device(batch: dict[str, Any], timeframes: list[str], device: Any) -> tuple[dict[str, Any], Any, Any, Any]:
    seqs = {tf: torch.as_tensor(batch[f"seq_{tf}"], device=device, dtype=torch.float32) for tf in timeframes}
    ctx = torch.as_tensor(batch["context"], device=device, dtype=torch.float32)
    labels = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
    future_return = torch.as_tensor(batch["future_return"], device=device, dtype=torch.float32)
    return seqs, ctx, labels, future_return


def _returns_from_pred_classes(
    classes: np.ndarray,
    future_returns: np.ndarray,
    *,
    probs: np.ndarray | None = None,
    decision_threshold: float = 0.58,
) -> np.ndarray:
    """Map class ids {0,1,2} -> {-1,0,+1} with confidence filter."""
    mapped = np.where(classes == 2, 1.0, np.where(classes == 0, -1.0, 0.0))
    if probs is not None and len(probs):
        conf = probs.max(axis=1)
        mapped = np.where(conf >= decision_threshold, mapped, 0.0)
    return mapped * future_returns


def _financial_metrics(returns: np.ndarray, periods_per_year: float = 252.0) -> dict[str, float]:
    r = returns[np.isfinite(returns)]
    traded = r[r != 0]
    if len(r) == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_return": 0.0, "n_trades": 0.0}
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-9) - 1.0
    base = traded if len(traded) >= 10 else r
    mu = float(np.mean(base)) if len(base) else 0.0
    sigma = float(np.std(base)) if len(base) else 0.0
    if len(traded) >= 10 and len(r) > 0:
        ann = np.sqrt(max(periods_per_year * (len(traded) / max(len(r), 1)), 1.0))
    else:
        ann = np.sqrt(max(periods_per_year, 1.0))
    return {
        "sharpe": float(mu / sigma * ann) if sigma > 0 else 0.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "win_rate": float((traded > 0).sum() / max(len(traded), 1)),
        "total_return": float(equity[-1] - 1.0),
        "n_trades": float(len(traded)),
    }


def train_llmodel(
    symbol: str,
    timeframes: list[str],
    *,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> LLModelArtifact:
    if not HAS_TORCH:
        raise RuntimeError("torch_not_installed")

    cfg = _cfg()
    deep = cfg["deep"]
    if progress:
        progress(2.0, "تحميل ملفات JSON وتجهيز التسلسلات متعددة الأطر")
    prepared = prepare_multitimeframe_sequences(symbol, timeframes)
    train_start, train_end = prepared.split_indices["train"]
    val_start, val_end = prepared.split_indices["val"]
    test_start, test_end = prepared.split_indices["test"]

    train_ds = MultiTimeframeDataset(prepared, train_start, train_end)
    val_ds = MultiTimeframeDataset(prepared, val_start, val_end)
    test_ds = MultiTimeframeDataset(prepared, test_start, test_end)

    batch_size = int(deep.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLTradingModel(
        {tf: len(prepared.per_timeframe_features[tf]) for tf in prepared.timeframes},
        len(prepared.context_features),
        int(deep.get("hidden_dim", 96)),
        int(deep.get("attention_heads", 4)),
        float(deep.get("dropout", 0.15)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep.get("learning_rate", 1e-3)),
        weight_decay=float(deep.get("weight_decay", 1e-4)),
    )
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    epochs = int(deep.get("epochs", 8))
    if log:
        log(
            f"[LLModel] base_tf={prepared.base_timeframe} "
            f"timeframes={','.join(prepared.timeframes)} rows={len(prepared.labels)}"
        )

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            seqs, ctx, labels, future_return = _batch_to_device(batch, prepared.timeframes, device)
            optimizer.zero_grad()
            out = model(seqs, ctx)
            loss = ce_loss(out["logits"], labels)
            loss = loss + 0.25 * mse_loss(out["expected_return"], future_return)
            loss = loss + 0.05 * mse_loss(out["risk"], future_return.abs().clamp(0.0, 1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                seqs, ctx, labels, future_return = _batch_to_device(batch, prepared.timeframes, device)
                out = model(seqs, ctx)
                loss = ce_loss(out["logits"], labels)
                loss = loss + 0.25 * mse_loss(out["expected_return"], future_return)
                loss = loss + 0.05 * mse_loss(out["risk"], future_return.abs().clamp(0.0, 1.0))
                val_losses.append(float(loss.detach().cpu().item()))

        epoch_train = float(np.mean(train_losses)) if train_losses else 0.0
        epoch_val = float(np.mean(val_losses)) if val_losses else epoch_train
        history.append({"epoch": float(epoch + 1), "train_loss": epoch_train, "val_loss": epoch_val})
        if progress:
            progress(10.0 + (60.0 * (epoch + 1) / max(1, epochs)), f"Epoch {epoch + 1}/{epochs}")
        if log:
            log(f"[LLModel] epoch={epoch + 1}/{epochs} train_loss={epoch_train:.6f} val_loss={epoch_val:.6f}")
        if epoch_val <= best_val:
            best_val = epoch_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _predict(loader: Any) -> dict[str, Any]:
        logits_all = []
        rets_all = []
        labels_all = []
        risk_all = []
        attn_all = []
        with torch.no_grad():
            for batch in loader:
                seqs, ctx, labels, future_return = _batch_to_device(batch, prepared.timeframes, device)
                out = model(seqs, ctx)
                logits_all.append(out["logits"].detach().cpu().numpy())
                rets_all.append(future_return.detach().cpu().numpy())
                labels_all.append(labels.detach().cpu().numpy())
                risk_all.append(out["risk"].detach().cpu().numpy())
                attn_all.append(out["attention"].detach().cpu().numpy())
        logits = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, 3), dtype=np.float32)
        future_ret = np.concatenate(rets_all, axis=0) if rets_all else np.empty((0,), dtype=np.float32)
        labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,), dtype=np.int64)
        risk = np.concatenate(risk_all, axis=0) if risk_all else np.empty((0,), dtype=np.float32)
        attn = np.concatenate(attn_all, axis=0) if attn_all else np.empty((0, len(prepared.timeframes)), dtype=np.float32)
        probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy() if len(logits) else np.empty((0, 3))
        pred = probs.argmax(axis=1) if len(probs) else np.empty((0,), dtype=np.int64)
        thr = float(deep.get("decision_threshold", 0.62))
        edge = float(deep.get("directional_edge", 0.15))
        if len(probs):
            # classes: 0=down, 1=flat, 2=up
            conf = probs.max(axis=1)
            p_dn = probs[:, 0]
            p_up = probs[:, 2]
            keep_up = (pred == 2) & (conf >= thr) & ((p_up - p_dn) >= edge)
            keep_dn = (pred == 0) & (conf >= thr) & ((p_dn - p_up) >= edge)
            pred = np.where(keep_up, 2, np.where(keep_dn, 0, 1))
        acc = float((pred == labels_np).mean()) if len(labels_np) else 0.0
        fin = _financial_metrics(
            _returns_from_pred_classes(pred, future_ret, probs=probs if len(probs) else None, decision_threshold=thr),
            periods_per_year=252.0 * 24.0,
        )
        return {
            "accuracy": acc,
            "financial": fin,
            "predictions": pred.tolist(),
            "probabilities": probs.tolist(),
            "labels": labels_np.tolist(),
            "future_returns": future_ret.tolist(),
            "risk": risk.tolist(),
            "attention": attn.tolist(),
        }

    if progress:
        progress(78.0, "تشغيل Validation")
    val_metrics = _predict(val_loader)
    if progress:
        progress(88.0, "تشغيل Testing")
    test_metrics = _predict(test_loader)

    artifact_name = str(deep.get("artifact_name", "LLModel"))
    model_dir = get_path("models")
    artifact_path = model_dir / artifact_name
    metrics_path = model_dir / f"{artifact_name}.metrics.json"
    metadata_path = model_dir / f"{artifact_name}.meta.json"

    payload = {
        "artifact_type": "llmodel",
        "symbol": symbol,
        "base_timeframe": prepared.base_timeframe,
        "timeframes": prepared.timeframes,
        "sequence_length": prepared.sequence_length,
        "context_features": prepared.context_features,
        "per_timeframe_features": prepared.per_timeframe_features,
        "feature_index": prepared.feature_index,
        "state_dict": model.state_dict(),
        "model_config": {
            "hidden_dim": int(deep.get("hidden_dim", 96)),
            "attention_heads": int(deep.get("attention_heads", 4)),
            "dropout": float(deep.get("dropout", 0.15)),
            "decision_threshold": float(deep.get("decision_threshold", 0.55)),
            "hold_threshold": float(deep.get("hold_threshold", 0.45)),
        },
        "training_report": {
            "history": history,
            "validation": {k: v for k, v in val_metrics.items() if k not in {"predictions", "probabilities", "labels", "future_returns", "risk", "attention"}},
            "test": {k: v for k, v in test_metrics.items() if k not in {"predictions", "probabilities", "labels", "future_returns", "risk", "attention"}},
        },
    }
    torch.save(payload, artifact_path)

    top_attention = {}
    if test_metrics["attention"]:
        attn_arr = np.asarray(test_metrics["attention"], dtype=float)
        mean_attn = attn_arr.mean(axis=0)
        top_attention = {
            tf: float(mean_attn[i]) for i, tf in enumerate(prepared.timeframes)
        }

    summary = {
        "artifact": artifact_name,
        "artifact_path": str(artifact_path),
        "symbol": symbol,
        "base_timeframe": prepared.base_timeframe,
        "timeframes": prepared.timeframes,
        "rows": len(prepared.labels),
        "sequence_length": prepared.sequence_length,
        "context_features": len(prepared.context_features),
        "timeframe_feature_counts": {tf: len(cols) for tf, cols in prepared.per_timeframe_features.items()},
        "history": history,
        "validation": {
            "accuracy": val_metrics["accuracy"],
            "financial": val_metrics["financial"],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "financial": test_metrics["financial"],
            "attention_mean": top_attention,
        },
        "final_model_ready": True,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_type": "llmodel",
                "symbol": symbol,
                "timeframes": prepared.timeframes,
                "base_timeframe": prepared.base_timeframe,
                "artifact_path": str(artifact_path),
                "metrics_path": str(metrics_path),
                "final_model_ready": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if progress:
        progress(100.0, "اكتمل Training/Validation/Testing وإنشاء Final Model")
    if log:
        log(
            f"[LLModel] final_model={artifact_path} "
            f"val_acc={val_metrics['accuracy']:.4f} test_acc={test_metrics['accuracy']:.4f}"
        )
    return LLModelArtifact(
        artifact_path=str(artifact_path),
        metadata_path=str(metadata_path),
        metrics_path=str(metrics_path),
        report=summary,
    )


def _build_model_from_payload(payload: dict[str, Any], device: Any) -> Any:
    cfg = payload["model_config"]
    model = LLTradingModel(
        {tf: len(payload["per_timeframe_features"][tf]) for tf in payload["timeframes"]},
        len(payload["context_features"]),
        int(cfg["hidden_dim"]),
        int(cfg["attention_heads"]),
        float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_llmodel(path: Path | None = None) -> dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("torch_not_installed")
    cfg = _cfg()["deep"]
    artifact_name = str(cfg.get("artifact_name", "LLModel"))
    artifact_path = path or (get_path("models") / artifact_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(artifact_path, map_location=device)
    payload["_device"] = device
    payload["_model"] = _build_model_from_payload(payload, device)
    payload["_artifact_path"] = str(artifact_path)
    return payload


def _live_context_from_frames(payload: dict[str, Any], frames: dict[str, pd.DataFrame]) -> np.ndarray:
    values = []
    for col in payload["context_features"]:
        tf, raw = col.split("__", 1)
        frame = _encode_categorical(frames[tf])
        series = frame[raw] if raw in frame.columns else pd.Series([0.0])
        value = float(series.iloc[-1]) if not series.empty and pd.notna(series.iloc[-1]) else 0.0
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def predict_with_llmodel(payload: dict[str, Any], frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("torch_not_installed")
    device = payload["_device"]
    model = payload["_model"]
    seq_len = int(payload["sequence_length"])
    sequences = {}
    feature_rankings: list[dict[str, Any]] = []
    for tf in payload["timeframes"]:
        frame = _encode_categorical(frames[tf]).copy()
        cols = payload["per_timeframe_features"][tf]
        missing = [c for c in cols if c not in frame.columns]
        for col in missing:
            frame[col] = 0.0
        frame = frame.tail(seq_len)
        if len(frame) < seq_len:
            raise ValueError(f"insufficient_live_rows:{tf}:{len(frame)}")
        arr = frame[cols].to_numpy(dtype=np.float32)
        sequences[tf] = torch.as_tensor(arr[None, :, :], dtype=torch.float32, device=device)
        last_abs = np.abs(arr[-1])
        top_idx = np.argsort(last_abs)[-5:][::-1]
        feature_rankings.append(
            {
                "timeframe": tf,
                "top_features": [
                    {"feature": cols[i], "magnitude": float(last_abs[i])} for i in top_idx
                ],
            }
        )
    context = torch.as_tensor(_live_context_from_frames(payload, frames)[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(sequences, context)
        probs = torch.softmax(out["logits"], dim=1)[0].detach().cpu().numpy()
        attention = out["attention"][0].detach().cpu().numpy()
        expected_return = float(out["expected_return"][0].detach().cpu().item())
        risk = float(out["risk"][0].detach().cpu().item())
    pred_idx = int(np.argmax(probs))
    decision_threshold = float(payload["model_config"].get("decision_threshold", 0.55))
    side_map = {0: -1, 1: 0, 2: 1}
    side = side_map[pred_idx]
    confidence = float(probs[pred_idx])
    if side != 0 and confidence < decision_threshold:
        side = 0
    scenarios = {
        "sell": float(probs[0]),
        "hold": float(probs[1]),
        "buy": float(probs[2]),
    }
    return {
        "pred": side,
        "confidence": confidence,
        "expected_return": expected_return,
        "risk_score": risk,
        "scenario_probabilities": scenarios,
        "attention_by_timeframe": {
            tf: float(attention[i]) for i, tf in enumerate(payload["timeframes"])
        },
        "feature_rankings": feature_rankings,
        "artifact_path": payload["_artifact_path"],
    }
