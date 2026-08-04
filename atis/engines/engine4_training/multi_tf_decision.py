"""Multi-timeframe decision layer — causal confirmation / veto for live & eval."""

from __future__ import annotations

from typing import Any


# Rank for HTF opposition checks (higher = slower / more authoritative).
_TF_RANK = {
    "M1": 1,
    "M5": 2,
    "M15": 3,
    "M30": 4,
    "H1": 5,
    "H4": 6,
    "D1": 7,
    "W1": 8,
    "MN1": 9,
}


def confirm_tfs_for_primary(
    primary_tf: str,
    cfg: dict[str, Any],
) -> list[str]:
    """Resolve confirmation TFs for a primary signal timeframe."""
    primary = str(primary_tf).upper()
    by_tf = cfg.get("confirm_by_primary_tf") or {}
    if primary in by_tf and by_tf[primary]:
        tfs = [str(t).upper() for t in by_tf[primary] if str(t).upper() != primary]
    else:
        tfs = [str(t).upper() for t in (cfg.get("confirm_timeframes") or []) if str(t).upper() != primary]
    # v16: quarantine weak/slow TFs from live confirmation by default
    quarantine = {str(t).upper() for t in (cfg.get("quarantine_confirm_tfs") or ["H4"])}
    if bool(cfg.get("quarantine_h4_confirm", True)):
        quarantine.add("H4")
    return [t for t in tfs if t not in quarantine]


def multi_tf_decision(
    primary_pred: int,
    primary_conf: float,
    confirmations: list[dict[str, Any]],
    *,
    mode: str = "soft_veto",
    min_confirm_agree: int = 1,
    veto_opposite_htf: bool = True,
    primary_tf: str | None = None,
    min_htf_conf: float = 0.55,
) -> tuple[int, dict[str, Any]]:
    """Combine primary TF signal with higher-TF confirmations.

    Modes
    -----
    soft_veto:
        Keep primary if ≥ min_confirm_agree TFs agree on side, OR no confirm
        TF produced a usable opposite signal. Strong HTF opposition still vetoes.
    hard_agree:
        Require ≥ min_confirm_agree same-side confirms; else flat.
    weighted:
        Score = primary + Σ sign(confirm)*conf; take sign of score.

    Returns (final_pred, debug_dict). Never peeks future — callers supply
    already-computed contemporaneous predictions only.
    """
    pred = int(primary_pred)
    dbg: dict[str, Any] = {
        "mode": mode,
        "primary_pred": pred,
        "primary_conf": float(primary_conf),
        "confirmations": confirmations,
        "agreed": 0,
        "opposed_htf": [],
        "reason": "pass",
    }
    if pred == 0:
        dbg["reason"] = "primary_flat"
        return 0, dbg

    usable = [c for c in confirmations if c.get("pred") is not None and "error" not in c]
    if not usable:
        dbg["reason"] = "no_confirm_data"
        # Soft: allow; hard: block.
        if str(mode).lower() == "hard_agree":
            return 0, {**dbg, "reason": "hard_agree_no_confirm"}
        return pred, dbg

    agreed = [c for c in usable if int(c.get("pred") or 0) == pred]
    opposed = [c for c in usable if int(c.get("pred") or 0) == -pred]
    dbg["agreed"] = len(agreed)

    primary_rank = _TF_RANK.get(str(primary_tf or "").upper(), 0)
    strong_htf_opp: list[dict[str, Any]] = []
    for c in opposed:
        ctf = str(c.get("tf") or "").upper()
        c_rank = _TF_RANK.get(ctf, 0)
        c_conf = float(c.get("conf") or 0.0)
        if c_rank > primary_rank and c_conf >= float(min_htf_conf):
            strong_htf_opp.append(c)
    dbg["opposed_htf"] = strong_htf_opp

    if veto_opposite_htf and strong_htf_opp:
        dbg["reason"] = "htf_opposite_veto"
        return 0, dbg

    mode_l = str(mode).lower()
    need = max(1, int(min_confirm_agree))

    if mode_l == "hard_agree":
        if len(agreed) >= need:
            dbg["reason"] = "hard_agree"
            return pred, dbg
        dbg["reason"] = "hard_agree_fail"
        return 0, dbg

    if mode_l == "weighted":
        score = float(pred) * max(float(primary_conf), 0.0)
        for c in usable:
            score += float(int(c.get("pred") or 0)) * max(float(c.get("conf") or 0.0), 0.0)
        dbg["weighted_score"] = score
        if score > 0 and pred > 0:
            dbg["reason"] = "weighted_long"
            return 1, dbg
        if score < 0 and pred < 0:
            dbg["reason"] = "weighted_short"
            return -1, dbg
        dbg["reason"] = "weighted_conflict"
        return 0, dbg

    # soft_veto (default): keep if any agree, or no opposing usable confirms.
    if len(agreed) >= need:
        dbg["reason"] = "soft_agree"
        return pred, dbg
    if not opposed:
        dbg["reason"] = "soft_no_opposition"
        return pred, dbg
    # Opposing same-rank/lower TFs without HTF veto → still allow if conf strong.
    if float(primary_conf) >= 0.70 and len(opposed) <= len(agreed):
        dbg["reason"] = "soft_strong_primary"
        return pred, dbg
    dbg["reason"] = "soft_veto"
    return 0, dbg


def fuse_multi_tf_votes(
    votes: list[dict[str, Any]],
    *,
    mode: str = "weighted_consensus",
    min_agree: int = 2,
    min_avg_conf: float = 0.52,
    veto_opposite_htf: bool = True,
    min_htf_conf: float = 0.55,
    tf_weights: dict[str, float] | None = None,
) -> tuple[int, float, dict[str, Any]]:
    """Merge independent per-TF model votes into one trade decision.

    Used for Multi-Timeframe Trading: each TF runs its *own* trained model on
    its own features, then votes are fused *before* any buy/sell.

    Modes
    -----
    weighted_consensus (default):
        Weighted score = Σ sign(pred) * conf * w(tf). Require |buy| or |sell|
        count ≥ min_agree among directional votes, and mean agreeing conf ≥ floor.
    majority:
        Side with more directional votes wins; ties → flat.
    hard_unanimous:
        All directional votes must agree; else flat.

    Returns (final_pred, fused_confidence, debug).
    """
    dbg: dict[str, Any] = {
        "mode": mode,
        "votes": votes,
        "buy_votes": 0,
        "sell_votes": 0,
        "flat_votes": 0,
        "reason": "pass",
        "execution_tf": None,
        "weighted_score": 0.0,
    }
    usable = [v for v in votes if v.get("pred") is not None and "error" not in v]
    if not usable:
        dbg["reason"] = "no_votes"
        return 0, 0.0, dbg

    weights = {str(k).upper(): float(w) for k, w in (tf_weights or {}).items()}
    buy: list[dict[str, Any]] = []
    sell: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    score = 0.0
    for v in usable:
        pred = int(v.get("pred") or 0)
        conf = max(float(v.get("conf") or 0.0), 0.0)
        tf = str(v.get("tf") or "").upper()
        w = float(weights.get(tf, 1.0 + 0.05 * _TF_RANK.get(tf, 0)))
        entry = {**v, "weight": w}
        if pred > 0:
            buy.append(entry)
            score += conf * w
        elif pred < 0:
            sell.append(entry)
            score -= conf * w
        else:
            flat.append(entry)
    dbg["buy_votes"] = len(buy)
    dbg["sell_votes"] = len(sell)
    dbg["flat_votes"] = len(flat)
    dbg["weighted_score"] = score

    need = max(1, int(min_agree))
    mode_l = str(mode).lower()

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _pick_side(side: int, agreeing: list[dict[str, Any]], opposed: list[dict[str, Any]]) -> tuple[int, float, dict[str, Any]]:
        if veto_opposite_htf and agreeing:
            max_agree_rank = max(_TF_RANK.get(str(a.get("tf") or "").upper(), 0) for a in agreeing)
            strong_opp = [
                o for o in opposed
                if _TF_RANK.get(str(o.get("tf") or "").upper(), 0) > max_agree_rank
                and float(o.get("conf") or 0.0) >= float(min_htf_conf)
            ]
            if strong_opp:
                dbg["opposed_htf"] = strong_opp
                dbg["reason"] = "htf_opposite_veto"
                return 0, 0.0, dbg
        avg_conf = float(_mean([float(a.get("conf") or 0.0) for a in agreeing])) if agreeing else 0.0
        if avg_conf < float(min_avg_conf):
            dbg["reason"] = "low_fused_confidence"
            dbg["fused_confidence"] = avg_conf
            return 0, avg_conf, dbg
        # Prefer highest-confidence agreeing TF for execution / ATR context.
        best = max(agreeing, key=lambda a: float(a.get("conf") or 0.0))
        dbg["execution_tf"] = str(best.get("tf") or "").upper() or None
        dbg["fused_confidence"] = avg_conf
        dbg["reason"] = "fused_agree"
        return side, avg_conf, dbg

    if mode_l == "hard_unanimous":
        directional = buy + sell
        if not directional:
            dbg["reason"] = "all_flat"
            return 0, 0.0, dbg
        if buy and not sell and len(buy) >= need:
            return _pick_side(1, buy, sell)
        if sell and not buy and len(sell) >= need:
            return _pick_side(-1, sell, buy)
        dbg["reason"] = "unanimous_fail"
        return 0, 0.0, dbg

    if mode_l == "majority":
        if len(buy) > len(sell) and len(buy) >= need:
            return _pick_side(1, buy, sell)
        if len(sell) > len(buy) and len(sell) >= need:
            return _pick_side(-1, sell, buy)
        dbg["reason"] = "majority_tie_or_short"
        return 0, 0.0, dbg

    # weighted_consensus (default): require min_agree directional votes.
    # Do NOT collapse to 1 when the opposite side is quiet — that made a lone
    # M15 vote open while other selected TFs were HOLD.
    if score > 0 and len(buy) >= need:
        return _pick_side(1, buy, sell)
    if score < 0 and len(sell) >= need:
        return _pick_side(-1, sell, buy)
    if abs(score) < 1e-12:
        dbg["reason"] = "weighted_flat"
    else:
        dbg["reason"] = "weighted_insufficient_agree"
    return 0, 0.0, dbg
