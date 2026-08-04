"""Composite reward / penalty for closed live trades.

Reward blends outcome + decision quality + RR + process compliance.
Classification is *outcome-primary*:
- winning trades are never labeled ``penalty``
- losing trades are never labeled ``reward``
Quality issues on winners reduce the reward magnitude and appear as lessons,
without flipping the UI kind to عقوبة.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _tanh(x: float) -> float:
    return math.tanh(x)


@dataclass
class RewardBreakdown:
    total: float
    is_reward: bool
    kind: str  # "reward" | "penalty" | "neutral"
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    process_score: float = 0.0
    realized_rr: float | None = None
    planned_rr: float | None = None
    pnl_norm: float = 0.0
    impact_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _planned_rr(record: dict[str, Any]) -> float | None:
    exit_meta = record.get("exit_meta") or {}
    if isinstance(exit_meta, dict):
        rr = exit_meta.get("reward_risk")
        if rr is not None:
            val = _f(rr, default=float("nan"))
            return val if math.isfinite(val) else None
        meta = exit_meta.get("meta") or {}
        if isinstance(meta, dict) and meta.get("min_rr") is not None:
            val = _f(meta.get("min_rr"), default=float("nan"))
            return val if math.isfinite(val) else None
    return None


def _has_valid_exit_price(record: dict[str, Any]) -> bool:
    raw = record.get("exit_price")
    if raw is None or raw == "":
        return False
    px = _f(raw, default=float("nan"))
    return math.isfinite(px) and px > 0.0


def _realized_rr(record: dict[str, Any]) -> float | None:
    """Price R-multiple. Returns None when exit price is missing/invalid.

    Previously ``exit_price=None`` became 0.0 and produced absurd RR values
    (thousands), flipping reward/penalty incorrectly.
    """
    if not _has_valid_exit_price(record):
        return None
    entry = _f(record.get("entry_price"))
    exit_px = _f(record.get("exit_price"))
    sl = _f(record.get("sl"))
    if entry <= 0 or sl <= 0:
        return None
    risk = abs(entry - sl)
    if risk <= 1e-12:
        return None
    side = str(record.get("side") or "").lower()
    pred = record.get("pred")
    if side not in {"buy", "sell"}:
        if pred is not None:
            side = "buy" if int(pred) > 0 else "sell" if int(pred) < 0 else ""
    if side == "buy":
        move = exit_px - entry
    elif side == "sell":
        move = entry - exit_px
    else:
        move = abs(exit_px - entry) * (1.0 if _f(record.get("net_profit")) >= 0 else -1.0)
    rr = move / risk
    # Guard against corrupted ticks / bad snapshots.
    if not math.isfinite(rr) or abs(rr) > 20.0:
        return None
    return rr


def _is_early_winner_close(record: dict[str, Any], net: float) -> bool:
    """Manual/desk close of a winner before full TP — partial RR is expected."""
    if net <= 0 and not bool(record.get("is_winner")):
        return False
    cr = str(record.get("close_reason") or "").lower()
    markers = (
        "atis winners",
        "atis winner",
        "winners",
        "manual",
        "atis close",
        "close winners",
    )
    return any(m in cr for m in markers)


def _atr_from_record(record: dict[str, Any]) -> float:
    exit_meta = record.get("exit_meta") or {}
    if isinstance(exit_meta, dict):
        meta = exit_meta.get("meta") or {}
        if isinstance(meta, dict) and meta.get("atr") is not None:
            return max(_f(meta.get("atr")), 1e-9)
    feats = record.get("feature_snapshot") or {}
    if isinstance(feats, dict) and feats.get("atr") is not None:
        return max(_f(feats.get("atr")), 1e-9)
    entry = _f(record.get("entry_price"))
    sl = _f(record.get("sl"))
    return max(abs(entry - sl), 1e-9)


def _parse_reason_flags(reason: str) -> dict[str, bool]:
    r = str(reason or "").lower()
    return {
        "soft_regime": "regime_soft" in r or "soft_regime" in r,
        "soft_meta": "soft_meta" in r or "e4_soft_meta" in r,
        "spread_filter_off": "spread_filter_off" in r,
        "independent_tf": "mode=independent" in r,
        "rr_tightened": "sl_tightened_for_rr" in r,
        "invalid_stops": "retcode=10016" in r or "invalid stops" in r,
    }


def _quality_score(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, list[str]]:
    """0..1 quality of the *decision* at entry (independent of PnL luck)."""
    reasons: list[str] = []
    conf = _f(record.get("confidence"))
    conf_floor = _f(cfg.get("min_confidence_quality", 0.58), 0.58)
    conf_good = _f(cfg.get("high_confidence_quality", 0.72), 0.72)

    q = 0.35
    if conf >= conf_good:
        q += 0.35
        reasons.append("ثقة دخول عالية")
    elif conf >= conf_floor:
        q += 0.18
        reasons.append("ثقة دخول مقبولة")
    else:
        q -= 0.15
        reasons.append("ثقة دخول منخفضة")

    # Align with live dynamic_exits.min_rr (default 1.15), not an unreachable 1.5.
    planned = _planned_rr(record)
    min_rr = _f(cfg.get("min_planned_rr", 1.15), 1.15)
    if planned is not None and math.isfinite(planned):
        if planned >= min_rr + 0.35:
            q += 0.2
            reasons.append(f"RR مخطط جيد ({planned:.2f})")
        elif planned >= min_rr - 1e-9:
            q += 0.08
            reasons.append(f"RR مخطط مقبول ({planned:.2f})")
        else:
            q -= 0.15
            reasons.append(f"RR مخطط دون الحد ({planned:.2f})")

    flags = _parse_reason_flags(str(record.get("reason") or ""))
    if flags["soft_regime"] or flags["soft_meta"]:
        q -= 0.12
        reasons.append("تجاوز بوابة لينة (نظام/ميتا)")
    if flags["spread_filter_off"]:
        q -= 0.05
        reasons.append("فلتر السبريد معطّل")
    if flags["invalid_stops"]:
        q -= 0.25
        reasons.append("Stops غير صالحة عند التنفيذ")

    patterns = list(record.get("pattern_keys") or [])
    if patterns:
        q += min(0.08, 0.02 * len(patterns))
        reasons.append(f"أنماط مرافقة: {len(patterns)}")

    return _clamp(q, 0.0, 1.0), reasons


def _process_score(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, list[str], float]:
    """Rule adherence score 0..1 and violation penalty magnitude."""
    reasons: list[str] = []
    flags = _parse_reason_flags(str(record.get("reason") or ""))
    score = 1.0
    violation = 0.0

    if flags["soft_regime"]:
        score -= 0.25
        violation += 0.35
        reasons.append("ملاحظة عملية: تجاوز فلتر النظام (regime soft)")
    if flags["soft_meta"]:
        score -= 0.2
        violation += 0.3
        reasons.append("ملاحظة عملية: تجاوز فلتر الميتا")
    if flags["independent_tf"]:
        mtf = record.get("multi_tf_context") or {}
        if isinstance(mtf, dict) and mtf.get("conflict"):
            score -= 0.35
            violation += 0.5
            reasons.append("ملاحظة عملية: تعارض اتجاهات متعدد الأطر")
        else:
            score -= 0.05
    if flags["spread_filter_off"]:
        # Config-level setting — mild process ding, not a hard "عقوبة" label driver.
        score -= 0.08
        violation += 0.08
        reasons.append("ملاحظة: فلتر السبريد معطّل في الإعدادات")
    if flags["invalid_stops"]:
        score -= 0.4
        violation += 0.6
        reasons.append("ملاحظة عملية: فشل جودة التنفيذ (stops)")

    close_reason = str(record.get("close_reason") or "").lower()
    if "manual" in close_reason or "atis close" in close_reason or "winners" in close_reason:
        reasons.append("إغلاق يدوي/API — لا عقوبة نتيجة إضافية")

    min_conf = _f(cfg.get("min_confidence_quality", 0.58), 0.58)
    if _f(record.get("confidence")) < min_conf:
        score -= 0.15
        violation += 0.2
        reasons.append("ملاحظة عملية: دخول دون عتبة الثقة")

    return _clamp(score, 0.0, 1.0), reasons, max(0.0, violation)


def _rr_term(
    record: dict[str, Any],
    *,
    net: float,
    planned: float | None,
    realized: float | None,
) -> tuple[float, list[str]]:
    """RR contribution. Early winner closes are not punished for partial RR."""
    notes: list[str] = []
    early = _is_early_winner_close(record, net)

    if early and net > 0:
        # Desk took profit early — credit a small positive RR term from PnL sign.
        if realized is not None and math.isfinite(realized) and realized > 0:
            term = _tanh(max(0.0, realized) * 0.85)
            notes.append(f"إغلاق رابح مبكر · RR محقّق جزئي ({realized:.2f})")
        else:
            term = 0.25
            notes.append("إغلاق رابح مبكر — لا معاقبة على عدم بلوغ RR الكامل")
        return term, notes

    if planned is not None and realized is not None and math.isfinite(planned) and math.isfinite(realized):
        # For winners, never let partial-but-positive RR dominate into a penalty.
        if net > 0 and realized >= 0:
            term = _tanh(realized - min(planned, max(realized, 0.0)))
            # Soft: compare to planned only as bonus when beating it.
            if realized >= planned:
                term = _tanh(realized - planned * 0.5)
            else:
                term = _tanh(realized * 0.6)  # still non-negative-ish for small wins
            return term, notes
        return _tanh(realized - planned), notes

    if realized is not None and math.isfinite(realized):
        return _tanh(realized - 1.0), notes
    return 0.0, notes


def score_closed_trade(
    record: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> RewardBreakdown:
    """Compute signed reward for one closed trade episode."""
    cfg = dict(cfg or {})
    w_pnl = _f(cfg.get("w_pnl", 0.40), 0.40)
    w_quality = _f(cfg.get("w_quality", 0.22), 0.22)
    w_rr = _f(cfg.get("w_rr", 0.18), 0.18)
    w_process = _f(cfg.get("w_process", 0.12), 0.12)
    w_violation = _f(cfg.get("w_violation", 0.25), 0.25)
    w_noise = _f(cfg.get("w_noise", 0.12), 0.12)
    abstain_bonus = _f(cfg.get("abstain_bonus", 0.0), 0.0)

    net = _f(record.get("net_profit"))
    # Strict outcome from realized PnL sign — any profit (even tiny) is a win.
    is_winner = net > 0
    is_loser = net < 0

    atr = _atr_from_record(record)
    vol = max(_f(record.get("volume"), 0.01), 0.01)
    pnl_norm = net / max(atr * vol * 100.0, 1e-6)
    # Floor a clear positive outcome so tiny winners still get a PnL signal.
    if is_winner and pnl_norm < 0.15:
        pnl_norm = max(pnl_norm, 0.35)
    if is_loser and pnl_norm > -0.15:
        pnl_norm = min(pnl_norm, -0.35)
    pnl_term = _tanh(pnl_norm / _f(cfg.get("pnl_norm_scale", 2.0), 2.0))

    quality, q_reasons = _quality_score(record, cfg)
    process, p_reasons, violation = _process_score(record, cfg)

    planned = _planned_rr(record)
    realized = _realized_rr(record)
    rr_term, rr_notes = _rr_term(record, net=net, planned=planned, realized=realized)

    flags = _parse_reason_flags(str(record.get("reason") or ""))
    noise = 0.0
    noise_reasons: list[str] = []
    if flags["spread_filter_off"] and is_loser:
        noise += 0.35
        noise_reasons.append("صفقة ضوضاء محتملة (سبريد غير مفلتر + خسارة)")
    if flags["soft_regime"] and is_loser:
        noise += 0.3
        noise_reasons.append("خسارة بعد تجاوز نظام السوق")

    quality_term = (quality - 0.5) * 2.0
    process_term = (process - 0.5) * 2.0

    # On winners, quality/process issues only *reduce* reward — they cannot dominate.
    if is_winner:
        quality_term = max(quality_term, -0.35)
        process_term = max(process_term, -0.25)
        violation = min(violation, 0.2)
        rr_term = max(rr_term, 0.0)

    total = (
        w_pnl * pnl_term
        + w_quality * quality_term
        + w_rr * rr_term
        + w_process * process_term
        - w_violation * violation
        - w_noise * noise
        + abstain_bonus
    )
    total = _clamp(total, -1.5, 1.5)

    reasons = [*q_reasons, *p_reasons, *rr_notes, *noise_reasons]
    if is_winner:
        reasons.insert(0, f"صفقة رابحة · صافي {net:.2f}")
    elif is_loser:
        reasons.insert(0, f"صفقة خاسرة · صافي {net:.2f}")
    else:
        reasons.insert(0, "صفقة متعادلة")

    lessons: list[str] = []
    if is_winner and quality >= 0.6:
        lessons.append("تعزيز سلوكيات الدخول الرابحة")
    if is_winner and rr_term > 0.15:
        lessons.append("تعزيز تحقيق عائد/مخاطرة إيجابي")
    if is_winner and quality < 0.45:
        lessons.append("ربح مع جودة قرار منخفضة — مكافأة مخفّضة (لا عقوبة)")
    if is_winner and planned is not None and planned < _f(cfg.get("min_planned_rr", 1.15), 1.15):
        lessons.append("حسّن RR المخطط لاحقاً — الصفقة رابحة وستُستخدم للتدريب بمكافأة مخفّضة")
    if is_winner and _is_early_winner_close(record, net):
        lessons.append("إغلاق ربح مبكر — النتيجة رابحة؛ حسّن RR المخطط لاحقاً")
    if is_loser and flags["soft_meta"]:
        lessons.append("تجنب تجاوز فلتر الميتا في ظروف مشابهة")
    if is_loser and flags["soft_regime"]:
        lessons.append("تجنب التداول خارج نطاق النظام (ATR regime)")
    if is_loser and planned is not None and planned < _f(cfg.get("min_planned_rr", 1.15), 1.15):
        lessons.append("رفض الإعدادات ذات RR المخطط الضعيف")
    if is_loser and _f(record.get("confidence")) < _f(cfg.get("min_confidence_quality", 0.58), 0.58):
        lessons.append("رفع عتبة الثقة قبل الدخول")
    if is_loser and quality >= 0.7:
        # Keep lesson, but do not soft-cap the penalty — losses must move policy.
        soft_cap = bool(cfg.get("soft_cap_good_quality_loss", False))
        if soft_cap:
            lessons.append("خسارة رغم جودة قرار جيدة — عقوبة مخفّفة (تباين السوق)")
            total = max(total, -0.35)
        else:
            lessons.append("خسارة رغم جودة قرار جيدة — عقوبة كاملة (لا تخفيف)")
    if not lessons:
        lessons.append("تحديث خفيف لأوزان السياسة وفق المكافأة المركّبة")

    # Strict PnL classification: +profit → مكافأة/رابحة, -profit → عقوبة/خاسرة.
    if is_winner:
        kind = "reward"
        is_reward = True
        if total <= 0.05:
            total = max(0.08, 0.12 + 0.25 * max(0.0, quality - 0.35))
        if quality < 0.45:
            impact = "صفقة رابحة (حتى لو الربح صغير) — مكافأة مع دروس لتحسين جودة الدخول"
        else:
            impact = "صفقة رابحة — زيادة وزن سلوكيات الدخول المشابهة"
    elif is_loser:
        kind = "penalty"
        is_reward = False
        if total >= -0.05:
            total = min(-0.08, -0.12 - 0.2 * max(0.0, 0.55 - quality))
        impact = "صفقة خاسرة — تقليل احتمال تكرار نفس سلوك الدخول"
    else:
        # Exact zero only — keep neutral; callers should refresh PnL from broker when possible.
        kind = "neutral"
        is_reward = False
        total = 0.0
        impact = "صافي ربح صفر بالضبط — بانتظار تأكيد الوسيط إن وُجد"
        lessons = ["صافي صفر: أعد جلب نتيجة الصفقة من الوسيط قبل التعلم"]

    return RewardBreakdown(
        total=round(total, 6),
        is_reward=is_reward,
        kind=kind,
        components={
            "pnl_term": round(pnl_term, 6),
            "quality_term": round(quality_term, 6),
            "rr_term": round(rr_term, 6),
            "process_term": round(process_term, 6),
            "violation": round(violation, 6),
            "noise": round(noise, 6),
        },
        reasons=reasons,
        lessons=lessons,
        quality_score=round(quality, 4),
        process_score=round(process, 4),
        realized_rr=None if realized is None or not math.isfinite(realized) else round(realized, 4),
        planned_rr=None if planned is None or not math.isfinite(planned) else round(planned, 4),
        pnl_norm=round(pnl_norm, 6),
        impact_hint=impact,
    )
