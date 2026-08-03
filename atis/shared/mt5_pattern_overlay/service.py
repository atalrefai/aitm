"""Orchestrates live pattern detection → async MT5 chart overlay."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from atis.shared.logging_utils import get_logger
from atis.shared.mt5_pattern_overlay.bridge import AsyncOverlayBridge, OverlaySnapshot
from atis.shared.mt5_pattern_overlay.detector import extract_active_overlays, top_signal_patterns
from atis.shared.mt5_pattern_overlay.history import append_history
from atis.shared.mt5_pattern_overlay.models import PatternOverlay, PatternStatus, TradeLink

logger = get_logger("atis.mt5_overlay.service")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PatternOverlayService:
    """
    Real-time Explainable-AI overlay for live trading.

    Call ``sync_from_features`` after each live feature computation.
    Call ``link_trade`` when a buy/sell is placed so the chart shows the reason.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        lookback_bars: int = 8,
        max_patterns: int = 40,
        bridge: AsyncOverlayBridge | None = None,
    ) -> None:
        self.enabled = enabled
        self.lookback_bars = lookback_bars
        self.max_patterns = max_patterns
        self._bridge = bridge or AsyncOverlayBridge()
        self._active: dict[str, PatternOverlay] = {}
        self._last_symbol = ""
        self._last_tf = ""
        self._last_broker_symbol = ""

    @property
    def bridge(self) -> AsyncOverlayBridge:
        return self._bridge

    def set_terminal_info_provider(self, provider) -> None:  # type: ignore[no-untyped-def]
        self._bridge._terminal_info_provider = provider

    def active_overlays(self) -> list[PatternOverlay]:
        return list(self._active.values())

    def sync_from_features(
        self,
        featured: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        broker_symbol: str | None = None,
        kb_stats: dict[str, dict[str, Any]] | None = None,
        publish: bool = True,
    ) -> list[PatternOverlay]:
        if not self.enabled:
            return []
        overlays = extract_active_overlays(
            featured,
            symbol=symbol,
            timeframe=timeframe,
            lookback_bars=self.lookback_bars,
            max_patterns=self.max_patterns,
            kb_stats=kb_stats,
            previous=self._active,
        )
        # Preserve trade links across sync
        for ov in overlays:
            prev = self._active.get(ov.id)
            if prev and prev.status == PatternStatus.LINKED:
                ov.status = PatternStatus.LINKED
                ov.trade = prev.trade
                # rebuild tooltip-bearing geometry already done in detector
        new_ids = {o.id for o in overlays}
        # Mark disappeared non-linked as invalidated for one publish cycle history
        for pid, prev in list(self._active.items()):
            if pid not in new_ids and prev.status == PatternStatus.LINKED:
                overlays.append(prev)

        self._active = {o.id: o for o in overlays}
        self._last_symbol = symbol
        self._last_tf = str(timeframe).upper()
        self._last_broker_symbol = broker_symbol or symbol

        # History: only newly confirmed on latest sync (dedupe by writing all confirmed)
        fresh = [
            o
            for o in overlays
            if o.status in {PatternStatus.CONFIRMED, PatternStatus.FORMING, PatternStatus.LINKED}
        ]
        if fresh:
            try:
                append_history(fresh, event="detect")
            except Exception as exc:
                logger.warning("pattern_history_append_failed", error=str(exc))

        if publish:
            self._publish(overlays)
        logger.info(
            "pattern_overlay_synced",
            symbol=symbol,
            timeframe=timeframe,
            count=len(overlays),
            confirmed=sum(1 for o in overlays if o.status == PatternStatus.CONFIRMED),
        )
        return overlays

    def link_trade(
        self,
        *,
        side: str,
        ticket: int | None,
        reason: str,
        pattern_ids: list[str] | None = None,
        limit: int = 3,
    ) -> list[PatternOverlay]:
        """Attach trade decision to the most relevant active patterns and republish."""
        if not self.enabled:
            return []
        side_l = side.lower().strip()
        targets: list[PatternOverlay]
        if pattern_ids:
            targets = [self._active[i] for i in pattern_ids if i in self._active]
        else:
            targets = top_signal_patterns(list(self._active.values()), side=side_l, limit=limit)

        linked_at = _utc_now_iso()
        for ov in targets:
            ov.status = PatternStatus.LINKED
            ov.trade = TradeLink(
                ticket=ticket,
                side=side_l,
                reason=reason[:240],
                linked_at=linked_at,
            )
            # Refresh tooltip / label objects text by rewriting geometry is heavy;
            # update object tooltips in-place.
            tip = ov.tooltip_text()
            label = ov.label_text()
            for obj in ov.objects:
                if obj.type in {"text", "arrow", "rectangle", "trendline"}:
                    obj.tooltip = tip
                if obj.type == "text" and "_lbl" in obj.name:
                    obj.text = label
                if obj.type == "text" and "_trade" in obj.name:
                    obj.text = (
                        f"→ {str(ov.trade.side or '').upper()} "
                        f"#{ov.trade.ticket or 'paper'}"
                    )
            self._active[ov.id] = ov

        if targets:
            try:
                append_history(targets, event="link_trade")
            except Exception as exc:
                logger.warning("pattern_history_link_failed", error=str(exc))
            self._publish(list(self._active.values()))
            logger.info(
                "pattern_trade_linked",
                side=side_l,
                ticket=ticket,
                patterns=[t.key for t in targets],
            )
        return targets

    def explain_decision(self, side: str, limit: int = 5) -> dict[str, Any]:
        picks = top_signal_patterns(list(self._active.values()), side=side, limit=limit)
        return {
            "side": side,
            "patterns": [
                {
                    "id": p.id,
                    "key": p.key,
                    "name": p.name,
                    "confidence": p.confidence,
                    "strength": p.strength,
                    "status": p.status.value,
                    "bias": p.bias,
                    "conditions": p.conditions,
                }
                for p in picks
            ],
            "summary": ", ".join(f"{p.name} ({p.confidence:.0%})" for p in picks) or "none",
        }

    def _publish(self, overlays: list[PatternOverlay]) -> None:
        snap = OverlaySnapshot(
            seq=self._bridge.next_seq(),
            symbol=self._last_symbol,
            broker_symbol=self._last_broker_symbol or self._last_symbol,
            timeframe=self._last_tf,
            patterns=overlays,
        )
        self._bridge.publish(snap)


# Process-wide service used by Engine 5 (lazy)
_SERVICE: PatternOverlayService | None = None


def get_overlay_service(cfg: dict[str, Any] | None = None) -> PatternOverlayService:
    global _SERVICE
    cfg = cfg or {}
    overlay_cfg = cfg.get("pattern_overlay") if isinstance(cfg.get("pattern_overlay"), dict) else cfg
    enabled = bool((overlay_cfg or {}).get("enabled", True))
    if _SERVICE is None:
        _SERVICE = PatternOverlayService(
            enabled=enabled,
            lookback_bars=int((overlay_cfg or {}).get("lookback_bars", 8)),
            max_patterns=int((overlay_cfg or {}).get("max_patterns", 40)),
        )
    else:
        _SERVICE.enabled = enabled
        if overlay_cfg:
            _SERVICE.lookback_bars = int(overlay_cfg.get("lookback_bars", _SERVICE.lookback_bars))
            _SERVICE.max_patterns = int(overlay_cfg.get("max_patterns", _SERVICE.max_patterns))
    return _SERVICE
