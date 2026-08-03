"""Real-time MT5 pattern overlay (Explainable AI chart drawings)."""

from atis.shared.mt5_pattern_overlay.bridge import AsyncOverlayBridge, OverlaySnapshot
from atis.shared.mt5_pattern_overlay.detector import extract_active_overlays, top_signal_patterns
from atis.shared.mt5_pattern_overlay.history import append_history, read_history
from atis.shared.mt5_pattern_overlay.models import (
    LEGEND_ENTRIES,
    PatternOverlay,
    PatternStatus,
)
from atis.shared.mt5_pattern_overlay.service import PatternOverlayService, get_overlay_service

__all__ = [
    "AsyncOverlayBridge",
    "LEGEND_ENTRIES",
    "OverlaySnapshot",
    "PatternOverlay",
    "PatternOverlayService",
    "PatternStatus",
    "append_history",
    "extract_active_overlays",
    "get_overlay_service",
    "read_history",
    "top_signal_patterns",
]
