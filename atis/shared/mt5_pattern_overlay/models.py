"""Data models for real-time MT5 pattern overlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PatternStatus(str, Enum):
    FORMING = "forming"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    LINKED = "linked"


# ARGB / RGB hex without alpha (MT5 color as int built in EA)
CATEGORY_COLORS: dict[str, dict[str, str]] = {
    "candle": {"bullish": "26A69A", "bearish": "EF5350", "neutral": "90A4AE"},
    "chart": {"bullish": "29B6F6", "bearish": "FFA726", "neutral": "78909C"},
    "compound": {"bullish": "AB47BC", "bearish": "EC407A", "neutral": "7E57C2"},
    "structure": {"bullish": "FFD54F", "bearish": "FF8A65", "neutral": "B0BEC5"},
    "linked": {"bullish": "FFFFFF", "bearish": "FFFFFF", "neutral": "FFFFFF"},
    "invalidated": {"bullish": "616161", "bearish": "616161", "neutral": "616161"},
}

LEGEND_ENTRIES: list[dict[str, str]] = [
    {"category": "candle", "label": "Candlestick / Price Action", "color": "26A69A"},
    {"category": "chart", "label": "Chart / Market Structure", "color": "29B6F6"},
    {"category": "compound", "label": "Compound Patterns", "color": "AB47BC"},
    {"category": "structure", "label": "BOS / CHOCH / Liquidity", "color": "FFD54F"},
    {"category": "linked", "label": "Trade-Linked Pattern", "color": "FFFFFF"},
    {"category": "invalidated", "label": "Invalidated", "color": "616161"},
]

STRUCTURE_KEYS = frozenset(
    {
        "pat_bos_up",
        "pat_bos_down",
        "pat_choch_bull",
        "pat_choch_bear",
        "pat_equal_highs",
        "pat_equal_lows",
        "pat_liquidity_sweep_high",
        "pat_liquidity_sweep_low",
        "pat_breakout_up",
        "pat_breakout_down",
    }
)


@dataclass
class DrawObject:
    """One MT5 chart primitive."""

    type: str  # arrow | text | trendline | rectangle | hline | vline
    name: str
    color: str
    time: int | None = None
    price: float | None = None
    t1: int | None = None
    p1: float | None = None
    t2: int | None = None
    p2: float | None = None
    text: str | None = None
    tooltip: str | None = None
    width: int = 1
    style: int = 0  # STYLE_SOLID
    arrow_code: int | None = None
    fontsize: int = 8
    fill: bool = False
    back: bool = True
    selectable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TradeLink:
    ticket: int | None = None
    side: str | None = None
    reason: str | None = None
    linked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternOverlay:
    """A detected pattern ready to draw / explain on MT5."""

    id: str
    key: str
    name: str
    category: str
    bias: str
    status: PatternStatus
    timeframe: str
    symbol: str
    detected_at: str
    bar_time: int
    bar_index: int
    confidence: float
    strength: float
    conditions: str = ""
    expected_direction: str = "neutral"
    objects: list[DrawObject] = field(default_factory=list)
    anchors: list[dict[str, Any]] = field(default_factory=list)
    trade: TradeLink = field(default_factory=TradeLink)
    meta: dict[str, Any] = field(default_factory=dict)

    def color(self) -> str:
        if self.status == PatternStatus.INVALIDATED:
            return CATEGORY_COLORS["invalidated"]["neutral"]
        if self.status == PatternStatus.LINKED:
            return CATEGORY_COLORS["linked"]["neutral"]
        cat = "structure" if self.key in STRUCTURE_KEYS else self.category
        palette = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["candle"])
        return palette.get(self.bias, palette["neutral"])

    def tooltip_text(self) -> str:
        trade_line = ""
        if self.trade.ticket or self.trade.side:
            trade_line = (
                f"\nTrade: {self.trade.side or '-'} "
                f"#{self.trade.ticket or '-'} | {self.trade.reason or ''}"
            )
        return (
            f"{self.name} ({self.key})\n"
            f"Status: {self.status.value}\n"
            f"TF: {self.timeframe} | Bias: {self.bias}\n"
            f"Confidence: {self.confidence:.2f} | Strength: {self.strength:.2f}\n"
            f"Detected: {self.detected_at}\n"
            f"Direction: {self.expected_direction}\n"
            f"Conditions: {self.conditions or '-'}"
            f"{trade_line}"
        )

    def label_text(self) -> str:
        conf_pct = int(round(self.confidence * 100))
        tag = {
            PatternStatus.FORMING: "FORM",
            PatternStatus.CONFIRMED: "OK",
            PatternStatus.INVALIDATED: "X",
            PatternStatus.LINKED: "TRADE",
        }.get(self.status, self.status.value)
        return f"{self.name} [{tag} {conf_pct}%] {self.timeframe}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "name": self.name,
            "category": self.category,
            "bias": self.bias,
            "status": self.status.value,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "detected_at": self.detected_at,
            "bar_time": int(self.bar_time),
            "bar_index": int(self.bar_index),
            "confidence": float(self.confidence),
            "strength": float(self.strength),
            "conditions": self.conditions,
            "expected_direction": self.expected_direction,
            "color": self.color(),
            "label": self.label_text(),
            "tooltip": self.tooltip_text(),
            "anchors": self.anchors,
            "objects": [o.to_dict() for o in self.objects],
            "trade": self.trade.to_dict(),
            "meta": self.meta,
        }
