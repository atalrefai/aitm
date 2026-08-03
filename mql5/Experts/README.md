# ATIS Pattern Overlay (MT5)

Python MetaTrader5 API cannot create chart objects. ATIS writes overlay state to disk;
this Expert Advisor draws it on the chart in real time.

## Install

1. Copy `ATIS_PatternOverlay.mq5` into your terminal folder:
   - `{MT5 data_path}\MQL5\Experts\`
2. Compile in MetaEditor (F7).
3. Attach the EA to the XAUUSD chart (same timeframe you trade live).
4. Enable **Allow Algo Trading** and **Allow DLL imports** is not required.
5. Inputs:
   - `InpUseCommonFiles=true` (reads `Common\Files\ATIS\overlay_state.json`)
   - `InpPollMs=250`

## Data flow

```
Engine 5 / AutoTrader
  → detect patterns from live features
  → async write overlay_state.json
       ├─ project: logs/live/mt5_overlay/
       └─ MT5:    Common\Files\ATIS\  (and/or MQL5\Files\ATIS\)
  → EA polls file → ObjectCreate (arrows, text, trendlines, rectangles)
  → click object → Comment tooltip with full explainability panel
```

## Legend colors

| Category | Color |
|----------|-------|
| Candlestick | Teal |
| Chart / Structure | Cyan |
| Compound | Purple |
| BOS / CHOCH / Liquidity | Gold |
| Trade-linked | White |
| Invalidated | Gray |

## Notes

- Drawing is asynchronous and never blocks order execution.
- Pattern history (without heavy geometry) is stored in
  `logs/live/mt5_overlay/patterns_history.jsonl`.
- Trade comments become `ATIS|{TF}|{pattern}` when a pattern is linked.
