"""Engine 4 — Research Factory (Training / Testing / Validation)

Pipeline version: `e4-v16.0-research-factory-20260731`

Walk-forward only (no random split). Triple-barrier labels. Realistic costs.
v16 implements the 10-point upgrade plan: financial HPO resolution, barrier sweep,
label cleaning, fold stability, crisis/recent holdouts, harsher stress, ensemble-on-conflict,
H4 quarantine, shadow challenger, research factory, and drift-aware retrain advisory.

```bash
python -m atis.engines.engine4_training.run --symbols XAUUSD --timeframes H1
```

## v16 lifecycle additions

1. Barrier sensitivity sweep (train-only label health)
2. Label-noise weight cleaning
3. Session hour features (when no session column)
4. Nested HP + Model Zoo conflict → ensemble / financial proxy winner
5. Fold stability gate (IQR / positive-fold fraction)
6. Expectancy vs execution-cost gate + raw trade Sharpe
7. Crisis + recent holdout slices
8. Harsher stress (latency extra, gap shock)
9. Shadow challenger registration when champion kept
10. Research factory board (`models/intelligence/research_factory.json`)
11. Retrain advisory with real PSI/decay (`retrain_advisory.json`)
12. Multi-TF confirm quarantines H4 by default

## Config highlights

- `barrier_sweep_enabled`, `label_cleaning_enabled`
- `fail_on_fold_unstable`, `fail_on_expectancy_below_cost`, `fail_on_crisis_holdout_weak`
- `prefer_ensemble_on_conflict`, `quarantine_h4_confirm`
- `promotion_validation_mode: cpcv_lite` (optional; set `validation_mode: cpcv_lite` to use)
- `stress_latency_extra`, `stress_gap_shock`

Champion: `models/{symbol}/{timeframe}/champion.json`
Shadow: `models/{symbol}/{timeframe}/shadow_challenger.json`
Research board: `models/intelligence/research_factory.json`
"""
