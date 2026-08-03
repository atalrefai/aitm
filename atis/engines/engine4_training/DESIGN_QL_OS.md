# ATIS Engine-4 — Quantitative Learning OS Design Brief (e4-v17.4 NA-QL Phase A)

## Goal

Turn Engine-4 from a training executor into an evidence-driven quant learning loop:
diagnose → hypothesize → patch → retest → promote only under honest time-series gates.

**v17.4 Phase A (Quality Decision Lock):** lock decision quality before expanding
neural heads — Consistency Score + quality compound objective + one falsifiable
hypothesis per run, without weakening promotion gates.

**v17.3 focus (shipped):** stop fill-to-cap trade policy (root of `trade_rate_saturated`
+ inflated Return), optimize for win_rate / expectancy / Acc-F1 quality, and never
label FinalModel `live_ready` when readiness says `paper_ready`.

## Closed loop (NA-QL)

```
Perceive → Decide → Act → Audit → Rewire → Retest
```

| Stage | Engine-4 mapping (incremental) |
|---|---|
| Perceive | Market state / regimes / cost / session · patterns as **context**, not sole driver |
| Decide | Entry policy + confidence/size + exit head (Phase B) · meta-label as quality gate |
| Act | `policy_from_proba` / live Engine-5 · dynamic exits bridge (Phase B) |
| Audit | `self_diagnostic` + Consistency Score + red-team holdouts |
| Rewire | One hypothesis → `suggested_config_diff` → research_factory / pending_overrides |
| Retest | Same WF + CPCV path · gates unchanged in strictness |

## Architecture (incremental layers)

```
┌─────────────────────────────────────────────────────────────┐
│ NA-QL L1 Perception Cortex (Phase B+)                       │
│    Market State · pattern→policy context (not sole driver)  │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ A) Market & Data Cognition (existing + harden later)        │
│    data_intelligence · feature_intelligence · label_quality │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Train / CPCV-lite WF (existing)                             │
│    validation_protocols · financial_hpo · model_zoo         │
│    + quality_compound_score (WR+F1+expectancy − sat/infl)   │
│    + regime_balanced_holdouts (Phase A)                     │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ NA-QL L2 Decision Spine (Phase B)                           │
│    Entry / Meta / Exit heads · pattern→policy by regime     │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ E) Anti-inflation + quality-first trade policy (v17.3)      │
│    train_confidence_floor headroom under hard cap           │
│    tune_trade_policy rewards win_rate + expectancy          │
│    financial_proxy anti-peg + win_rate + F1 term            │
│    promotion_v16.inflated_sharpe_report                     │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ B / L3 Critic & Audit                                       │
│    self_diagnostic → diagnosis.json                         │
│    Consistency Score (fold × regime WR/F1/Exp)              │
│    Unified recommenders → ONE hypothesis / run              │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ F) Readiness (hardened)                                     │
│    never live_ready / 100 with saturation·inflation·gaps    │
│    consistency soft/hard blend · FinalModel follows verdict │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ C / L4 Memory Fabric (partial → Phase C)                    │
│    research_factory ← diagnosis knobs                       │
│    enterprise propose_config_overrides ← diagnosis only     │
│    knowledge_loop episodes → apply_pending_overrides        │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ D / L5 Consistency Forge (NEXT)                             │
│    promotion_validation_mode: cpcv (full) on promo path     │
│    hard consistency gate for live_ready (Phase C)           │
└─────────────────────────────────────────────────────────────┘
```

## Phase roadmap

| Phase | Name | Scope |
|---|---|---|
| **A** | Quality Decision Lock | Quality compound · wire diagnosis knobs · unify hypothesis · Consistency Score |
| **B** | Decision Spine | Entry/Meta/Exit heads · pattern→policy · shadow auto-challenge · dynamic_exits |
| **C** | Neural Unification | LLModel same gates · counterfactual critic · regime memory without full retrain |

## Root-cause taxonomy

`TradePolicy | Model/HP | Features | Labels | DataQuality | ValidationDesign | MetricInflation | RegimeShift`

Priority when multiple smells: MetricInflation → TradePolicy → Model/HP → ValidationDesign → …

## Diagnosis knobs (Phase A — applied via pending_overrides / research next hyp)

| Code | Intent |
|---|---|
| `desaturate_trade_policy` | Lower cap + raise confidence; quality_first |
| `stabilize_early_fold_signal` | Milder decay + early AUC + honest Val |
| `regularize_capacity` | Depth/λ/features for overfit TFs (e.g. M5) |
| `regime_balanced_holdouts` | Require edge across ≥K regimes / balanced slices |
| `demote_uncapped_path_sharpe` | Trade-level primary; fail inflation |
| `regime_stable_policy` | Policy consensus ≥3 folds; no gap-gate spiral |

## Artifacts per run

| File | Purpose |
|---|---|
| `diagnosis.json` | Machine-readable causal diagnosis (+ consistency) |
| `evaluation_report.md` | + Self-Diagnostic section (EN+AR) |
| `enterprise_dossier.md` | + §7b Self-Diagnostic |
| `metrics_report.json` | `self_diagnosis` · `consistency` · `quality_compound` |
| `research_factory.json` | **One** hypothesis from diagnosis + suggested_config_diff |

## Ship order

1. Metric honesty + saturation (done)
2. Self-Diagnostic Engine (done)
3. Quality-first trade policy (v17.3 — done)
4. **Phase A — Quality Decision Lock (this ship)**
5. Phase B Decision Spine
6. Stricter full-CPCV promotion + Phase C neural unification

## Non-goals

- No gate weakening to force passes
- No parallel training pipeline
- No rewrite of CPCV core (upgrade flag prepared via diagnosis knobs)
- No FinalModel `live_ready` when readiness is `paper_ready`
- Prefer honest generalization over cosmetic Acc when they conflict
