# Migration note — report / UI fields (e4-v17.4 NA-QL Phase A)

## New / changed fields (v17.4 Phase A)

| Path | Type | Notes |
|---|---|---|
| `metrics.consistency` | object | Fold×regime Consistency Score report |
| `metrics.consistency.score` | float 0–100 | Higher = lower WR/F1/Exp dispersion |
| `metrics.consistency.dispersion` | object | CV of fold/regime WR·F1·expectancy |
| `metrics.consistency.summary_ar` / `summary_en` | string | Human-readable |
| `metrics.quality_compound` | object | WR+F1+expectancy − sat/inflation penalties |
| `metrics.quality_compound.score` | float | Ranking score (not a gate by itself) |
| `metrics.regime_balanced_holdouts` | object | Optional regime coverage gate |
| `metrics.self_diagnosis.consistency_score` | float\|null | Mirror of consistency.score |
| `metrics.self_diagnosis.unified_hypothesis` | object | **One** hypothesis + knobs + acceptance |
| `metrics.unified_hypothesis` | object | Top-level mirror for UI |
| `metrics.suggested_config_diff` | object | Dominant knobs (unchanged path) |
| `research_factory.hypothesis` | object | Always prefers diagnosis when present |
| `research_factory.suggested_config_diff` | object | Apply-path knobs |
| `smart_recommendations.unified_from_diagnosis` | bool | Primary item = diagnosis |
| `smart_recommendations.primary_knobs` | object | Knobs of primary recommendation |
| `live_readiness` material warning `low_consistency` | string | Caps live_ready when consistency weak |
| Artifact `diagnosis.json` | file | version 2 + unified_hypothesis + consistency |

## Prior fields (v17.2 Self-Diagnostic — still valid)

| Path | Type | Notes |
|---|---|---|
| `metrics.self_diagnosis` | object | Full causal diagnosis |
| `metrics.self_diagnosis.primary_root_cause` | string | One of 8 taxonomy values |
| `metrics.self_diagnosis.primary_root_cause_ar` | string | Arabic label |
| `metrics.self_diagnosis.metric_honesty_score` | float 0–100 | Anti-inflation honesty |
| `metrics.self_diagnosis.generalization_score` | float 0–100 | Train/Val/Test + folds |
| `metrics.self_diagnosis.live_tradability_score` | float 0–100 | Post-cost / deploy / stress |
| `metrics.self_diagnosis.evidence` | array | Gate→metric→value→threshold rows |
| `metrics.self_diagnosis.next_actions` | array | Ranked falsifiable patches |
| `metrics.self_diagnosis.suggested_config_diff` | object | Dominant knobs for next run |
| `metrics.self_diagnosis.safe_for_live` | object | `{verdict, verdict_ar, constraints_failed, safe_for_live}` |
| `metrics.self_diagnosis.narrative_en` / `narrative_ar` | string | Risk-meeting narrative |
| `live_readiness.material_warnings` | string[] | Blocks live_ready when hard smells |
| `sharpe_inflation.reasons` | string[] | Why inflated |
| `sharpe_inflation.diagnostic_only_uncapped` | bool | Always true |

## Compatibility

- Existing `gate_failures` / `gate_failures_detail` / Arabic `GATE_FAILURE_AR` unchanged except additive key `regime_balanced_holdouts_weak`.
- `live_readiness.verdict` values unchanged (`live_ready|paper_ready|research_only|blocked`).
- `regime_balanced_holdouts` defaults **false** — diagnosis may enable it next run without weakening other gates.
- Prefer displaying `quality_compound.score` as a research ranking aid, **not** as a live KPI that overrides honesty gates.

## UI suggestions

1. Show `safe_for_live.verdict_ar` next to readiness pill.
2. Add honesty / generalization / live_tradability / **consistency** quartet under Advanced Validation.
3. Render `unified_hypothesis` (or `next_actions[0]`) as “التجربة التالية” with knobs diff + acceptance checklist.
4. Show `quality_compound.score` beside WR/F1/Expectancy with penalty chips (`trade_rate_pegged`, `metric_inflation`).
5. Do **not** display `sharpe_uncapped` as a primary KPI; keep under “Diagnostic only”.
