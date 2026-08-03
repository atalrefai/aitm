"""Example Self-Diagnostic replay on 20260803 multi-TF run fingerprints.

Reconstructs diagnosis objects from published evaluation_report numbers
(artifacts may be offline / OneDrive). Writes under models/intelligence/examples/.
"""

from __future__ import annotations

import json
from pathlib import Path

from atis.engines.engine4_training.self_diagnostic import build_self_diagnosis, write_diagnosis_json


# Fingerprints from ATIS Training Report 2026-08-03 (e4-v17.1-weakness-hardening)
RUNS = {
    "M1": {
        "passed": False,
        "gates": ["trade_rate_saturated", "inflated_sharpe"],
        "metrics": {
            "classification": {"roc_auc_ovr": 0.8402, "accuracy": 0.7583, "trade_rate_filtered": 0.1157},
            "financial_oos": {
                "sharpe": 9.0561,
                "sharpe_uncapped": 38.3483,
                "trade_sharpe_raw": 0.08,
                "n_trades": 967,
                "expectancy": 0.00171,
            },
            "financial_train": {"sharpe": 9.4799},
            "financial_validation": {"sharpe": 7.3904},
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": 2.0896,
                "sharpe_gap_val_test": 1.6657,
            },
            "fold_stability": {
                "trade_rate_pegged": True,
                "early_folds_weak": False,
                "stable": True,
            },
            "sharpe_inflation": {
                "inflated": True,
                "sharpe": 9.0561,
                "sharpe_uncapped": 38.3483,
                "trade_sharpe_raw": 0.08,
                "uncapped_ratio": 4.23,
                "path_vs_trade_gap": True,
            },
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.99},
            "stress_testing": {"robust": True, "worst_sharpe": 5.0},
        },
    },
    "M5": {
        "passed": False,
        "gates": ["trade_rate_saturated", "inflated_sharpe"],
        "metrics": {
            "classification": {"roc_auc_ovr": 0.82, "accuracy": 0.75, "trade_rate_filtered": 0.12},
            "financial_oos": {
                "sharpe": 8.61,
                "sharpe_uncapped": 20.26,
                "trade_sharpe_raw": 0.07,
                "n_trades": 500,
                "expectancy": 0.0015,
            },
            "financial_train": {"sharpe": 9.92},
            "financial_validation": {"sharpe": 9.01},
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": 0.91,
                "sharpe_gap_val_test": 0.4,
            },
            "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {
                "inflated": True,
                "sharpe": 8.61,
                "sharpe_uncapped": 20.26,
                "trade_sharpe_raw": 0.07,
                "uncapped_ratio": 2.35,
                "path_vs_trade_gap": True,
            },
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.98},
        },
    },
    "M15": {
        "passed": False,
        "gates": ["trade_rate_saturated", "early_folds_weak"],
        "metrics": {
            "classification": {"roc_auc_ovr": 0.80, "accuracy": 0.74, "trade_rate_filtered": 0.12},
            "financial_oos": {
                "sharpe": 7.69,
                "sharpe_uncapped": 11.24,
                "trade_sharpe_raw": 0.2,
                "n_trades": 400,
                "expectancy": 0.0014,
            },
            "financial_train": {"sharpe": 9.89},
            "financial_validation": {"sharpe": 9.41},
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": 0.48,
                "sharpe_gap_val_test": 1.72,
            },
            "fold_stability": {
                "trade_rate_pegged": True,
                "early_folds_weak": True,
                "stable": True,
                "early_fold_stats": {
                    "n_early": 2,
                    "mean_accuracy": 0.54,
                    "mean_test_sharpe": -0.2,
                    "frac_negative_test": 0.5,
                },
            },
            "sharpe_inflation": {"inflated": False},
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.9},
        },
    },
    "M30": {
        "passed": False,
        "gates": ["overfit_sharpe_gap", "trade_rate_saturated"],
        "metrics": {
            "classification": {"roc_auc_ovr": 0.8433, "accuracy": 0.7602, "trade_rate_filtered": 0.1503},
            "financial_oos": {
                "sharpe": 7.0791,
                "sharpe_uncapped": 8.2393,
                "trade_sharpe_raw": 0.22,
                "n_trades": 611,
                "expectancy": 0.00238,
            },
            "financial_train": {"sharpe": 13.8114},
            "financial_validation": {"sharpe": 9.7877},
            "fit_diagnosis": {
                "status": "overfitting",
                "sharpe_gap_train_val": 4.0237,
                "sharpe_gap_val_test": 2.7086,
                "accuracy_gap_train_val": 0.1612,
            },
            "fold_stability": {"trade_rate_pegged": True, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {"inflated": False},
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.5}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.85},
        },
    },
    "H1": {
        "passed": True,
        "gates": [],
        "metrics": {
            "classification": {"roc_auc_ovr": 0.8243, "accuracy": 0.7672, "trade_rate_filtered": 0.1006},
            "financial_oos": {
                "sharpe": 5.6014,
                "sharpe_uncapped": 5.6014,
                "trade_sharpe_raw": 0.28,
                "n_trades": 328,
                "expectancy": 0.00171,
            },
            "financial_train": {"sharpe": 8.1607},
            "financial_validation": {"sharpe": 8.8383},
            "financial_deploy_holdout": {"sharpe": 4.8826, "n_trades": 95},
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": -0.6776,
                "sharpe_gap_val_test": 3.2369,
            },
            "fold_stability": {"trade_rate_pegged": False, "early_folds_weak": False, "stable": True},
            "sharpe_inflation": {"inflated": False},
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 1.0}, "pbo": {"pbo": 0.25}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.75},
            "stress_testing": {"robust": True, "worst_sharpe": 2.0},
            "expectancy_vs_cost": {"covers": True},
        },
    },
    "H4": {
        "passed": False,
        "gates": [
            "weak_sharpe_ci",
            "deploy_holdout_sharpe",
            "h4_no_edge",
            "crisis_holdout_weak",
            "early_folds_weak",
        ],
        "metrics": {
            "classification": {"roc_auc_ovr": 0.5120, "accuracy": 0.5005, "trade_rate_filtered": 0.0731},
            "financial_oos": {
                "sharpe": 1.0232,
                "sharpe_uncapped": 1.0232,
                "trade_sharpe_raw": 0.05,
                "n_trades": 58,
                "expectancy": 0.00066,
            },
            "financial_train": {"sharpe": -0.9397},
            "financial_validation": {"sharpe": 2.3213},
            "financial_deploy_holdout": {"sharpe": -0.0688, "n_trades": 21},
            "fit_diagnosis": {
                "status": "balanced",
                "sharpe_gap_train_val": -3.261,
                "sharpe_gap_val_test": 1.2981,
                "filter_driven_edge_risk": True,
            },
            "fold_stability": {
                "trade_rate_pegged": False,
                "early_folds_weak": True,
                "stable": True,
                "early_fold_stats": {
                    "n_early": 2,
                    "mean_accuracy": 0.50,
                    "mean_test_sharpe": -0.3,
                    "frac_negative_test": 0.5,
                },
            },
            "sharpe_inflation": {"inflated": False},
            "advanced_eval": {"deflated_sharpe": {"deflated_sharpe": 0.0}, "pbo": {"pbo": 0.25}},
            "monte_carlo": {"enabled": True, "stable": True, "p_profit": 0.55},
            "data_quality": {
                "n_features": 12,
                "quality_flags": {"excessive_gaps": True},
                "gate_pass": True,
            },
        },
    },
}

CFG = {
    "max_fold_trade_rate": 0.12,
    "max_sharpe_uncapped": 15.0,
    "max_uncapped_to_capped_ratio": 2.5,
    "max_path_vs_trade_gap": 0.12,
    "max_train_val_sharpe_gap": 2.0,
    "early_fold_min_acc": 0.58,
    "min_auc_for_live": 0.515,
}


def main() -> None:
    from atis.config import PROJECT_ROOT

    root = Path(PROJECT_ROOT) / "models" / "intelligence" / "examples" / "20260803"
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for tf, row in RUNS.items():
        m = dict(row["metrics"])
        m["gate_failures"] = list(row["gates"])
        diag = build_self_diagnosis(
            m,
            timeframe=tf,
            passed_gates=bool(row["passed"]),
            cfg=CFG,
            gate_failures=list(row["gates"]),
        )
        write_diagnosis_json(root / f"{tf}_diagnosis.json", diag)
        summary.append(
            {
                "tf": tf,
                "passed": row["passed"],
                "primary_root_cause": diag["primary_root_cause"],
                "honesty": diag["metric_honesty_score"],
                "generalization": diag["generalization_score"],
                "live_tradability": diag["live_tradability_score"],
                "safe_for_live": diag["safe_for_live"]["verdict"],
                "next": (diag["next_actions"] or [{}])[0].get("code"),
                "narrative_ar": diag["narrative_ar"],
            }
        )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# Example diagnosis — 20260803 multi-TF run", ""]
    md.append("| TF | Passed | Root cause | Honesty | Gen | Live | Safe | Next |")
    md.append("|---|---|---|---:|---:|---:|---|---|")
    for s in summary:
        md.append(
            f"| {s['tf']} | {s['passed']} | {s['primary_root_cause']} | "
            f"{s['honesty']} | {s['generalization']} | {s['live_tradability']} | "
            f"{s['safe_for_live']} | `{s['next']}` |"
        )
    md.extend(["", "## Narratives (AR)", ""])
    for s in summary:
        md.append(f"- **{s['tf']}**: {s['narrative_ar']}")
    (root / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {len(summary)} diagnoses -> {root}")


if __name__ == "__main__":
    main()
