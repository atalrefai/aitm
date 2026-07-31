"""Write models/EVALUATION_SUMMARY.md from latest per-TF metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    lines = [
        "# ATIS Final Training Evaluation Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "- Pipeline: e4-v4-eval-robust-20260731",
        "",
    ]
    fm = json.loads((MODELS / "FinalModel" / "FINAL_MODEL.json").read_text(encoding="utf-8"))
    lines += [
        "## Final Model",
        f"- TF: **{fm.get('timeframe')}**",
        f"- Mode: **{fm.get('mode')}**",
        f"- Version: `{fm.get('version')}`",
        f"- Passed gates: {fm.get('passed_gates')}",
        "",
        "## Per-Timeframe Snapshot",
        "| TF | Test Sharpe | Val Sharpe | Train Sharpe | Acc | AUC | Diagnosis | Gates |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for tf in ["H1", "H4", "M30", "M15"]:
        p = sorted((MODELS / "XAUUSD" / tf).glob("*/metrics_report.json"))[-1]
        m = json.loads(p.read_text(encoding="utf-8"))
        fin = m.get("financial_oos") or {}
        val = m.get("financial_validation") or {}
        tr = m.get("financial_train") or {}
        cls = m.get("classification") or {}
        d = (m.get("fit_diagnosis") or {}).get("status")
        lines.append(
            f"| {tf} | {float(fin.get('sharpe', 0)):.3f} | {float(val.get('sharpe', 0)):.3f} | "
            f"{float(tr.get('sharpe', 0)):.3f} | {float(cls.get('accuracy', 0)):.3f} | "
            f"{float(cls.get('roc_auc_ovr', 0)):.3f} | {d} | {m.get('passed_gates')} |"
        )
    lines += [
        "",
        "## Strengths",
        "- Purged walk-forward + nested validation reduces leakage.",
        "- Overfit detector blocks Train>>Val models (H1/H4 flagged).",
        "- M30 passed with positive OOS Sharpe under stricter diagnosis.",
        "- `evaluation_report.md` written per TF version.",
        "",
        "## Weaknesses",
        "- Classification accuracy/AUC still near chance (~0.47–0.53 / ~0.50).",
        "- Financial edge depends on sparse filtered trades.",
        "- Val still optimistic vs Test on several TFs.",
        "",
        "## Recommendations",
        "- Next: meta-labeling / barrier redesign to lift directional AUC above 0.55.",
        "- Keep H1 trade rate moderate; avoid chasing Train Sharpe.",
        "- Prefer FinalModel M30 for paper/live until H1 overfit gap shrinks.",
        "",
    ]
    out = MODELS / "EVALUATION_SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
