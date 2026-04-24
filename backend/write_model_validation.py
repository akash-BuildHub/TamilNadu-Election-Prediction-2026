"""
Write the model-validation artefacts so users and downstream consumers never
mistake the ~91% train.py CV score for real-world forecasting accuracy.

Outputs (all reproducible -- re-run after each training cycle):
  backend/backtests/model_validation_summary.md    human-readable
  backend/backtests/model_validation_summary.json  machine-readable
  backend/predictions_2026_validated.csv           copy of predictions_2026.csv
                                                   + confidence_type
                                                   + validation_note

Does NOT modify predictions_2026.csv, train.py outputs, or any checkpoints.

Usage:
  python backend/write_model_validation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
BACKTESTS_DIR = BACKEND_DIR / "backtests"
PREDICTIONS_IN = BACKEND_DIR / "predictions_2026.csv"
PREDICTIONS_OUT = BACKEND_DIR / "predictions_2026_validated.csv"
MD_OUT = BACKTESTS_DIR / "model_validation_summary.md"
JSON_OUT = BACKTESTS_DIR / "model_validation_summary.json"

# The authoritative validation numbers. Sourced from:
#   train.py                 -> synthetic-label CV run
#   backtest_2021.py         -> party-level historical backtest
#   backtest_2021_alliance.py -> alliance-level historical backtest
#
# Update these constants if you re-run any of the three and want the summary
# to reflect the latest numbers. Each has a corresponding JSON in backtests/.
TRAIN_CV_ACCURACY = 0.9104
TRAIN_CV_STD = 0.1037
PARTY_BACKTEST_CV = 0.6325
PARTY_BACKTEST_HOLDOUT = 0.6596
ALLIANCE_BACKTEST_CV = 0.7607
ALLIANCE_BACKTEST_HOLDOUT = 0.7234
ALLIANCE_MAJORITY_BASELINE = 0.6795
PARTY_MAJORITY_BASELINE = 0.5684  # 133 DMK seats / 234 ACs in 2021

# The canonical user-facing disclaimer. Every code path that surfaces
# confidence numbers should reference this string verbatim.
VALIDATION_NOTE = (
    "Internal synthetic-label CV accuracy: 91.04%. "
    "Historical backtest accuracy: 63.25% party-level and 76.07% alliance-level. "
    "Treat constituency-level predictions as directional, not guaranteed."
)
CONFIDENCE_TYPE = "relative_model_confidence_not_true_probability"


def write_markdown() -> None:
    lines = [
        "# Model Validation Summary — Tamil Nadu 2026 Prediction",
        "",
        "_This file is the single source of truth for how to interpret",
        "prediction confidence. Re-run `python backend/write_model_validation.py`",
        "after retraining._",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value | What it measures |",
        "|---|---|---|",
        f"| `train.py` synthetic-label CV accuracy | **{TRAIN_CV_ACCURACY:.4f} ± {TRAIN_CV_STD:.4f}** | How well the neural net reproduces `proj_2026_winner`, a synthetic label built by `create_dataset.py` using 2021 AC-level winner as its strong prior. |",
        f"| Party-level historical backtest CV accuracy | **{PARTY_BACKTEST_CV:.4f}** | 2016 features → real 2021 winner party (8 classes, RandomForest, K=2 stratified). |",
        f"| Party-level holdout accuracy | {PARTY_BACKTEST_HOLDOUT:.4f} | stratified 80/20 holdout, same setup. |",
        f"| Alliance-level historical backtest CV accuracy | **{ALLIANCE_BACKTEST_CV:.4f}** | 2016 features → real 2021 winning alliance (3 classes, RandomForest, K=5 stratified). |",
        f"| Alliance-level holdout accuracy | {ALLIANCE_BACKTEST_HOLDOUT:.4f} | stratified 80/20 holdout, same setup. |",
        f"| Alliance majority baseline (always DMK-led) | {ALLIANCE_MAJORITY_BASELINE:.4f} | naive ‘always predict the majority class’ baseline, alliance level. |",
        f"| Party majority baseline (always DMK) | {PARTY_MAJORITY_BASELINE:.4f} | naive ‘always predict DMK’ baseline, party level. |",
        "",
        "## Why the 91% number is not real forecast accuracy",
        "",
        "`train.py` reports its cross-validation accuracy against the label",
        "`proj_2026_winner`, which is **not** a real election outcome. That",
        "label is synthesised inside `create_dataset.py` using 2021",
        "constituency-level winners as the anchor, then perturbed by state",
        "swing, alliance supply caps, and a small amount of noise. The",
        "verified sidecar features added by `data_loader.py` also carry 2021",
        "AC-level data. Since the input features and the label both derive",
        "from the same 2021 base, the model is effectively being graded on",
        "reproducing projection logic — a reproduction-of-heuristics score,",
        "not a prediction score.",
        "",
        "## Use these numbers for real-world reliability",
        "",
        f"- Historical backtest (party level): **{PARTY_BACKTEST_CV*100:.2f}%**",
        f"  — about **+{(PARTY_BACKTEST_CV - PARTY_MAJORITY_BASELINE)*100:.1f} pp** over the always-DMK baseline.",
        f"- Historical backtest (alliance level): **{ALLIANCE_BACKTEST_CV*100:.2f}%**",
        f"  — about **+{(ALLIANCE_BACKTEST_CV - ALLIANCE_MAJORITY_BASELINE)*100:.1f} pp** over the always-DMK-alliance baseline.",
        "- Recommended range to cite in any report: **63–76%**,",
        "  depending on whether the target is party-level or alliance-level.",
        "",
        "## One-line validation note (used by the API and the validated CSV)",
        "",
        f"> {VALIDATION_NOTE}",
        "",
        "## Confidence columns in predictions_2026.csv",
        "",
        "The `confidence` column is the top-1 predicted-party probability",
        "from the model's softmax. It is a relative model-confidence score,",
        "**not a calibrated probability of the real-world event**. The",
        "validated CSV explicitly tags every row with",
        f"`confidence_type = \"{CONFIDENCE_TYPE}\"` to make this unambiguous",
        "to downstream consumers.",
        "",
        "## Changing these numbers",
        "",
        "Any of the three validation numbers can be refreshed by re-running",
        "the corresponding script, then editing the constants at the top of",
        "`backend/write_model_validation.py` and re-running this script.",
        "",
        "- `train.py` → `TRAIN_CV_ACCURACY`, `TRAIN_CV_STD`",
        "- `backtest_2021.py` → `PARTY_BACKTEST_CV`, `PARTY_BACKTEST_HOLDOUT`",
        "- `backtest_2021_alliance.py` → `ALLIANCE_BACKTEST_CV`, `ALLIANCE_BACKTEST_HOLDOUT`",
        "",
    ]
    MD_OUT.parent.mkdir(exist_ok=True)
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_json() -> None:
    payload = {
        "train_py_synthetic_cv_accuracy": TRAIN_CV_ACCURACY,
        "train_py_synthetic_cv_std": TRAIN_CV_STD,
        "party_level_backtest": {
            "cv_accuracy": PARTY_BACKTEST_CV,
            "holdout_accuracy": PARTY_BACKTEST_HOLDOUT,
            "majority_baseline": PARTY_MAJORITY_BASELINE,
        },
        "alliance_level_backtest": {
            "cv_accuracy": ALLIANCE_BACKTEST_CV,
            "holdout_accuracy": ALLIANCE_BACKTEST_HOLDOUT,
            "majority_baseline": ALLIANCE_MAJORITY_BASELINE,
        },
        "realistic_accuracy_range": {
            "min": PARTY_BACKTEST_CV,
            "max": ALLIANCE_BACKTEST_CV,
            "note": "party-level lower bound, alliance-level upper bound",
        },
        "validation_note": VALIDATION_NOTE,
        "confidence_type": CONFIDENCE_TYPE,
        "sources": {
            "train_py_cv": "backend/train.py (CV Accuracy line)",
            "party_backtest": "backend/backtests/backtest_2021_metrics.json",
            "alliance_backtest": "backend/backtests/backtest_2021_alliance_metrics.json",
        },
    }
    JSON_OUT.parent.mkdir(exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_validated_predictions() -> None:
    if not PREDICTIONS_IN.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_IN} not found. Run `python backend/train.py` first."
        )

    with PREDICTIONS_IN.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        in_fields = list(reader.fieldnames or [])

    out_fields = in_fields + ["confidence_type", "validation_note"]
    # Skip if columns already present (idempotent re-run).
    for row in rows:
        row["confidence_type"] = CONFIDENCE_TYPE
        row["validation_note"] = VALIDATION_NOTE

    with PREDICTIONS_OUT.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    write_markdown()
    write_json()
    write_validated_predictions()
    print("Wrote:")
    for p in (MD_OUT, JSON_OUT, PREDICTIONS_OUT):
        rel = p.relative_to(BACKEND_DIR)
        print(f"  {rel}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
