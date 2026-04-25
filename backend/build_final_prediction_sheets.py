"""
Build the four "final prediction sheets" (one per UI section) under
backend/dataset/final_ml_ready/.

Outputs:
    tamilnadu_2026_default_prediction.csv         (Default Prediction tab)
    tamilnadu_2026_long_term_trend_sheet.csv      (Long-Term Trend tab)
    tamilnadu_2026_recent_swing_sheet.csv         (Recent Swing tab)
    tamilnadu_2026_live_intelligence_score_sheet.csv (Live Intelligence tab)

Re-runnable. Reads from predictions_2026.csv and the analysis module so
the sheets stay in sync with whatever the UI is showing. Does not touch
the existing predictions or training pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

from analysis import ANALYSIS_TYPES, run_analysis
from config import DATASET_DIR, PREDICTIONS_DIR

FINAL_DIR = Path(DATASET_DIR) / "final_ml_ready"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
PRED_FILE = Path(PREDICTIONS_DIR) / "predictions_2026.csv"


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Default Prediction sheet (mirrors the model's predictions_2026.csv)
# ---------------------------------------------------------------------------

def build_default_prediction_sheet() -> Path:
    if not PRED_FILE.exists():
        raise FileNotFoundError(
            f"{PRED_FILE} not found. Run `python backend/train.py` first."
        )
    dst = FINAL_DIR / "tamilnadu_2026_default_prediction.csv"
    header = [
        "ac_no", "constituency", "district", "predicted", "winning_score",
        "DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS",
    ]
    rows: List[List] = []
    with PRED_FILE.open("r", encoding="utf-8", newline="") as fp:
        for r in csv.DictReader(fp):
            rows.append([
                r.get("ac_no", ""),
                r.get("constituency", ""),
                r.get("district", ""),
                r.get("predicted", ""),
                r.get("confidence", ""),
                r.get("DMK_ALLIANCE", ""),
                r.get("AIADMK_NDA", ""),
                r.get("TVK", ""),
                r.get("NTK", ""),
                r.get("OTHERS", ""),
            ])
    rows.sort(key=lambda x: int(x[0] or 0))
    _write_csv(dst, header, rows)
    return dst


# ---------------------------------------------------------------------------
# Per-tab analysis sheets
# ---------------------------------------------------------------------------

# Per-tab columns appended after the shared base columns.
_TAB_EXTRA_COLUMNS = {
    "long_term_trend": [
        "winner_2016", "winner_2021",
        "historical_strength", "party_growth_score",
        "long_term_trend_score",
    ],
    "recent_swing": [
        "winner_party_2021", "runner_up_party_2021",
        "incumbency_status", "constituency_swing",
        "seat_retention_probability", "opposition_gain_probability",
        "recent_swing_score",
    ],
    "live_intelligence_score": [
        "sentiment_adjusted_prediction", "tvk_impact_score",
        "confidence_level", "live_intelligence_score",
    ],
}

_BASE_COLUMNS = [
    "ac_no", "constituency", "district",
    "model_predicted", "analysis_predicted", "win_probability",
    "DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS",
]


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if value is None:
        return ""
    return value


def build_analysis_sheet(analysis_type: str) -> Path:
    rows, _meta = run_analysis(analysis_type)
    extras = _TAB_EXTRA_COLUMNS[analysis_type]
    header = [*_BASE_COLUMNS, *extras, "final_prediction_score"]

    out_rows: List[List] = []
    for r in rows:
        base = [
            r.get("ac_no", ""),
            r.get("constituency", ""),
            r.get("district", ""),
            r.get("predicted", ""),
            r.get("analysis_predicted", r.get("predicted", "")),
            r.get("win_probability", r.get("confidence", "")),
            r.get("DMK_ALLIANCE", ""),
            r.get("AIADMK_NDA", ""),
            r.get("TVK", ""),
            r.get("NTK", ""),
            r.get("OTHERS", ""),
        ]
        extra_values = [_flatten(r.get(col, "")) for col in extras]
        base.extend(extra_values)
        base.append(r.get("final_prediction_score", ""))
        out_rows.append(base)

    out_rows.sort(key=lambda x: int(x[0] or 0))
    dst = FINAL_DIR / f"tamilnadu_2026_{analysis_type}_sheet.csv"
    _write_csv(dst, header, out_rows)
    return dst


def main() -> None:
    written: List[Path] = []
    written.append(build_default_prediction_sheet())
    for at in ANALYSIS_TYPES:
        written.append(build_analysis_sheet(at))
    print("Final prediction sheets written:")
    for p in written:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
