"""
Builds the analysis-bucket dataset folders consumed by the
`/api/predictions?analysis_type=...` filters.

Populates:
    backend/dataset/long_term_trend/
    backend/dataset/recent_swing/
    backend/dataset/live_intelligence_score/
    backend/dataset/final_ml_ready/

Re-runnable. Existing files are overwritten with up-to-date content
copied / derived from the canonical sources in
backend/dataset/data_files/ and backend/dataset/predictions/.

Does NOT touch any existing training, prediction, or backtest path.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from config import DATA_FILES_DIR, DATASET_DIR, PARTIES, PREDICTIONS_DIR

LONG_TERM_DIR = Path(DATASET_DIR) / "long_term_trend"
RECENT_SWING_DIR = Path(DATASET_DIR) / "recent_swing"
LIVE_INTEL_DIR = Path(DATASET_DIR) / "live_intelligence_score"
FINAL_ML_DIR = Path(DATASET_DIR) / "final_ml_ready"

DATA_FILES = Path(DATA_FILES_DIR)
PRED_FILE = Path(PREDICTIONS_DIR) / "predictions_2026.csv"
ASSEMBLY_2026_PROJECTION = DATA_FILES / "tamilnadu_assembly_2026.csv"


def _ensure_dirs() -> None:
    for d in (LONG_TERM_DIR, RECENT_SWING_DIR, LIVE_INTEL_DIR, FINAL_ML_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source CSV missing: {src}")
    shutil.copyfile(src, dst)


def _build_prediction_base(dst: Path) -> None:
    """
    Per-constituency 2026 prediction base: shares + winner + margin.
    Used by both long_term_trend and recent_swing as the 2026 anchor.
    """
    if PRED_FILE.exists():
        rows = []
        with PRED_FILE.open("r", encoding="utf-8", newline="") as fp:
            for r in csv.DictReader(fp):
                rows.append(r)
        fields = [
            "ac_no", "constituency", "district",
            "predicted", "confidence",
            "DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS",
        ]
        with dst.open("w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        return

    # Fallback: derive from projection layer if predictions_2026.csv hasn't
    # been generated yet. Keeps this builder runnable end-to-end.
    if not ASSEMBLY_2026_PROJECTION.exists():
        raise FileNotFoundError(
            "Neither predictions_2026.csv nor tamilnadu_assembly_2026.csv "
            "was found. Run the training pipeline first."
        )

    fields = [
        "ac_no", "constituency", "district",
        "predicted", "confidence",
        "DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS",
    ]
    with ASSEMBLY_2026_PROJECTION.open("r", encoding="utf-8", newline="") as fp, \
         dst.open("w", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        for r in csv.DictReader(fp):
            shares = {
                "DMK_ALLIANCE": float(r.get("proj_2026_dmk_alliance_pct", 0) or 0),
                "AIADMK_NDA":   float(r.get("proj_2026_aiadmk_nda_pct", 0) or 0),
                "TVK":          float(r.get("proj_2026_tvk_pct", 0) or 0),
                "NTK":          float(r.get("proj_2026_ntk_pct", 0) or 0),
                "OTHERS":       float(r.get("proj_2026_others_pct", 0) or 0),
            }
            predicted = r.get("proj_2026_winner") or max(shares, key=shares.get)
            w.writerow({
                "ac_no": r.get("ac_no", ""),
                "constituency": r.get("ac_name", ""),
                "district": r.get("district", ""),
                "predicted": predicted,
                "confidence": shares.get(predicted, 0.0),
                **shares,
            })


def _build_long_term_trend() -> None:
    _copy(DATA_FILES / "tamilnadu_assembly_election_2016.csv",
          LONG_TERM_DIR / "tamilnadu_assembly_2016.csv")
    _copy(DATA_FILES / "tamilnadu_assembly_election_2021.csv",
          LONG_TERM_DIR / "tamilnadu_assembly_2021.csv")
    _copy(DATA_FILES / "tamilnadu_lok_sabha_election_2014.csv",
          LONG_TERM_DIR / "tamilnadu_lok_sabha_2014.csv")
    _copy(DATA_FILES / "tamilnadu_lok_sabha_election_2019.csv",
          LONG_TERM_DIR / "tamilnadu_lok_sabha_2019.csv")
    _copy(DATA_FILES / "tamilnadu_lok_sabha_election_2024.csv",
          LONG_TERM_DIR / "tamilnadu_lok_sabha_2024.csv")
    _build_prediction_base(LONG_TERM_DIR / "tamilnadu_assembly_2026_prediction_base.csv")


def _build_recent_swing() -> None:
    _copy(DATA_FILES / "tamilnadu_assembly_election_2021.csv",
          RECENT_SWING_DIR / "tamilnadu_assembly_2021.csv")
    _copy(DATA_FILES / "tamilnadu_lok_sabha_election_2024.csv",
          RECENT_SWING_DIR / "tamilnadu_lok_sabha_2024.csv")
    _build_prediction_base(RECENT_SWING_DIR / "tamilnadu_assembly_2026_prediction_base.csv")


# ---------------------------------------------------------------------------
# Live intelligence seed sentiment scores
# ---------------------------------------------------------------------------
# Values are 0.0 - 1.0 normalised "favourability" scores. They are seed
# defaults sourced from tamilnadu_sentiment_analysis_2026.csv categorical
# bands (Low / Low-Medium / Medium / Medium-High / High) + manually curated
# qualitative reads of the 2026 cycle so the live_intelligence pipeline has
# something to consume out of the box. Replace with live_collectors output
# when ENABLE_LIVE_SENTIMENT is on.

PARTY_SENTIMENT_SEED = {
    "DMK_ALLIANCE": 0.62,
    "AIADMK_NDA":   0.55,
    "TVK":          0.48,
    "NTK":          0.32,
    "OTHERS":       0.25,
}

LEADER_SENTIMENT_SEED = {
    "DMK_ALLIANCE": ("M.K. Stalin",         0.64),
    "AIADMK_NDA":   ("Edappadi K. Palaniswami", 0.52),
    "TVK":          ("Thalapathy Vijay",    0.71),
    "NTK":          ("Seeman",              0.41),
    "OTHERS":       ("Various",             0.28),
}

CANDIDATE_SENTIMENT_SEED = {
    "DMK_ALLIANCE": 0.58,
    "AIADMK_NDA":   0.51,
    "TVK":          0.46,
    "NTK":          0.36,
    "OTHERS":       0.27,
}

SOCIAL_MEDIA_SEED = {
    "DMK_ALLIANCE": 0.57,
    "AIADMK_NDA":   0.49,
    "TVK":          0.74,
    "NTK":          0.44,
    "OTHERS":       0.26,
}

NEWS_SENTIMENT_SEED = {
    "DMK_ALLIANCE": 0.60,
    "AIADMK_NDA":   0.50,
    "TVK":          0.55,
    "NTK":          0.34,
    "OTHERS":       0.28,
}

LOCAL_ISSUE_SEED = [
    ("Cost of Living",            "DMK_ALLIANCE", 0.42),
    ("Freebies / Welfare",        "DMK_ALLIANCE", 0.66),
    ("Corruption Allegations",    "AIADMK_NDA",   0.48),
    ("Law and Order",             "AIADMK_NDA",   0.52),
    ("Youth Unemployment",        "TVK",          0.61),
    ("Tamil Identity",            "NTK",          0.55),
    ("Hindi Imposition",          "DMK_ALLIANCE", 0.63),
    ("Education Quality",         "DMK_ALLIANCE", 0.58),
    ("Anti-Incumbency",           "TVK",          0.49),
    ("Women Safety",              "AIADMK_NDA",   0.46),
]

TVK_IMPACT_SEED = {
    "tvk_state_buzz_score":       0.74,
    "vijay_personal_rating":      0.71,
    "youth_pull_score":           0.78,
    "first_time_voter_score":     0.69,
    "media_coverage_score":       0.66,
    "vote_split_risk_dmk":        0.42,
    "vote_split_risk_aiadmk":     0.31,
    "expected_vote_share_2026":   0.124,
}


def _write_party_seed(dst: Path, scores: dict, score_col: str) -> None:
    with dst.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["party", score_col])
        for p in PARTIES:
            w.writerow([p, scores.get(p, 0.0)])


def _write_leader_seed(dst: Path) -> None:
    with dst.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["party", "leader", "leader_sentiment_score"])
        for p in PARTIES:
            leader, score = LEADER_SENTIMENT_SEED.get(p, ("", 0.0))
            w.writerow([p, leader, score])


def _write_local_issue_seed(dst: Path) -> None:
    with dst.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["issue", "favoured_party", "issue_score"])
        for issue, party, score in LOCAL_ISSUE_SEED:
            w.writerow([issue, party, score])


def _write_tvk_impact_seed(dst: Path) -> None:
    with dst.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["metric", "value"])
        for k, v in TVK_IMPACT_SEED.items():
            w.writerow([k, v])


def _build_live_intelligence() -> None:
    _write_party_seed(LIVE_INTEL_DIR / "party_sentiment_2026.csv",
                      PARTY_SENTIMENT_SEED, "party_sentiment_score")
    _write_leader_seed(LIVE_INTEL_DIR / "leader_sentiment_2026.csv")
    _write_party_seed(LIVE_INTEL_DIR / "candidate_sentiment_2026.csv",
                      CANDIDATE_SENTIMENT_SEED, "candidate_sentiment_score")
    _write_party_seed(LIVE_INTEL_DIR / "social_media_sentiment_2026.csv",
                      SOCIAL_MEDIA_SEED, "social_media_sentiment_score")
    _write_party_seed(LIVE_INTEL_DIR / "news_sentiment_2026.csv",
                      NEWS_SENTIMENT_SEED, "news_sentiment_score")
    _write_local_issue_seed(LIVE_INTEL_DIR / "local_issue_score_2026.csv")
    _write_tvk_impact_seed(LIVE_INTEL_DIR / "tvk_impact_2026.csv")


def _build_final_ml_ready() -> None:
    """
    A flat per-constituency table that joins the 2026 prediction base with
    the analysis-typed scores once the analysis module computes them. We
    seed an initial copy so file presence holds; the analysis module
    refreshes it on demand.
    """
    dst = FINAL_ML_DIR / "tamilnadu_2026_prediction_dataset.csv"
    _build_prediction_base(dst)


def main() -> None:
    _ensure_dirs()
    _build_long_term_trend()
    _build_recent_swing()
    _build_live_intelligence()
    _build_final_ml_ready()
    print("Analysis dataset folders populated under:", DATASET_DIR)


if __name__ == "__main__":
    main()
