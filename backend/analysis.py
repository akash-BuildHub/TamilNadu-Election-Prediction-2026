"""
Analysis filter system for the 2026 Tamil Nadu Assembly prediction.

Exposes three analysis types consumed by:
    GET /api/predictions?analysis_type=long_term_trend
    GET /api/predictions?analysis_type=recent_swing
    GET /api/predictions?analysis_type=live_intelligence_score

Each produces:
    - per-constituency rows (PredictionRow shape) with extra fields for
      the selected analysis type
    - an `analysis_meta` block describing what was compared and how

The final per-constituency score blends all three:
    final_prediction_score = 0.40 * long_term + 0.35 * recent_swing
                           + 0.25 * live_intelligence

This module is read-only with respect to the existing prediction
pipeline (predictions_2026.csv stays the source of truth for the
default `/api/predictions` response). The analysis filters only adjust
how that data is presented and which extra fields are surfaced.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DATASET_DIR, PARTIES

# Election-year mapping
#   2016 Assembly -> 2014 Lok Sabha
#   2021 Assembly -> 2019 Lok Sabha
#   2026 Assembly -> 2024 Lok Sabha
ASSEMBLY_TO_LOKSABHA = {2016: 2014, 2021: 2019, 2026: 2024}
TARGET_ASSEMBLY_YEAR = 2026
TARGET_LOK_SABHA_YEAR = ASSEMBLY_TO_LOKSABHA[TARGET_ASSEMBLY_YEAR]

ANALYSIS_TYPES: Tuple[str, ...] = (
    "long_term_trend",
    "recent_swing",
    "live_intelligence_score",
)

# Final score weights. Sum = 1.0.
FINAL_SCORE_WEIGHTS = {
    "long_term_trend": 0.40,
    "recent_swing":    0.35,
    "live_intelligence_score": 0.25,
}

# Party-to-alliance projections used when collapsing the historical
# party-level vote-share tables into the 4-class space the model uses.
# Mirrors PARTY_TO_ALLIANCE_2026 in config.py with the addition of
# the older standalone parties so the 2014/2016 rows project cleanly.
PARTY_TO_ALLIANCE = {
    # DMK_ALLIANCE
    "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE",
    "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE",
    "VCK": "DMK_ALLIANCE", "MDMK": "DMK_ALLIANCE",
    "IUML": "DMK_ALLIANCE",
    # AIADMK_NDA
    "AIADMK": "AIADMK_NDA", "BJP": "AIADMK_NDA",
    "PMK": "AIADMK_NDA", "AMMK": "AIADMK_NDA",
    "DMDK": "AIADMK_NDA",  # 2014/2016 cycle alignment
    # Standalone
    "TVK": "TVK",
    "NTK": "NTK",
    # Everything else
    "MNM": "OTHERS",
    "Other": "OTHERS",
    "IND": "OTHERS",
}

LONG_TERM_DIR = Path(DATASET_DIR) / "long_term_trend"
RECENT_SWING_DIR = Path(DATASET_DIR) / "recent_swing"
LIVE_INTEL_DIR = Path(DATASET_DIR) / "live_intelligence_score"


# ---------------------------------------------------------------------------
# Small CSV helpers
# ---------------------------------------------------------------------------

def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _alliance_share_table(rows: List[dict]) -> Dict[str, float]:
    """
    Collapse a state-level party-row CSV (party / vote_share) down to the
    5-class alliance space used by the model.
    """
    out = {p: 0.0 for p in PARTIES}
    for r in rows:
        party = (r.get("party") or "").strip()
        share = _to_float(r.get("vote_share", 0))
        alliance = PARTY_TO_ALLIANCE.get(party, "OTHERS")
        out[alliance] += share
    return out


def _alliance_seat_table(rows: List[dict]) -> Dict[str, int]:
    out = {p: 0 for p in PARTIES}
    for r in rows:
        party = (r.get("party") or "").strip()
        seats = _to_int(r.get("seats_won", 0))
        alliance = PARTY_TO_ALLIANCE.get(party, "OTHERS")
        out[alliance] += seats
    return out


def _pick_winner(scores: Dict[str, float]) -> str:
    """Argmax over the per-party score dict, deterministic tie-break by PARTIES order."""
    best_party = PARTIES[0]
    best_score = scores.get(best_party, 0.0)
    for p in PARTIES[1:]:
        if scores.get(p, 0.0) > best_score:
            best_score = scores[p]
            best_party = p
    return best_party


# ---------------------------------------------------------------------------
# Prediction-base loading
# ---------------------------------------------------------------------------

@dataclass
class PredictionBaseRow:
    ac_no: int
    constituency: str
    district: str
    predicted: str
    confidence: float
    shares: Dict[str, float]


def _load_prediction_base(folder: Path) -> List[PredictionBaseRow]:
    path = folder / "tamilnadu_assembly_2026_prediction_base.csv"
    out: List[PredictionBaseRow] = []
    for r in _read_csv(path):
        shares = {p: _to_float(r.get(p, 0)) for p in PARTIES}
        predicted = r.get("predicted") or max(shares, key=shares.get)
        out.append(PredictionBaseRow(
            ac_no=_to_int(r.get("ac_no", 0)),
            constituency=r.get("constituency", ""),
            district=r.get("district", ""),
            predicted=predicted,
            confidence=_to_float(r.get("confidence", shares.get(predicted, 0.0))),
            shares=shares,
        ))
    return out


def _load_assembly_2021_per_ac() -> Dict[int, dict]:
    """Per-AC 2021 winner / runner-up indexed by ac_no."""
    path = Path(DATASET_DIR) / "data_files" / "tamilnadu_assembly_2021_results.csv"
    out: Dict[int, dict] = {}
    for r in _read_csv(path):
        ac_no = _to_int(r.get("ac_no", 0))
        if not ac_no:
            continue
        out[ac_no] = r
    return out


def _load_assembly_2016_per_ac() -> Dict[int, dict]:
    path = Path(DATASET_DIR) / "data_files" / "tamilnadu_assembly_2016_results.csv"
    out: Dict[int, dict] = {}
    for r in _read_csv(path):
        ac_no = _to_int(r.get("ac_no", 0))
        if not ac_no:
            continue
        out[ac_no] = r
    return out


def _load_vote_shares_per_ac() -> Dict[int, Dict[str, float]]:
    """
    Per-AC underlying vote shares (vs_*) from predictions_2026.csv. These
    are the model's vote-share estimates (typically DMK ~40%, AIADMK ~39%,
    TVK ~11%) -- much less polarized than the softmax winner probability,
    so they're used for the live-mode per-row calculation where modest
    sentiment shifts need room to flip a seat.
    """
    path = Path(DATASET_DIR) / "predictions" / "predictions_2026.csv"
    out: Dict[int, Dict[str, float]] = {}
    for r in _read_csv(path):
        ac_no = _to_int(r.get("ac_no", 0))
        if not ac_no:
            continue
        out[ac_no] = {
            p: _to_float(r.get(f"vs_{p}", 0.0)) for p in PARTIES
        }
    return out


# ---------------------------------------------------------------------------
# 1. Long-term trend (2016 -> 2021 -> 2026 + Lok Sabha context)
# ---------------------------------------------------------------------------

def compute_long_term_trend() -> Tuple[List[dict], dict]:
    base = _load_prediction_base(LONG_TERM_DIR)
    asm_2016 = _alliance_share_table(_read_csv(LONG_TERM_DIR / "tamilnadu_assembly_2016.csv"))
    asm_2021 = _alliance_share_table(_read_csv(LONG_TERM_DIR / "tamilnadu_assembly_2021.csv"))
    seats_2016 = _alliance_seat_table(_read_csv(LONG_TERM_DIR / "tamilnadu_assembly_2016.csv"))
    seats_2021 = _alliance_seat_table(_read_csv(LONG_TERM_DIR / "tamilnadu_assembly_2021.csv"))

    ls_2014 = _alliance_share_table(_read_csv(LONG_TERM_DIR / "tamilnadu_lok_sabha_2014.csv"))
    ls_2019 = _alliance_share_table(_read_csv(LONG_TERM_DIR / "tamilnadu_lok_sabha_2019.csv"))
    ls_2024 = _alliance_share_table(_read_csv(LONG_TERM_DIR / "tamilnadu_lok_sabha_2024.csv"))

    asm_2026 = _state_share_from_base(base)

    party_growth: Dict[str, float] = {}
    vote_share_trend: Dict[str, List[float]] = {}
    seat_swing_trend: Dict[str, Dict[str, int]] = {}
    long_term_party_score: Dict[str, float] = {}

    for p in PARTIES:
        a16, a21, a26 = asm_2016.get(p, 0.0), asm_2021.get(p, 0.0), asm_2026.get(p, 0.0)
        l14, l19, l24 = ls_2014.get(p, 0.0), ls_2019.get(p, 0.0), ls_2024.get(p, 0.0)
        # growth = (now - then) / then, blended assembly + lok sabha
        growth_asm = ((a26 - a16) / a16) if a16 else (a26 / 100.0)
        growth_lok = ((l24 - l14) / l14) if l14 else (l24 / 100.0)
        growth = 0.6 * growth_asm + 0.4 * growth_lok
        party_growth[p] = round(growth, 4)

        vote_share_trend[p] = [round(a16, 2), round(a21, 2), round(a26, 2)]
        seat_swing_trend[p] = {
            "2016": seats_2016.get(p, 0),
            "2021": seats_2021.get(p, 0),
            "2026_predicted": _seats_from_base(base, p),
        }

        # Long-term party trend score in [0, 1]:
        # assembly average weight 0.5 + lok sabha average weight 0.3 + recency 0.2
        avg_asm = (a16 + a21 + a26) / 3.0 / 100.0
        avg_lok = (l14 + l19 + l24) / 3.0 / 100.0
        recency = a26 / 100.0
        long_term_party_score[p] = round(
            min(1.0, max(0.0, 0.5 * avg_asm + 0.3 * avg_lok + 0.2 * recency)),
            4,
        )

    # Per-constituency historical strength: ratio of times the predicted
    # alliance won the seat over the last two cycles.
    asm_2016_per_ac = _load_assembly_2016_per_ac()
    asm_2021_per_ac = _load_assembly_2021_per_ac()

    rows: List[dict] = []
    for r in base:
        prev_2016 = (asm_2016_per_ac.get(r.ac_no) or {}).get("winner_alliance", "")
        prev_2021 = (asm_2021_per_ac.get(r.ac_no) or {}).get("winner_alliance", "")
        wins = sum(1 for a in (prev_2016, prev_2021) if a == r.predicted)
        historical_strength = round(wins / 2.0, 2)

        # Long-term trend score blends party-level score with the
        # constituency's historical strength.
        lts = long_term_party_score.get(r.predicted, 0.0)
        long_term_trend_score = round(
            min(1.0, max(0.0, 0.6 * lts + 0.4 * historical_strength)),
            4,
        )

        # Per-row long-term winner: weighted between current model shares,
        # per-party long-term strength, and a per-AC historical bonus for
        # parties that won this seat in 2016 or 2021.
        per_row_long_score: Dict[str, float] = {}
        for p in PARTIES:
            historical_bonus = 0.0
            if prev_2016 == p:
                historical_bonus += 0.45
            if prev_2021 == p:
                historical_bonus += 0.65
            per_row_long_score[p] = (
                0.40 * r.shares.get(p, 0.0)
                + 0.30 * long_term_party_score.get(p, 0.0)
                + 0.30 * historical_bonus
            )
        analysis_predicted = _pick_winner(per_row_long_score)

        rows.append({
            **_base_row_to_dict(r),
            "analysis_predicted": analysis_predicted,
            "long_term_trend_score": long_term_trend_score,
            "historical_strength": historical_strength,
            "winner_2016": prev_2016,
            "winner_2021": prev_2021,
            "party_growth_score": party_growth.get(r.predicted, 0.0),
            "vote_share_trend_for_predicted": vote_share_trend.get(r.predicted, []),
        })

    meta = {
        "compared_years": [2016, 2021, 2026],
        "lok_sabha_reference_years": [2014, 2019, 2024],
        "long_term_party_trend_score": long_term_party_score,
        "party_growth_score": party_growth,
        "vote_share_trend": vote_share_trend,
        "seat_swing_trend": seat_swing_trend,
        "assembly_state_share": {
            "2016": _round_dict(asm_2016, 2),
            "2021": _round_dict(asm_2021, 2),
            "2026_predicted": _round_dict(asm_2026, 2),
        },
        "lok_sabha_state_share": {
            "2014": _round_dict(ls_2014, 2),
            "2019": _round_dict(ls_2019, 2),
            "2024": _round_dict(ls_2024, 2),
        },
    }
    return rows, meta


# ---------------------------------------------------------------------------
# 2. Recent swing (2021 -> 2026 with 2024 Lok Sabha overlay)
# ---------------------------------------------------------------------------

def compute_recent_swing() -> Tuple[List[dict], dict]:
    base = _load_prediction_base(RECENT_SWING_DIR)
    asm_2021 = _alliance_share_table(_read_csv(RECENT_SWING_DIR / "tamilnadu_assembly_2021.csv"))
    seats_2021 = _alliance_seat_table(_read_csv(RECENT_SWING_DIR / "tamilnadu_assembly_2021.csv"))
    ls_2024 = _alliance_share_table(_read_csv(RECENT_SWING_DIR / "tamilnadu_lok_sabha_2024.csv"))
    asm_2026 = _state_share_from_base(base)

    asm_2021_per_ac = _load_assembly_2021_per_ac()

    # Party-wise momentum: blends assembly delta + Lok Sabha context.
    recent_swing_party_score: Dict[str, float] = {}
    party_recent_momentum: Dict[str, float] = {}
    for p in PARTIES:
        a21, a26 = asm_2021.get(p, 0.0), asm_2026.get(p, 0.0)
        l24 = ls_2024.get(p, 0.0)
        delta_assembly = (a26 - a21) / 100.0
        delta_lok = (l24 - a21) / 100.0
        momentum = 0.7 * delta_assembly + 0.3 * delta_lok
        party_recent_momentum[p] = round(momentum, 4)
        # Score in [0, 1] - rebase momentum onto a 0..1 axis around 0.5.
        recent_swing_party_score[p] = round(
            min(1.0, max(0.0, 0.5 + momentum)),
            4,
        )

    rows: List[dict] = []
    for r in base:
        prev = asm_2021_per_ac.get(r.ac_no) or {}
        winner_2021 = prev.get("winner_alliance", "")
        runner_up_2021 = prev.get("runner_up_alliance", "")
        winner_party_2021 = prev.get("winner_party", "")
        runner_up_party_2021 = prev.get("runner_up_party", "")
        winner_votes = _to_float(prev.get("winner_votes", 0))
        runner_up_votes = _to_float(prev.get("runner_up_votes", 0))
        total_votes = _to_float(prev.get("total_valid_votes", 0))
        winning_margin_2021 = (
            round(((winner_votes - runner_up_votes) / total_votes), 4)
            if total_votes > 0 else 0.0
        )
        vote_share_2021 = (
            round((winner_votes / total_votes), 4) if total_votes > 0 else 0.0
        )

        incumbency_status = "incumbent_holds" if winner_2021 == r.predicted else "anti_incumbency"

        # Constituency-level swing = (predicted-party 2026 share)
        #                          - (predicted-party state-level 2021 share / 100)
        baseline_2021_share = asm_2021.get(r.predicted, 0.0) / 100.0
        constituency_swing = round(r.shares.get(r.predicted, 0.0) - baseline_2021_share, 4)

        # Seat retention probability: confidence if winner_2021 == predicted,
        # otherwise 1 - confidence (interpreted as opposition gain).
        if winner_2021 == r.predicted:
            seat_retention_prob = round(r.confidence, 4)
            opposition_gain_prob = round(1.0 - r.confidence, 4)
        else:
            seat_retention_prob = round(1.0 - r.confidence, 4)
            opposition_gain_prob = round(r.confidence, 4)

        recent_swing_score = round(
            min(1.0, max(0.0,
                0.55 * recent_swing_party_score.get(r.predicted, 0.5)
                + 0.25 * r.confidence
                + 0.20 * (1.0 if winner_2021 == r.predicted else 0.4)
            )),
            4,
        )

        # Per-row recent-swing winner: weighted between current model
        # shares, per-party momentum (2021 -> 2026 + 2024 LS), and a
        # 2021-incumbent anchor. Negative momentum erodes the incumbent.
        per_row_swing_score: Dict[str, float] = {}
        for p in PARTIES:
            incumbency_bonus = 0.55 if winner_2021 == p else 0.0
            momentum = party_recent_momentum.get(p, 0.0)
            per_row_swing_score[p] = (
                0.35 * r.shares.get(p, 0.0)
                + 0.30 * recent_swing_party_score.get(p, 0.0)
                + 0.25 * incumbency_bonus
                + 0.10 * momentum
            )
        analysis_predicted = _pick_winner(per_row_swing_score)

        rows.append({
            **_base_row_to_dict(r),
            "analysis_predicted": analysis_predicted,
            "recent_swing_score": recent_swing_score,
            "winner_2021": winner_2021,
            "runner_up_2021": runner_up_2021,
            "winner_party_2021": winner_party_2021,
            "runner_up_party_2021": runner_up_party_2021,
            "winning_margin_2021": winning_margin_2021,
            "vote_share_2021": vote_share_2021,
            "incumbency_status": incumbency_status,
            "constituency_swing": constituency_swing,
            "seat_retention_probability": seat_retention_prob,
            "opposition_gain_probability": opposition_gain_prob,
        })

    meta = {
        "compared_years": [2021, 2026],
        "lok_sabha_reference_year": 2024,
        "recent_swing_party_score": recent_swing_party_score,
        "party_recent_momentum": party_recent_momentum,
        "assembly_state_share": {
            "2021": _round_dict(asm_2021, 2),
            "2026_predicted": _round_dict(asm_2026, 2),
        },
        "lok_sabha_state_share_2024": _round_dict(ls_2024, 2),
        "seats_2021": seats_2021,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# 3. Live intelligence score (current sentiment-only)
# ---------------------------------------------------------------------------

def _read_score_table(path: Path, score_col: str) -> Dict[str, float]:
    out = {p: 0.0 for p in PARTIES}
    for r in _read_csv(path):
        party = r.get("party", "")
        if party in out:
            out[party] = _to_float(r.get(score_col, 0))
    return out


def compute_live_intelligence_score() -> Tuple[List[dict], dict]:
    base = _load_prediction_base(LIVE_INTEL_DIR.parent / "long_term_trend")  # reuse base
    party_sent = _read_score_table(LIVE_INTEL_DIR / "party_sentiment_2026.csv",
                                   "party_sentiment_score")
    leader_rows = _read_csv(LIVE_INTEL_DIR / "leader_sentiment_2026.csv")
    leader_sent: Dict[str, float] = {p: 0.0 for p in PARTIES}
    leader_names: Dict[str, str] = {p: "" for p in PARTIES}
    for r in leader_rows:
        party = r.get("party", "")
        if party in leader_sent:
            leader_sent[party] = _to_float(r.get("leader_sentiment_score", 0))
            leader_names[party] = r.get("leader", "")
    candidate_sent = _read_score_table(LIVE_INTEL_DIR / "candidate_sentiment_2026.csv",
                                       "candidate_sentiment_score")
    social_sent = _read_score_table(LIVE_INTEL_DIR / "social_media_sentiment_2026.csv",
                                    "social_media_sentiment_score")
    news_sent = _read_score_table(LIVE_INTEL_DIR / "news_sentiment_2026.csv",
                                  "news_sentiment_score")

    issue_rows = _read_csv(LIVE_INTEL_DIR / "local_issue_score_2026.csv")
    local_issue_party_score: Dict[str, float] = {p: 0.0 for p in PARTIES}
    for r in issue_rows:
        party = r.get("favoured_party", "")
        if party in local_issue_party_score:
            local_issue_party_score[party] += _to_float(r.get("issue_score", 0))
    # Normalise so the largest sits in [0, 1].
    max_issue = max(local_issue_party_score.values()) or 1.0
    local_issue_party_score = {
        p: round(local_issue_party_score[p] / max_issue, 4) for p in PARTIES
    }

    tvk_rows = _read_csv(LIVE_INTEL_DIR / "tvk_impact_2026.csv")
    tvk_impact: Dict[str, float] = {}
    for r in tvk_rows:
        tvk_impact[r.get("metric", "")] = _to_float(r.get("value", 0))
    tvk_impact_score = round(
        0.4 * tvk_impact.get("vijay_personal_rating", 0)
        + 0.3 * tvk_impact.get("youth_pull_score", 0)
        + 0.3 * tvk_impact.get("tvk_state_buzz_score", 0),
        4,
    )

    party_live_score: Dict[str, float] = {}
    for p in PARTIES:
        # Live mode weights "current momentum" signals (leader buzz, social
        # virality, TVK impact) much heavier than slow-moving party-brand /
        # candidate / news / issue signals. Encodes that "live now"
        # sentiment != entrenched party loyalty.
        score = (
            0.10 * party_sent.get(p, 0.0)
            + 0.30 * leader_sent.get(p, 0.0)
            + 0.10 * candidate_sent.get(p, 0.0)
            + 0.30 * social_sent.get(p, 0.0)
            + 0.10 * news_sent.get(p, 0.0)
            + 0.05 * local_issue_party_score.get(p, 0.0)
            + (0.10 * tvk_impact_score if p == "TVK" else 0.0)
        )
        party_live_score[p] = round(min(1.0, max(0.0, score)), 4)

    # Confidence level bands for the live intel layer.
    def _confidence_band(score: float) -> str:
        if score >= 0.65:
            return "High"
        if score >= 0.50:
            return "Medium-High"
        if score >= 0.35:
            return "Medium"
        if score >= 0.20:
            return "Low-Medium"
        return "Low"

    # Underlying vote shares (vs_*) -- realistic 0.40/0.39/0.11 splits
    # rather than the polarized softmax winner probabilities. Live mode
    # blends these with sentiment so TVK's leader/social momentum can
    # flip competitive seats without wiping out AIADMK strongholds.
    vote_shares = _load_vote_shares_per_ac()

    rows: List[dict] = []
    for r in base:
        live = party_live_score.get(r.predicted, 0.0)
        # Sentiment-adjusted prediction: blend model confidence with live score.
        adjusted = round(0.7 * r.confidence + 0.3 * live, 4)

        # Per-row live-intelligence winner. Uses the realistic vote-share
        # vector (vs_*) instead of the model's polarized winner-probability
        # so the sentiment + TVK-impact signals have room to shift outcomes
        # in competitive seats.
        vs = vote_shares.get(r.ac_no, r.shares)
        per_row_live_score: Dict[str, float] = {}
        for p in PARTIES:
            tvk_boost = tvk_impact_score * 0.18 if p == "TVK" else 0.0
            per_row_live_score[p] = (
                0.50 * vs.get(p, 0.0)
                + 0.35 * party_live_score.get(p, 0.0)
                + tvk_boost
            )
        analysis_predicted = _pick_winner(per_row_live_score)

        rows.append({
            **_base_row_to_dict(r),
            "analysis_predicted": analysis_predicted,
            "live_intelligence_score": live,
            "sentiment_adjusted_prediction": adjusted,
            "confidence_level": _confidence_band(live),
            "tvk_impact_score": tvk_impact_score,
        })

    meta = {
        "reference_year": TARGET_ASSEMBLY_YEAR,
        "party_sentiment_score": party_sent,
        "leader_sentiment_score": leader_sent,
        "leader_names": leader_names,
        "candidate_sentiment_score": candidate_sent,
        "social_media_sentiment_score": social_sent,
        "news_sentiment_score": news_sent,
        "local_issue_party_score": local_issue_party_score,
        "tvk_impact_metrics": tvk_impact,
        "tvk_impact_score": tvk_impact_score,
        "party_live_intelligence_score": party_live_score,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Final-score blend
# ---------------------------------------------------------------------------

def compute_final_prediction_score() -> Tuple[List[dict], dict]:
    """
    Per-constituency blend of all three analysis-typed scores. Used as the
    "everything-on" payload for the UI.
    """
    long_rows, long_meta = compute_long_term_trend()
    swing_rows, swing_meta = compute_recent_swing()
    live_rows, live_meta = compute_live_intelligence_score()

    long_index = {r["ac_no"]: r for r in long_rows}
    swing_index = {r["ac_no"]: r for r in swing_rows}
    live_index = {r["ac_no"]: r for r in live_rows}

    rows: List[dict] = []
    for ac_no, lr in long_index.items():
        sr = swing_index.get(ac_no, {})
        ir = live_index.get(ac_no, {})
        long_score = lr.get("long_term_trend_score", 0.0)
        swing_score = sr.get("recent_swing_score", 0.0)
        live_score = ir.get("live_intelligence_score", 0.0)
        final_score = round(
            FINAL_SCORE_WEIGHTS["long_term_trend"] * long_score
            + FINAL_SCORE_WEIGHTS["recent_swing"] * swing_score
            + FINAL_SCORE_WEIGHTS["live_intelligence_score"] * live_score,
            4,
        )
        merged = {**lr, **sr, **ir, "final_prediction_score": final_score}
        rows.append(merged)

    meta = {
        "weights": FINAL_SCORE_WEIGHTS,
        "long_term_trend": long_meta,
        "recent_swing": swing_meta,
        "live_intelligence_score": live_meta,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Top-level dispatch + analysis context
# ---------------------------------------------------------------------------

def _gap_category(gap_years: int) -> str:
    if gap_years <= 1:
        return "Same Cycle"
    if gap_years <= 5:
        return "Short Gap"
    if gap_years <= 10:
        return "Medium Gap"
    return "Long Gap"


def build_analysis_context(analysis_type: str) -> dict:
    """
    Universal context block surfaced alongside every analysis-typed
    response. Mirrors what the frontend wants to display in the header.
    """
    if analysis_type == "long_term_trend":
        cm_year = TARGET_ASSEMBLY_YEAR
        ref_years = list(ASSEMBLY_TO_LOKSABHA.values())  # 2014, 2019, 2024
        gap = max(ref_years) - min(ref_years)
    elif analysis_type == "recent_swing":
        cm_year = TARGET_ASSEMBLY_YEAR
        ref_years = [TARGET_LOK_SABHA_YEAR]
        gap = TARGET_ASSEMBLY_YEAR - TARGET_LOK_SABHA_YEAR
    else:  # live_intelligence_score
        cm_year = TARGET_ASSEMBLY_YEAR
        ref_years = [TARGET_ASSEMBLY_YEAR]
        gap = 0

    return {
        "analysis_type": analysis_type,
        "cm_election_year": cm_year,
        "lok_sabha_reference_years": ref_years,
        "gap_years": gap,
        "gap_category": _gap_category(gap),
        "prediction_mode": True,
    }


_NON_TVK_PARTIES = ("DMK_ALLIANCE", "AIADMK_NDA", "NTK", "OTHERS")


def _suppress_tvk_in_row(row: dict) -> None:
    """
    TVK was founded in 2024 and has no signal in the historical actuals
    (2011/2016/2021) or the 2024 Lok Sabha tables. The long-term-trend and
    recent-swing views therefore must not produce TVK winners; only the
    live-intelligence view (which carries actual TVK sentiment) is allowed
    to. This mutator re-picks both the model winner and the analysis winner
    from the four non-TVK alliances when either is TVK. The TVK probability
    column itself is left intact for transparency.
    """
    def best_non_tvk():
        return max(_NON_TVK_PARTIES, key=lambda p: row.get(p, 0.0))

    if row.get("predicted") == "TVK":
        new_pred = best_non_tvk()
        row["predicted"] = new_pred
        row["confidence"] = row.get(new_pred, 0.0)
    if row.get("analysis_predicted") == "TVK":
        row["analysis_predicted"] = best_non_tvk()


def run_analysis(analysis_type: str) -> Tuple[List[dict], dict]:
    """
    Public entry point. Returns (rows, meta) for the requested analysis
    type. The default (no analysis_type) caller path is handled in
    server.py and does NOT invoke this module.
    """
    if analysis_type == "long_term_trend":
        rows, type_meta = compute_long_term_trend()
    elif analysis_type == "recent_swing":
        rows, type_meta = compute_recent_swing()
    elif analysis_type == "live_intelligence_score":
        rows, type_meta = compute_live_intelligence_score()
    else:
        raise ValueError(
            f"Unknown analysis_type '{analysis_type}'. "
            f"Expected one of: {', '.join(ANALYSIS_TYPES)}."
        )

    if analysis_type != "live_intelligence_score":
        for r in rows:
            _suppress_tvk_in_row(r)

    # Always blend in the final prediction score so the UI can render a
    # single "Final Prediction Score" column regardless of which analysis
    # tab is active.
    final_rows, final_meta = compute_final_prediction_score()
    final_index = {r["ac_no"]: r["final_prediction_score"] for r in final_rows}

    enriched: List[dict] = []
    score_key = {
        "long_term_trend": "long_term_trend_score",
        "recent_swing": "recent_swing_score",
        "live_intelligence_score": "live_intelligence_score",
    }[analysis_type]

    party_win_counts: Dict[str, int] = {p: 0 for p in PARTIES}
    analysis_seat_counts: Dict[str, int] = {p: 0 for p in PARTIES}
    party_score_sum: Dict[str, float] = {p: 0.0 for p in PARTIES}
    party_score_count: Dict[str, int] = {p: 0 for p in PARTIES}
    confidence_buckets: Dict[str, int] = {"High": 0, "Medium-High": 0, "Medium": 0, "Low-Medium": 0, "Low": 0}

    for row in rows:
        final_score = final_index.get(row["ac_no"], 0.0)
        enriched_row = {
            **row,
            "final_prediction_score": final_score,
            "win_probability": row.get("confidence", 0.0),
        }
        # Confidence band derived from the analysis-specific score.
        score = row.get(score_key, 0.0)
        if score >= 0.65:
            band = "High"
        elif score >= 0.50:
            band = "Medium-High"
        elif score >= 0.35:
            band = "Medium"
        elif score >= 0.20:
            band = "Low-Medium"
        else:
            band = "Low"
        enriched_row.setdefault("confidence_level", band)
        confidence_buckets[band] = confidence_buckets.get(band, 0) + 1

        predicted = row["predicted"]
        if predicted in party_win_counts:
            party_win_counts[predicted] += 1
            party_score_sum[predicted] += score
            party_score_count[predicted] += 1

        analysis_predicted = row.get("analysis_predicted", predicted)
        if analysis_predicted in analysis_seat_counts:
            analysis_seat_counts[analysis_predicted] += 1

        enriched.append(enriched_row)

    party_avg_score = {
        p: round(party_score_sum[p] / party_score_count[p], 4) if party_score_count[p] else 0.0
        for p in PARTIES
    }

    state_share = _state_share_from_dict_rows(enriched)

    response_meta = {
        **build_analysis_context(analysis_type),
        "weights": FINAL_SCORE_WEIGHTS,
        "party_seat_counts": party_win_counts,
        "analysis_seat_counts": analysis_seat_counts,
        "party_average_score": party_avg_score,
        "party_state_share_2026": _round_dict(state_share, 2),
        "confidence_buckets": confidence_buckets,
        "analysis_specific": type_meta,
    }
    return enriched, response_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_share_from_base(base: List[PredictionBaseRow]) -> Dict[str, float]:
    if not base:
        return {p: 0.0 for p in PARTIES}
    totals = {p: 0.0 for p in PARTIES}
    for r in base:
        for p in PARTIES:
            totals[p] += r.shares.get(p, 0.0)
    n = float(len(base))
    # Convert mean fraction to percent so it lines up with the historical
    # CSVs which store vote_share as percent.
    return {p: round((totals[p] / n) * 100.0, 4) for p in PARTIES}


def _state_share_from_dict_rows(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {p: 0.0 for p in PARTIES}
    totals = {p: 0.0 for p in PARTIES}
    for r in rows:
        for p in PARTIES:
            totals[p] += _to_float(r.get(p, 0))
    n = float(len(rows))
    return {p: round((totals[p] / n) * 100.0, 4) for p in PARTIES}


def _seats_from_base(base: List[PredictionBaseRow], party: str) -> int:
    return sum(1 for r in base if r.predicted == party)


def _round_dict(d: Dict[str, float], places: int) -> Dict[str, float]:
    return {k: round(float(v), places) for k, v in d.items()}


def _base_row_to_dict(r: PredictionBaseRow) -> dict:
    return {
        "ac_no": r.ac_no,
        "constituency": r.constituency,
        "district": r.district,
        "predicted": r.predicted,
        "confidence": r.confidence,
        "DMK_ALLIANCE": r.shares["DMK_ALLIANCE"],
        "AIADMK_NDA":   r.shares["AIADMK_NDA"],
        "TVK":          r.shares["TVK"],
        "NTK":          r.shares["NTK"],
        "OTHERS":       r.shares["OTHERS"],
    }
