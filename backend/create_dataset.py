"""
Tamil Nadu 2026 projection engine.

Flow:
  1. Load the constituency master (234 ACs) and the REAL per-AC 2016/2021
     results from backend/data_files/.
  2. Load state-level 2014/2019/2024 Lok Sabha + 2016/2021 Assembly vote
     shares to compute alliance-level swing signals.
  3. For each AC, blend:
        - real 2021 winner/runner-up base share  (primary signal)
        - 2024 Lok Sabha state swing          (momentum signal)
        - alliance seat-sharing coverage      (2026 supply signal)
        - TVK/NTK-driven share transfer       (new-entrant signal)
        - district/region/reservation factors (structural signal)
     to produce proj_2026 vote-share targets per AC and a projected winner.
  4. Write backend/data_files/tamilnadu_assembly_2026.csv -- consumed by
     data_loader.load_training_dataframe().

This module is cleanly separable from the model. A curated
tamilnadu_assembly_2026.csv can replace the generated CSV later without
touching train.py or data_loader.py.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from config import PARTIES  # ["DMK_ALLIANCE","AIADMK_NDA","TVK","OTHERS"]

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BACKEND_DIR, "data_files")

# ---------------------------------------------------------------------------
# Historical alliance mappings per year. Parties move between alliances, so
# these are year-specific (not the same as config.PARTY_TO_ALLIANCE_2026).
# ---------------------------------------------------------------------------

ASSEMBLY_ALLIANCE_BY_YEAR: Dict[int, Dict[str, str]] = {
    2016: {
        "AIADMK": "AIADMK_NDA", "PMK": "AIADMK_NDA", "DMDK": "AIADMK_NDA",
        "BJP": "OTHERS",  # BJP contested solo in 2016
        "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "Congress": "DMK_ALLIANCE",
        "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE", "VCK": "DMK_ALLIANCE",
        "IUML": "DMK_ALLIANCE", "MDMK": "DMK_ALLIANCE",
        "NTK": "NTK",
    },
    2021: {
        "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "Congress": "DMK_ALLIANCE",
        "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE", "VCK": "DMK_ALLIANCE",
        "MDMK": "DMK_ALLIANCE", "IUML": "DMK_ALLIANCE",
        "AIADMK": "AIADMK_NDA", "BJP": "AIADMK_NDA", "PMK": "AIADMK_NDA",
        "NTK": "NTK",
        "AMMK": "OTHERS", "DMDK": "OTHERS", "MNM": "OTHERS",
    },
}

# Floors prevent degenerate 0% rows that wreck softmax targets.
ALLIANCE_FLOOR = {
    "DMK_ALLIANCE": 0.04,
    "AIADMK_NDA":   0.04,
    "TVK":          0.005,
    "NTK":          0.005,
    "OTHERS":       0.005,
}

# Projection weight for 2021->2026 momentum from state-level LS24 swing.
ALPHA_MOMENTUM = 0.22

# TVK draws disproportionately from AIADMK_NDA in early polling.
TVK_DRAW_FROM = {"AIADMK_NDA": 0.55, "DMK_ALLIANCE": 0.30, "OTHERS": 0.15}

# NTK (Seeman) is a Tamil-nationalist standalone; it draws most from
# anti-incumbency-oriented youth and from DMK_ALLIANCE's left-flank.
NTK_DRAW_FROM = {"DMK_ALLIANCE": 0.55, "AIADMK_NDA": 0.25, "OTHERS": 0.20}

# Regional TVK penetration envelope (higher in metro/urban belts). These are
# baseline TVK shares, not win probabilities.
TVK_REGION_ENVELOPE = {
    "Chennai_Metro": 0.12,
    "North":         0.08,
    "West":          0.09,
    "Central":       0.05,
    "South":         0.04,
}

# NTK historically polls ~3-4% statewide; higher in South and North.
NTK_REGION_ENVELOPE = {
    "Chennai_Metro": 0.03,
    "North":         0.04,
    "West":          0.025,
    "Central":       0.03,
    "South":         0.045,
}

# Wildcard injection rates -- fraction of ACs in each region that get a
# stochastic TVK / NTK / OTHERS surge large enough to potentially flip the
# seat. Tuned so the training targets project ~3-8 TVK wins, 0-2 NTK wins,
# and ~1-3 OTHERS wins statewide, consistent with early 2026 polling.
TVK_WILDCARD_RATE = {
    "Chennai_Metro": 0.04,
    "North":         0.02,
    "West":          0.02,
    "Central":       0.01,
    "South":         0.01,
}
NTK_WILDCARD_RATE = 0.004   # NTK has never won a TN AC seat
OTHERS_WILDCARD_RATE = 0.012


def _read(name: str) -> pd.DataFrame:
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}. "
                                f"Run build_data_files.py first.")
    return pd.read_csv(path)


def _alliance_state_shares(df: pd.DataFrame, year: int) -> Dict[str, float]:
    """Collapse party-level state election table into alliance-level shares."""
    mapping = ASSEMBLY_ALLIANCE_BY_YEAR.get(year, {})
    out = {p: 0.0 for p in PARTIES}
    for _, row in df.iterrows():
        party = str(row["party"]).strip()
        alliance = mapping.get(party, "OTHERS")
        out[alliance] += float(row.get("vote_share", 0.0)) / 100.0
    if sum(out.values()) < 0.5:
        raise ValueError(f"Alliance share roll-up looks broken for year {year}: {out}")
    return out


def _ls_alliance_shares(df: pd.DataFrame) -> Dict[str, float]:
    mapping = {
        "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "Congress": "DMK_ALLIANCE",
        "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE",
        "VCK": "DMK_ALLIANCE", "MDMK": "DMK_ALLIANCE", "IUML": "DMK_ALLIANCE",
        "AIADMK": "AIADMK_NDA", "BJP": "AIADMK_NDA",
        "PMK": "AIADMK_NDA", "DMDK": "AIADMK_NDA",
        "NTK": "NTK",
        "MNM": "OTHERS", "AMMK": "OTHERS",
    }
    out = {p: 0.0 for p in PARTIES}
    for _, row in df.iterrows():
        alliance = mapping.get(str(row["party"]).strip(), "OTHERS")
        out[alliance] += float(row.get("vote_share", 0.0)) / 100.0
    return out


def _compute_alliance_supply() -> Dict[str, float]:
    """Fraction of 234 seats each alliance is fielding in 2026."""
    dmk = _read("tamilnadu_DMK_ALLIANCE_seat_sharing_2026.csv")
    nda = _read("tamilnadu_AIADMK_NDA_seat_sharing_2026.csv")
    return {
        "DMK_ALLIANCE": min(1.0, float(dmk["seats"].sum()) / 234.0),
        "AIADMK_NDA":   min(1.0, float(nda["seats"].sum()) / 234.0),
        "TVK":          1.0,   # TVK announced full 234
        "NTK":          1.0,   # NTK announced full 234
        "OTHERS":       1.0,   # independents / rebels cover all seats
    }


def _project_shares_for_row(
    row: pd.Series,
    state_ls24: Dict[str, float],
    state_as21: Dict[str, float],
    supply: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Project vote shares for one AC. All intermediate values are in [0,1].

    The strong prior is the 2021 result *at this specific AC* -- a seat
    AIADMK won in 2021 gets AIADMK as its base, even if the DMK alliance
    won more seats overall. State-level swing is applied as perturbation,
    not as the base.
    """
    winner21_all = row["winner_alliance_2021"]
    runner21_all = row["runner_up_alliance_2021"]

    # AC-identity prior: winner=0.44, runner-up=0.36, residual split among
    # the other alliances. Actual 2021 avg margin in TN was ~6-10% per seat.
    base = {p: 0.0 for p in PARTIES}
    base[winner21_all] = 0.44
    base[runner21_all] = 0.36
    residual = 1.0 - base[winner21_all] - base[runner21_all]
    other_parties = [p for p in PARTIES if p not in (winner21_all, runner21_all)]
    for p in other_parties:
        base[p] = residual / len(other_parties)

    # If the AC has a real measured 2021 margin, narrow the gap for close
    # seats (smaller margin -> closer base shares).
    real_margin = float(row.get("margin_pct_2021") or 0.0)
    if real_margin > 0:
        gap = max(0.0, 0.08 - real_margin) * 0.5
        base[winner21_all] -= gap
        base[runner21_all] += gap

    # State-level swing (2021 AS -> 2024 LS) as a light perturbation. This
    # captures anti-incumbency without drowning out AC identity.
    momentum = {p: state_ls24.get(p, 0.0) - state_as21.get(p, 0.0) for p in PARTIES}
    proj = {p: base[p] + ALPHA_MOMENTUM * momentum[p] for p in PARTIES}

    # Reserved seats historically lean DMK_ALLIANCE in TN.
    if int(row["is_reserved"]) == 1:
        proj["DMK_ALLIANCE"] += 0.015
        proj["AIADMK_NDA"]   -= 0.010
        proj["OTHERS"]       -= 0.005

    region = str(row["region_5way"])
    if region == "Chennai_Metro":
        proj["DMK_ALLIANCE"] += 0.010
        proj["AIADMK_NDA"]   -= 0.005
    elif region == "West":
        proj["AIADMK_NDA"]   += 0.010
        proj["DMK_ALLIANCE"] -= 0.005

    # TVK injection: takes region-scaled share primarily from AIADMK_NDA.
    tvk_share = TVK_REGION_ENVELOPE.get(region, 0.05)
    # Stochastic wildcard: a small fraction of ACs in each region get a
    # large TVK surge (modelling unknown strong local candidates). This
    # lets the model see TVK wins in training instead of learning that
    # TVK never wins.
    if rng.random() < TVK_WILDCARD_RATE.get(region, 0.05):
        tvk_share = min(0.55, tvk_share + rng.uniform(0.25, 0.40))
    proj["TVK"] = tvk_share
    for donor, frac in TVK_DRAW_FROM.items():
        proj[donor] = max(0.0, proj[donor] - tvk_share * frac)

    # NTK injection: baseline Tamil-nationalist vote bank, region-scaled.
    # NTK has never won a TN AC, so wildcard rate is deliberately tiny --
    # only a handful of seats should show NTK-winning scenarios.
    ntk_share = NTK_REGION_ENVELOPE.get(region, 0.03)
    if rng.random() < NTK_WILDCARD_RATE:
        ntk_share = min(0.45, ntk_share + rng.uniform(0.22, 0.35))
    proj["NTK"] = ntk_share
    for donor, frac in NTK_DRAW_FROM.items():
        proj[donor] = max(0.0, proj[donor] - ntk_share * frac)

    # OTHERS wildcard: strong independent/rebel performance at a few ACs.
    if rng.random() < OTHERS_WILDCARD_RATE:
        others_surge = rng.uniform(0.20, 0.38)
        proj["OTHERS"] = max(proj["OTHERS"], others_surge)
        # Donors: split evenly between the two main alliances.
        deduction = others_surge * 0.5
        proj["DMK_ALLIANCE"] = max(0.0, proj["DMK_ALLIANCE"] - deduction)
        proj["AIADMK_NDA"]   = max(0.0, proj["AIADMK_NDA"]   - deduction)

    # Alliance supply cap.
    for p in PARTIES:
        proj[p] *= supply.get(p, 1.0)

    # Per-AC noise so the model isn't learning a strictly linear function.
    for p in PARTIES:
        proj[p] += rng.normal(0.0, 0.012)
        proj[p] = max(ALLIANCE_FLOOR[p], proj[p])

    total = sum(proj.values())
    return {p: v / total for p, v in proj.items()}


def build_assembly_2026(seed: int = 42) -> pd.DataFrame:
    """Produce the per-AC 2026 projection CSV consumed by data_loader."""
    rng = np.random.default_rng(seed)

    master = _read("tamilnadu_constituency_master_2026.csv")
    res16  = _read("tamilnadu_assembly_2016_results.csv")
    res21  = _read("tamilnadu_assembly_2021_results.csv")

    as21 = _alliance_state_shares(_read("tamilnadu_assembly_election_2021.csv"), 2021)
    ls24 = _ls_alliance_shares(_read("tamilnadu_lok_sabha_election_2024.csv"))
    supply = _compute_alliance_supply()

    hist16 = res16[["ac_no", "winner_party", "winner_alliance",
                    "runner_up_party", "runner_up_alliance"]].rename(columns={
        "winner_party": "winner_party_2016",
        "winner_alliance": "winner_alliance_2016",
        "runner_up_party": "runner_up_party_2016",
        "runner_up_alliance": "runner_up_alliance_2016",
    })
    hist21 = res21[["ac_no", "winner_party", "winner_alliance",
                    "runner_up_party", "runner_up_alliance",
                    "winner_votes", "runner_up_votes", "total_valid_votes"]].rename(columns={
        "winner_party": "winner_party_2021",
        "winner_alliance": "winner_alliance_2021",
        "runner_up_party": "runner_up_party_2021",
        "runner_up_alliance": "runner_up_alliance_2021",
        "winner_votes": "winner_votes_2021",
        "runner_up_votes": "runner_up_votes_2021",
        "total_valid_votes": "total_valid_votes_2021",
    })

    df = master.merge(hist16, on="ac_no", how="left").merge(hist21, on="ac_no", how="left")
    if df["winner_alliance_2021"].isna().any():
        missing = df[df["winner_alliance_2021"].isna()]["ac_no"].tolist()
        raise ValueError(f"Missing 2021 historical result for ACs: {missing}")

    # A few rows in the scraped historical data have no recorded runner-up.
    # Default to the opposite major alliance so the downstream feature joins
    # never see NaN. Same fallback applied to runner_up_party for completeness.
    opposite = {"DMK_ALLIANCE": "AIADMK_NDA", "AIADMK_NDA": "DMK_ALLIANCE",
                "TVK": "DMK_ALLIANCE", "OTHERS": "DMK_ALLIANCE"}
    df["runner_up_alliance_2021"] = df["runner_up_alliance_2021"].fillna(
        df["winner_alliance_2021"].map(opposite)
    )
    df["runner_up_party_2021"] = df["runner_up_party_2021"].fillna("IND")
    df["runner_up_alliance_2016"] = df["runner_up_alliance_2016"].fillna(
        df["winner_alliance_2016"].map(opposite)
    )
    df["runner_up_party_2016"] = df["runner_up_party_2016"].fillna("IND")

    valid_votes = df["total_valid_votes_2021"].fillna(0).astype(int)
    margin_votes = (df["winner_votes_2021"].fillna(0).astype(int)
                    - df["runner_up_votes_2021"].fillna(0).astype(int))
    df["margin_pct_2021"] = np.where(valid_votes > 0,
                                     margin_votes / valid_votes, 0.0).round(4)
    df["vote_share_2021"] = np.where(valid_votes > 0,
                                     df["winner_votes_2021"].fillna(0) / valid_votes,
                                     0.0).round(4)

    proj_rows = [
        _project_shares_for_row(row, ls24, as21, supply, rng)
        for _, row in df.iterrows()
    ]

    for p in PARTIES:
        df[f"proj_2026_{p.lower()}_pct"] = [round(r[p], 6) for r in proj_rows]

    canonical = {
        "proj_2026_dmk_alliance_pct": "DMK_ALLIANCE",
        "proj_2026_aiadmk_nda_pct":   "AIADMK_NDA",
        "proj_2026_tvk_pct":          "TVK",
        "proj_2026_ntk_pct":          "NTK",
        "proj_2026_others_pct":       "OTHERS",
    }
    pct_cols = list(canonical.keys())
    df["proj_2026_winner"] = df[pct_cols].idxmax(axis=1).map(canonical)

    def _top_margin(row: pd.Series) -> float:
        vals = sorted([row[c] for c in pct_cols], reverse=True)
        return round(vals[0] - vals[1], 6)

    df["proj_2026_margin_pct"] = df.apply(_top_margin, axis=1)

    out_cols = [
        "ac_no", "ac_name", "district", "region_5way", "reservation", "is_reserved",
        "winner_party_2016", "winner_alliance_2016",
        "runner_up_party_2016", "runner_up_alliance_2016",
        "winner_party_2021", "winner_alliance_2021",
        "runner_up_party_2021", "runner_up_alliance_2021",
        "winner_votes_2021", "runner_up_votes_2021", "total_valid_votes_2021",
        "margin_pct_2021", "vote_share_2021",
        "proj_2026_dmk_alliance_pct", "proj_2026_aiadmk_nda_pct",
        "proj_2026_tvk_pct", "proj_2026_ntk_pct", "proj_2026_others_pct",
        "proj_2026_winner", "proj_2026_margin_pct",
    ]
    out = df[out_cols].copy()
    out_path = os.path.join(_DATA_DIR, "tamilnadu_assembly_2026.csv")
    out.to_csv(out_path, index=False)
    print(f"  tamilnadu_assembly_2026.csv: {len(out)} rows")
    print(f"  Projected winner distribution: {out['proj_2026_winner'].value_counts().to_dict()}")
    return out


def main() -> None:
    print("=" * 60)
    print("  TAMIL NADU 2026 - CONSTITUENCY PROJECTION ENGINE")
    print("=" * 60)
    build_assembly_2026()
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()
