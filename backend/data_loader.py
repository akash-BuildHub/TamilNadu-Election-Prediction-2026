"""
CSV-driven training-data loader for Tamil Nadu Assembly Election 2026.

Reads all Tamil Nadu CSVs under backend/data_files/ and merges them into a
single 234-row constituency-level DataFrame ready for the model.

Inputs (all under backend/data_files/):
  Per-constituency (234 rows):
    tamilnadu_constituency_master_2026.csv  -- spine: ac_no, ac_name, district,
                                               region_5way, reservation, is_reserved
    tamilnadu_assembly_2016_results.csv     -- real per-AC 2016 winner/runner-up
    tamilnadu_assembly_2021_results.csv     -- real per-AC 2021 winner/runner-up
    tamilnadu_assembly_2026.csv             -- projection targets (built by
                                               create_dataset.py OR hand-curated)
  Per-district (38 rows):
    tamilnadu_demographics.csv              -- district-level voter aggregates
  State-level historical:
    tamilnadu_lok_sabha_election_2014.csv
    tamilnadu_lok_sabha_election_2019.csv
    tamilnadu_lok_sabha_election_2024.csv
    tamilnadu_assembly_election_2016.csv
    tamilnadu_assembly_election_2021.csv
  Per-alliance / per-party (2026):
    tamilnadu_sentiment_analysis_2026.csv
    tamilnadu_party_wise_seat_table_2026.csv
    tamilnadu_DMK_ALLIANCE_seat_sharing_2026.csv
    tamilnadu_AIADMK_NDA_seat_sharing_2026.csv
    tamilnadu_main_parties_2026.csv
  State-level voter aggregates (2026):
    tamilnadu_electorate_total_2026.csv
    tamilnadu_people_yet_to_vote_2026.csv
    tamilnadu_first_time_voters_2026.csv
    tamilnadu_nominations_and_candidates_2026.csv
    tamilnadu_gender_wise_voters_2026.csv
  Cross-checks only:
    tamilnadu_election_comparison_table.csv
    tamilnadu_elections_results_past_10_years.csv

Output:
  load_training_dataframe() -> pd.DataFrame with 234 rows, including
    target columns (proj_2026_winner + proj_2026_{alliance}_pct) and all
    engineered features.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd

from config import PARTIES  # ["DMK_ALLIANCE","AIADMK_NDA","TVK","OTHERS"]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")

CONFIDENCE_MAP = {
    "Low": 0.20,
    "Low-Medium": 0.40,
    "Medium": 0.60,
    "Medium-High": 0.75,
    "High": 0.90,
    "Slight edge / virtual tie": 0.55,
    "Competitive": 0.50,
    "Third front": 0.30,
    "Minor": 0.15,
}

# Year-specific party -> alliance mapping used when computing state-level
# historical alliance vote shares. Mirrors create_dataset.py.
_ASSEMBLY_ALLIANCE_BY_YEAR: Dict[int, Dict[str, str]] = {
    2016: {
        "AIADMK": "AIADMK_NDA", "PMK": "AIADMK_NDA", "DMDK": "AIADMK_NDA",
        "BJP": "OTHERS",
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

_LS_ALLIANCE = {
    "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "Congress": "DMK_ALLIANCE",
    "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE",
    "VCK": "DMK_ALLIANCE", "MDMK": "DMK_ALLIANCE", "IUML": "DMK_ALLIANCE",
    "AIADMK": "AIADMK_NDA", "BJP": "AIADMK_NDA",
    "PMK": "AIADMK_NDA", "DMDK": "AIADMK_NDA",
    "NTK": "NTK",
    "MNM": "OTHERS", "AMMK": "OTHERS",
}


def _read(name: str) -> pd.DataFrame:
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _try_read(name: str) -> pd.DataFrame | None:
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) <= 3:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _alliance_shares_from_table(df: pd.DataFrame, year: int | None, kind: str) -> Dict[str, float]:
    """Collapse a party-level state election table into alliance shares."""
    if kind == "Assembly":
        mapping = _ASSEMBLY_ALLIANCE_BY_YEAR.get(year, {})
    else:
        mapping = _LS_ALLIANCE
    out = {p: 0.0 for p in PARTIES}
    for _, row in df.iterrows():
        alliance = mapping.get(str(row["party"]).strip(), "OTHERS")
        out[alliance] += float(row.get("vote_share", 0.0)) / 100.0
    return out


def _state_alliance_trends() -> Dict[str, float]:
    """
    Alliance-level vote shares across all historical elections + inter-period
    swings. Keys: "{ALLIANCE}_{tag}" where tag is ls14/ls19/ls24/as16/as21
    or one of the swing tags.
    """
    specs = [
        ("ls14", "tamilnadu_lok_sabha_election_2014.csv", 2014, "Lok Sabha"),
        ("ls19", "tamilnadu_lok_sabha_election_2019.csv", 2019, "Lok Sabha"),
        ("ls24", "tamilnadu_lok_sabha_election_2024.csv", 2024, "Lok Sabha"),
        ("as16", "tamilnadu_assembly_election_2016.csv", 2016, "Assembly"),
        ("as21", "tamilnadu_assembly_election_2021.csv", 2021, "Assembly"),
    ]
    trends: Dict[str, float] = {}
    for tag, fname, year, kind in specs:
        shares = _alliance_shares_from_table(_read(fname), year, kind)
        for p, v in shares.items():
            trends[f"{p}_{tag}"] = v

    for p in PARTIES:
        trends[f"{p}_ls_swing_2024_2019"] = trends[f"{p}_ls24"] - trends[f"{p}_ls19"]
        trends[f"{p}_ls_swing_2019_2014"] = trends[f"{p}_ls19"] - trends[f"{p}_ls14"]
        trends[f"{p}_as_swing_2021_2016"] = trends[f"{p}_as21"] - trends[f"{p}_as16"]
    return trends


def _alliance_sentiment() -> Dict[str, float]:
    df = _read("tamilnadu_sentiment_analysis_2026.csv")
    df["party"] = df["party"].astype(str).str.strip()
    df["score"] = df["confidence_pre_result"].map(CONFIDENCE_MAP)
    if df["score"].isna().any():
        unknown = df.loc[df["score"].isna(), "confidence_pre_result"].unique()
        raise ValueError(f"Unknown confidence_pre_result label(s): {list(unknown)}")
    return df.set_index("party")["score"].to_dict()


def _alliance_concentration() -> Dict[str, float]:
    """Herfindahl index per front: sum of squared seat-share fractions."""
    df = _read("tamilnadu_party_wise_seat_table_2026.csv")
    out: Dict[str, float] = {}
    # Front column maps to our alliance classes when possible. The 2026 data
    # lumps TVK and NTK together under "Independent front"; we approximate
    # each as 1.0 (single-party front, maximum concentration).
    front_to_alliance = {
        "DMK-led alliance": "DMK_ALLIANCE",
        "AIADMK-led NDA":   "AIADMK_NDA",
    }
    for front, group in df.groupby("front"):
        alliance = front_to_alliance.get(str(front).strip())
        if alliance is None:
            continue
        total = float(group["seats"].sum())
        if total <= 0:
            out[alliance] = 0.0
            continue
        shares = group["seats"].astype(float) / total
        out[alliance] = float((shares ** 2).sum())
    out.setdefault("TVK", 1.0)
    out.setdefault("NTK", 1.0)
    out.setdefault("OTHERS", 0.3)
    return out


def _alliance_breadth() -> Dict[str, int]:
    """Distinct allied parties per alliance."""
    out: Dict[str, int] = {}
    for alliance, fname in [
        ("DMK_ALLIANCE", "tamilnadu_DMK_ALLIANCE_seat_sharing_2026.csv"),
        ("AIADMK_NDA",   "tamilnadu_AIADMK_NDA_seat_sharing_2026.csv"),
    ]:
        df = _read(fname)
        out[alliance] = int(df["party"].nunique())
    out["TVK"] = 1
    out["NTK"] = 1
    out["OTHERS"] = 2  # IND + rebels (approximation, NTK split out)
    return out


def _state_voter_features() -> Dict[str, float]:
    elec = _read("tamilnadu_electorate_total_2026.csv").set_index("metric")["value"]
    yet = _read("tamilnadu_people_yet_to_vote_2026.csv").set_index("metric")["value"]
    ftv = _read("tamilnadu_first_time_voters_2026.csv")
    nom = _read("tamilnadu_nominations_and_candidates_2026.csv").set_index("metric")["count"]

    # Pick the most recent electorate figure available
    total_electorate = 0.0
    for candidate_key in (
        "updated electorate (6 April 2026)",
        "final SIR electorate",
    ):
        v = elec.get(candidate_key)
        if v is not None and not pd.isna(v):
            total_electorate = float(v)
            break
    if total_electorate <= 0:
        # Fallback: take max positive value from the table
        total_electorate = float(pd.to_numeric(elec, errors="coerce").dropna().max() or 0.0)
    if total_electorate <= 0:
        raise ValueError("No usable electorate total in tamilnadu_electorate_total_2026.csv")

    cast = 0.0
    for k in ("updated roll", "final SIR roll"):
        v = yet.get(k)
        if v is not None and not pd.isna(v):
            cast = float(v)
            break

    # First-time voters: pick the 2026 AS value if present
    ftv_value = 0
    for _, row in ftv.iterrows():
        metric = str(row.get("metric", "")).lower()
        if "2026" in metric:
            ftv_value = int(row["count"])
            break
    if ftv_value == 0 and len(ftv) > 0:
        ftv_value = int(pd.to_numeric(ftv["count"], errors="coerce").dropna().max() or 0)

    # Final candidates in fray
    final_candidates = 0.0
    for k in ("final candidates in fray", "Total candidates", "Nominations filed"):
        v = nom.get(k)
        if v is not None and not pd.isna(v):
            final_candidates = float(v)
            break
    if final_candidates == 0:
        # Derive from nominations filed - rejected - withdrawals
        filed = float(nom.get("Nominations filed", 0) or 0)
        rejected = float(nom.get("Rejected", 0) or 0)
        withdrawn = float(nom.get("Withdrawals", 0) or 0)
        final_candidates = max(0.0, filed - rejected - withdrawn)

    return {
        "state_turnout_pct": 0.72,  # TN historical turnout baseline (~71-73%)
        "state_first_time_voter_pct": float(ftv_value) / total_electorate if total_electorate else 0.02,
        "state_candidates_per_seat": final_candidates / 234.0 if final_candidates else 20.0,
    }


def _validate_cross_checks() -> None:
    cmp_df = _try_read("tamilnadu_election_comparison_table.csv")
    hist_df = _try_read("tamilnadu_elections_results_past_10_years.csv")
    # These files are sanity-check artefacts. Mismatches emit warnings only.
    if cmp_df is None or hist_df is None:
        return
    for _, row in hist_df.iterrows():
        year = int(row["year"])
        etype = str(row["election_type"]).strip()
        fname = ("tamilnadu_lok_sabha_election_" if etype == "Lok Sabha"
                 else "tamilnadu_assembly_election_") + f"{year}.csv"
        df = _try_read(fname)
        if df is None:
            continue
        winner = df.sort_values("seats_won", ascending=False).iloc[0]["party"]
        expected = str(row["winner"]).strip()
        if expected.upper() not in winner.upper() and winner.upper() not in expected.upper():
            warnings.warn(
                f"Cross-check: past_10_years says '{expected}' won {etype} {year}, "
                f"but {fname} top seat-winner is '{winner}'.", stacklevel=2,
            )


def load_training_dataframe() -> pd.DataFrame:
    """
    234-row DataFrame with all engineered features and target columns.
    Raises ValueError on data-quality problems.
    """
    # Spine: 234 ACs from the constituency master
    master = _read("tamilnadu_constituency_master_2026.csv")
    if len(master) != 234:
        raise ValueError(f"Expected 234 constituencies in master, got {len(master)}")
    spine = master.rename(columns={"ac_name": "constituency"})[
        ["ac_no", "constituency", "district", "region_5way", "reservation", "is_reserved"]
    ].copy()

    # Per-AC historical + 2026 targets
    ac = _read("tamilnadu_assembly_2026.csv")
    keep_cols = [
        "ac_no",
        "winner_alliance_2016", "winner_alliance_2021", "runner_up_alliance_2021",
        "vote_share_2021", "margin_pct_2021",
        "proj_2026_winner",
        "proj_2026_dmk_alliance_pct", "proj_2026_aiadmk_nda_pct",
        "proj_2026_tvk_pct", "proj_2026_ntk_pct", "proj_2026_others_pct",
    ]
    df = spine.merge(ac[keep_cols], on="ac_no", how="left")

    missing = df["winner_alliance_2021"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} constituencies missing 2021 history. "
            f"Regenerate backend/data_files/tamilnadu_assembly_2026.csv."
        )

    # District-level aggregates
    demo = _read("tamilnadu_demographics.csv")
    df = df.merge(demo, on="district", how="left")
    if df["total_voters"].isna().any():
        bad = df.loc[df["total_voters"].isna(), "district"].unique().tolist()
        raise ValueError(f"District demographics missing for: {bad}")

    # State-level trend features -> incumbent / runner-up interaction columns.
    trends = _state_alliance_trends()
    for tag in ("ls_swing_2024_2019", "ls_swing_2019_2014", "as_swing_2021_2016"):
        df[f"incumbent_{tag}"] = df["winner_alliance_2021"].map(
            lambda p, t=tag: trends.get(f"{p}_{t}", 0.0)
        )
    for tag in ("ls_swing_2024_2019", "as_swing_2021_2016"):
        df[f"runnerup_{tag}"] = df["runner_up_alliance_2021"].map(
            lambda p, t=tag: trends.get(f"{p}_{t}", 0.0)
        )

    # Sentiment + alliance structure joined via incumbent / challenger
    sent = _alliance_sentiment()
    conc = _alliance_concentration()
    breadth = _alliance_breadth()

    df["incumbent_sentiment"] = df["winner_alliance_2021"].map(sent).fillna(0.5)
    df["challenger_sentiment"] = df["runner_up_alliance_2021"].map(sent).fillna(0.5)
    df["incumbent_concentration"] = df["winner_alliance_2021"].map(conc).fillna(0.5)
    df["challenger_concentration"] = df["runner_up_alliance_2021"].map(conc).fillna(0.5)
    df["incumbent_breadth"] = df["winner_alliance_2021"].map(breadth).fillna(0).astype(float)
    df["challenger_breadth"] = df["runner_up_alliance_2021"].map(breadth).fillna(0).astype(float)

    # State-level voter aggregates (broadcast as constants)
    voter = _state_voter_features()
    for k, v in voter.items():
        df[k] = v

    # Sanity / completeness
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaN values remain after merging in columns: {bad_cols}")

    target_cols = [
        "proj_2026_dmk_alliance_pct", "proj_2026_aiadmk_nda_pct",
        "proj_2026_tvk_pct", "proj_2026_ntk_pct", "proj_2026_others_pct",
    ]
    sums = df[target_cols].sum(axis=1)
    if (sums - 1.0).abs().max() > 0.05:
        raise ValueError(
            f"Vote-share targets in tamilnadu_assembly_2026.csv deviate from 1.0 "
            f"by more than 5% on at least one row (max drift {(sums - 1.0).abs().max():.4f})."
        )
    df[target_cols] = df[target_cols].div(sums, axis=0)

    _validate_cross_checks()

    return df


if __name__ == "__main__":
    out = load_training_dataframe()
    print(f"Loaded {len(out)} constituencies, {out.shape[1]} columns")
    print("Target winner counts:", out["proj_2026_winner"].value_counts().to_dict())
    print("Sample row:")
    print(out.iloc[0])
