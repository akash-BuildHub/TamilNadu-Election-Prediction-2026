"""
Build the ML-ready Tamil Nadu 2026 Assembly election dataset.

Joins the 234-AC master spine with verified TCPD-schema constituency-level
historical results (2016, 2021) fetched from OpenCity's CKAN mirror of the
TCPD/Ashoka dataset. Writes three CSVs to backend/dataset/:

  tn_model_dataset_updated.csv       - 234 rows, full requested schema
  tn_model_dataset_missing_report.csv - per-column coverage + reason
  tn_model_dataset_source_notes.csv  - which source supports which columns

Rules honoured:
  - No fabricated constituency-level values. Unverifiable fields are left
    empty (numeric) or set to "needs_source" (text), and tracked in the
    missing-report output.
  - All 234 AC rows preserved.
  - Match is on AC number (Constituency_No == ac_no), with a name cross-check
    used only for sanity logging; mismatches do not drop rows.
  - Source tag per data group (source_2016, source_2021, ...).

Standalone. Does NOT modify the existing training pipeline (train.py still
reads backend/data_files/). To feed the new file into the model, add an
adapter in data_loader.py.

Usage:
  python backend/build_model_dataset.py               # use cache if present
  python backend/build_model_dataset.py --refetch     # force re-download
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
SPINE_PATH = os.path.join(DATASET_DIR, "tamilnadu_assembly_constituency_2026.csv")

OUT_MAIN = os.path.join(DATASET_DIR, "tn_model_dataset_updated.csv")
OUT_MISSING = os.path.join(DATASET_DIR, "tn_model_dataset_missing_report.csv")
OUT_SOURCES = os.path.join(DATASET_DIR, "tn_model_dataset_source_notes.csv")

OPENCITY_2016_URL = (
    "https://data.opencity.in/dataset/b0933494-6a9c-4b91-816a-a97a69f2348c/"
    "resource/d17eb781-2e8a-4230-bdce-88e361a803f8/download/tn_2016_elections.csv"
)
OPENCITY_2021_URL = (
    "https://data.opencity.in/dataset/74f05dff-eac3-4e2b-80cd-3382e9c190d5/"
    "resource/5ca1808c-4b8a-4fa6-957e-409e690f509d/download/"
    "b89dceb3-b295-4d9b-af6d-b71d2a86ee49.csv"
)

# Source CSVs live directly under backend/dataset/ (no separate raw/ subdir).
SRC_2016 = os.path.join(DATASET_DIR, "opencity_tn_2016.csv")
SRC_2021 = os.path.join(DATASET_DIR, "opencity_tn_2021.csv")

# OpenCity / TCPD party codes -> canonical codes used in this project.
PARTY_CANON = {
    "ADMK": "AIADMK",
    "AIADMK": "AIADMK",
    "DMK": "DMK",
    "INC": "INC",
    "BJP": "BJP",
    "CPI": "CPI",
    "CPM": "CPI(M)",
    "CPI(M)": "CPI(M)",
    "VCK": "VCK",
    "PMK": "PMK",
    "DMDK": "DMDK",
    "MDMK": "MDMK",
    "IUML": "IUML",
    "AMMK": "AMMK",
    "MNM": "MNM",
    "NTK": "NTK",
    "TVK": "TVK",
    "IND": "IND",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def _fetch(url: str, dest: str, force: bool) -> None:
    """Download `url` to `dest`. Skip if cached and not forcing."""
    if os.path.exists(dest) and not force:
        _log(f"  cache hit: {os.path.relpath(dest, BACKEND_DIR)}")
        return
    if requests is None:
        raise RuntimeError(
            "requests is not installed and no cache is available at "
            f"{dest}. Install requests or drop the file in place."
        )
    _log(f"  fetching {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(r.content)
    _log(f"  saved: {os.path.relpath(dest, BACKEND_DIR)} ({len(r.content)} bytes)")


def fetch_sources(force: bool = False) -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    _log("Fetching source files (OpenCity TCPD mirror):")
    _fetch(OPENCITY_2016_URL, SRC_2016, force)
    _fetch(OPENCITY_2021_URL, SRC_2021, force)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _canon(name: str) -> str:
    """Canonicalise an AC name for cross-checking only (not a match key)."""
    if name is None:
        return ""
    s = str(name).upper()
    s = re.sub(r"\s*\(\s*(SC|ST)\s*\)\s*$", "", s)
    s = re.sub(r"\s+(SC|ST)\s*$", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    # Frequent spelling variants that show up between OpenCity and CEO.
    replacements = [
        ("THIRU", "TIRU"),
        ("POONAMALLEE", "POONAMALLEE"),
        ("MADAVARAM", "MADHAVARAM"),
        ("PUNDI", "POONDI"),
        ("KANYAKUMARI", "KANNIYAKUMARI"),
        ("GANDHARVAKOTTAI", "GANDARVAKOTTAI"),
        ("BODINAYAKKANUR", "BODINAYAKANUR"),
        ("VILLUPURAM", "VILUPPURAM"),
    ]
    for src, dst in replacements:
        s = s.replace(src, dst)
    return s


def _norm_party(p: Optional[str]) -> str:
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "Other"
    p = str(p).strip().upper()
    if not p:
        return "Other"
    return PARTY_CANON.get(p, "Other")


# --------------------------------------------------------------------------
# Per-election aggregation
# --------------------------------------------------------------------------

# Canonical party buckets for the year-specific vote-share columns.
SHARE_PARTY_COLS = {
    "DMK": "dmk_vote_share",
    "AIADMK": "aiadmk_vote_share",
    "BJP": "bjp_vote_share",
    "INC": "congress_vote_share",
    "NTK": "ntk_vote_share",
    # "others" is a residual computed at the end
}


def aggregate_ac(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Collapse the candidate-level OpenCity TCPD CSV to one row per AC, with
    the columns our output schema expects for that year.

    Expected columns in raw:
      Constituency_No, Constituency_Name, Position, Candidate, Party,
      Votes, Valid_Votes, Electors, Turnout_Percentage, Margin,
      Margin_Percentage
    """
    required = {
        "Constituency_No", "Constituency_Name", "Position", "Candidate",
        "Party", "Votes", "Valid_Votes", "Electors",
        "Turnout_Percentage", "Margin", "Margin_Percentage",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{year} raw file missing columns: {sorted(missing)}")

    df = raw.copy()
    df["Constituency_No"] = pd.to_numeric(df["Constituency_No"], errors="coerce").astype("Int64")
    for c in ("Position", "Votes", "Valid_Votes", "Electors", "Margin"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Turnout_Percentage", "Margin_Percentage"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Constituency_No", "Position"])

    # One row per (ac, position 1) = winner; (ac, position 2) = runner-up.
    winners = (
        df[df["Position"] == 1]
        .sort_values("Constituency_No")
        .drop_duplicates("Constituency_No", keep="first")
    )
    runners = (
        df[df["Position"] == 2]
        .sort_values("Constituency_No")
        .drop_duplicates("Constituency_No", keep="first")
    )

    # AC-level scalars: Valid_Votes, Electors, Turnout_Percentage are repeated
    # on every candidate row for a given AC. Take max (all equal) to be safe.
    ac_scalars = (
        df.groupby("Constituency_No")
          .agg(valid_votes=("Valid_Votes", "max"),
               electors=("Electors", "max"),
               turnout_pct=("Turnout_Percentage", "max"),
               constituency_name=("Constituency_Name", "first"))
          .reset_index()
    )

    w = winners[["Constituency_No", "Candidate", "Party", "Votes", "Margin",
                 "Margin_Percentage"]].rename(columns={
        "Candidate": f"winner_{year}",
        "Party":     f"winner_party_{year}_raw",
        "Votes":     f"winner_votes_{year}",
        "Margin":    f"margin_votes_{year}",
        "Margin_Percentage": f"margin_pct_{year}",
    })
    r = runners[["Constituency_No", "Candidate", "Party", "Votes"]].rename(columns={
        "Candidate": f"runner_up_{year}",
        "Party":     f"runner_up_party_{year}_raw",
        "Votes":     f"runner_up_votes_{year}",
    })

    agg = ac_scalars.merge(w, on="Constituency_No", how="left") \
                    .merge(r, on="Constituency_No", how="left")

    # Per-party vote-share columns. Sum Votes for all candidates of a party,
    # divide by Valid_Votes for that AC.
    df["_party"] = df["Party"].map(_norm_party)
    party_sums = (
        df.groupby(["Constituency_No", "_party"])["Votes"].sum().reset_index()
    )
    # wide pivot: one column per canonical party
    wide = party_sums.pivot(index="Constituency_No", columns="_party", values="Votes").fillna(0)

    # Merge the valid_votes denominator from agg
    wide = wide.merge(agg[["Constituency_No", "valid_votes"]], on="Constituency_No", how="left")

    for src_party, out_name in SHARE_PARTY_COLS.items():
        col_votes = wide[src_party] if src_party in wide.columns else 0.0
        share = (col_votes / wide["valid_votes"]).where(wide["valid_votes"] > 0)
        agg[f"{out_name}_{year}"] = (share * 100.0).round(4)

    # others_vote_share = 100 - sum of named parties (clipped at 0 for safety)
    named_cols = [f"{name}_{year}" for name in SHARE_PARTY_COLS.values()]
    agg[f"others_vote_share_{year}"] = (
        100.0 - agg[named_cols].sum(axis=1)
    ).clip(lower=0).round(4)

    # Normalise party codes for winner/runner-up.
    agg[f"winner_party_{year}"] = agg[f"winner_party_{year}_raw"].map(_norm_party)
    agg[f"runner_up_party_{year}"] = agg[f"runner_up_party_{year}_raw"].map(_norm_party)
    agg = agg.drop(columns=[f"winner_party_{year}_raw", f"runner_up_party_{year}_raw"])

    agg = agg.rename(columns={
        "Constituency_No": "ac_no",
        "turnout_pct": f"turnout_pct_{year}",
        "constituency_name": f"_src_name_{year}",
    })

    # Keep only the columns we care about in the final dataset.
    out_cols = [
        "ac_no",
        f"_src_name_{year}",
        f"winner_{year}",
        f"winner_party_{year}",
        f"winner_votes_{year}",
        f"runner_up_{year}",
        f"runner_up_party_{year}",
        f"runner_up_votes_{year}",
        f"margin_votes_{year}",
        f"margin_pct_{year}",
        f"turnout_pct_{year}",
    ] + [f"{name}_{year}" for name in SHARE_PARTY_COLS.values()] \
      + [f"others_vote_share_{year}"]

    out = agg[out_cols].copy()
    out["ac_no"] = out["ac_no"].astype(int)
    return out


# --------------------------------------------------------------------------
# Schema: the full set of columns the user wants in the output.
# Each entry: (column_name, default, source_group)
# Columns with source_group in {"spine", "2016_results", "2021_results"}
# get populated from upstream sources. Everything else defaults to
# needs_source (text) or empty (numeric), and is listed in the missing-
# report output.
# --------------------------------------------------------------------------

NEEDS = "needs_source"
# Groups used as sourcing tags. Keep in sync with source_notes below.
G_SPINE = "spine"
G_2016  = "2016_results"
G_2021  = "2021_results"
G_VOTER = "voter_aggregates_per_ac"
G_2024  = "2024_lok_sabha_ac_segment"
G_2026  = "2026_context"
G_DEMO  = "demographics"
G_TVK   = "tvk_indicators"
G_TREND = "trend_and_prediction"

# Numeric columns get blank (NaN) when unsourced so dtypes stay clean.
# Everything else gets the string literal "needs_source".
NUMERIC_COLS = {
    "total_voters", "male_voters", "female_voters", "gender_gap", "first_time_voters",
    "winner_votes_2016", "runner_up_votes_2016",
    "margin_votes_2016", "margin_pct_2016", "turnout_pct_2016",
    "dmk_vote_share_2016", "aiadmk_vote_share_2016", "bjp_vote_share_2016",
    "congress_vote_share_2016", "ntk_vote_share_2016", "others_vote_share_2016",
    "winner_votes_2021", "runner_up_votes_2021",
    "margin_votes_2021", "margin_pct_2021", "turnout_pct_2021",
    "dmk_vote_share_2021", "aiadmk_vote_share_2021", "bjp_vote_share_2021",
    "congress_vote_share_2021", "ntk_vote_share_2021", "others_vote_share_2021",
    "dmk_alliance_vote_share_2024", "aiadmk_alliance_vote_share_2024",
    "bjp_nda_vote_share_2024", "ntk_vote_share_2024", "others_vote_share_2024",
    "urbanization_pct", "literacy_rate",
    "sc_population_pct", "st_population_pct", "minority_population_pct",
    "youth_voter_strength", "women_voter_strength",
    "tvk_youth_appeal_score", "tvk_urban_appeal_score",
    "tvk_kongu_impact_score", "tvk_anti_establishment_score",
    "predicted_vote_share_dmk_alliance", "predicted_vote_share_aiadmk_nda",
    "predicted_vote_share_tvk", "predicted_vote_share_ntk_others",
    "confidence_score",
}

SCHEMA: list[tuple[str, str]] = [
    # Spine (from the constituency master)
    ("ac_no", G_SPINE),
    ("ac_name", G_SPINE),
    ("district", G_SPINE),
    ("region", G_SPINE),
    ("reservation", G_SPINE),
    # Voter aggregates the user's base partial CSV had (per-AC). Not in this
    # repo; left blank here.
    ("total_voters", G_VOTER),
    ("male_voters", G_VOTER),
    ("female_voters", G_VOTER),
    ("gender_gap", G_VOTER),
    ("first_time_voters", G_VOTER),

    # 2016 Assembly
    ("winner_2016", G_2016),
    ("winner_party_2016", G_2016),
    ("runner_up_2016", G_2016),
    ("runner_up_party_2016", G_2016),
    ("winner_votes_2016", G_2016),
    ("runner_up_votes_2016", G_2016),
    ("margin_votes_2016", G_2016),
    ("margin_pct_2016", G_2016),
    ("turnout_pct_2016", G_2016),
    ("dmk_vote_share_2016", G_2016),
    ("aiadmk_vote_share_2016", G_2016),
    ("bjp_vote_share_2016", G_2016),
    ("congress_vote_share_2016", G_2016),
    ("ntk_vote_share_2016", G_2016),
    ("others_vote_share_2016", G_2016),

    # 2021 Assembly
    ("winner_2021", G_2021),
    ("winner_party_2021", G_2021),
    ("runner_up_2021", G_2021),
    ("runner_up_party_2021", G_2021),
    ("winner_votes_2021", G_2021),
    ("runner_up_votes_2021", G_2021),
    ("margin_votes_2021", G_2021),
    ("margin_pct_2021", G_2021),
    ("turnout_pct_2021", G_2021),
    ("dmk_vote_share_2021", G_2021),
    ("aiadmk_vote_share_2021", G_2021),
    ("bjp_vote_share_2021", G_2021),
    ("congress_vote_share_2021", G_2021),
    ("ntk_vote_share_2021", G_2021),
    ("others_vote_share_2021", G_2021),

    # 2024 Lok Sabha mapped to AC segment (NOT available as a clean CSV)
    ("dmk_alliance_vote_share_2024", G_2024),
    ("aiadmk_alliance_vote_share_2024", G_2024),
    ("bjp_nda_vote_share_2024", G_2024),
    ("ntk_vote_share_2024", G_2024),
    ("others_vote_share_2024", G_2024),
    ("winning_alliance_2024", G_2024),

    # 2026 candidate context
    ("contesting_alliance_2026", G_2026),
    ("dmk_alliance_candidate_2026", G_2026),
    ("aiadmk_nda_candidate_2026", G_2026),
    ("tvk_candidate_2026", G_2026),
    ("ntk_candidate_2026", G_2026),
    ("bjp_candidate_2026_if_separate", G_2026),
    ("incumbent_party", G_2026),
    ("incumbent_candidate", G_2026),
    ("incumbent_contesting_again", G_2026),
    ("alliance_change_flag", G_2026),
    ("rebel_candidate_flag", G_2026),
    ("strong_local_candidate_flag", G_2026),

    # Demographic and socio-political
    ("urban_rural_type", G_DEMO),
    ("urbanization_pct", G_DEMO),
    ("literacy_rate", G_DEMO),
    ("sc_population_pct", G_DEMO),
    ("st_population_pct", G_DEMO),
    ("minority_population_pct", G_DEMO),
    ("youth_voter_strength", G_DEMO),
    ("women_voter_strength", G_DEMO),
    ("caste_influence_notes", G_DEMO),
    ("economic_profile", G_DEMO),
    ("key_local_issues", G_DEMO),

    # TVK indicators
    ("tvk_presence_level", G_TVK),
    ("tvk_youth_appeal_score", G_TVK),
    ("tvk_urban_appeal_score", G_TVK),
    ("tvk_kongu_impact_score", G_TVK),
    ("tvk_anti_establishment_score", G_TVK),
    ("tvk_vote_split_target", G_TVK),
    ("tvk_estimated_vote_share_range", G_TVK),
    ("tvk_seat_win_probability", G_TVK),
    ("tvk_notes", G_TVK),

    # Current trend and prediction
    ("current_trend_dmk_alliance", G_TREND),
    ("current_trend_aiadmk_nda", G_TREND),
    ("current_trend_tvk", G_TREND),
    ("current_trend_ntk", G_TREND),
    ("likely_winner_alliance_2026", G_TREND),
    ("predicted_runner_up_alliance_2026", G_TREND),
    ("predicted_vote_share_dmk_alliance", G_TREND),
    ("predicted_vote_share_aiadmk_nda", G_TREND),
    ("predicted_vote_share_tvk", G_TREND),
    ("predicted_vote_share_ntk_others", G_TREND),
    ("confidence_score", G_TREND),
    ("prediction_basis_notes", G_TREND),
]

# Per-group source labels written into the source column for every row.
SOURCE_LABELS = {
    G_SPINE: "tamilnadu_assembly_constituency_2026.csv",
    G_2016:  "opencity.in TCPD mirror (tn_2016_elections.csv)",
    G_2021:  "opencity.in TCPD mirror (tn_2021_elections.csv)",
    G_VOTER: NEEDS,
    G_2024:  NEEDS,
    G_2026:  NEEDS,
    G_DEMO:  NEEDS,
    G_TVK:   NEEDS,
    G_TREND: NEEDS,
}

# Supplementary patch: the 2 ACs whose 2016 general poll was countermanded
# by ECI (cash-for-votes probe) and held as a byelection on 19 Nov 2016.
# Both won by AIADMK per ECI's byelection statistical report. OpenCity's
# general-election-day CSV does not include these byelection results.
# We populate only party-level winner/runner-up (publicly verified); vote
# counts, margin, turnout require the specific byelection Form 20 and are
# left blank.
BYELECTION_2016_PATCH = {
    134: {  # Aravakurichi, Karur
        "winner_party_2016": "AIADMK",
        "runner_up_party_2016": "DMK",
    },
    174: {  # Thanjavur, Thanjavur
        "winner_party_2016": "AIADMK",
        "runner_up_party_2016": "DMK",
    },
}
BYELECTION_2016_SOURCE = "ECI 2016 byelection statistical report (19 Nov 2016)"


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def load_spine() -> pd.DataFrame:
    df = pd.read_csv(SPINE_PATH)
    if len(df) != 234:
        raise ValueError(f"Spine should have 234 rows, got {len(df)}")
    df = df.rename(columns={"region_5way": "region"})
    keep = ["ac_no", "ac_name", "district", "region", "reservation"]
    return df[keep].copy()


def build_dataset(refetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fetch_sources(force=refetch)

    _log("\nAggregating candidate-level rows to AC-level:")
    raw16 = pd.read_csv(SRC_2016)
    raw21 = pd.read_csv(SRC_2021)
    _log(f"  2016 raw candidate rows: {len(raw16)}")
    _log(f"  2021 raw candidate rows: {len(raw21)}")

    ac16 = aggregate_ac(raw16, 2016)
    ac21 = aggregate_ac(raw21, 2021)
    _log(f"  2016 AC-level aggregated rows: {len(ac16)}")
    _log(f"  2021 AC-level aggregated rows: {len(ac21)}")

    spine = load_spine()
    _log(f"\nSpine rows: {len(spine)}")

    merged = spine.merge(ac16, on="ac_no", how="left") \
                  .merge(ac21, on="ac_no", how="left")

    # Name sanity check (no drops, just logs).
    def _name_mismatches(year: int) -> list[tuple[int, str, str]]:
        src_col = f"_src_name_{year}"
        bad = []
        for _, r in merged.iterrows():
            if pd.isna(r[src_col]):
                continue
            if _canon(r["ac_name"]) != _canon(r[src_col]):
                bad.append((int(r["ac_no"]), r["ac_name"], r[src_col]))
        return bad

    mm16 = _name_mismatches(2016)
    mm21 = _name_mismatches(2021)
    _log(f"  Name mismatches 2016 (merged on ac_no, names differ post-canon): {len(mm16)}")
    if mm16:
        for ac_no, master, src in mm16[:10]:
            _log(f"    ac {ac_no}: master='{master}' vs src='{src}'")
        if len(mm16) > 10:
            _log(f"    ... and {len(mm16) - 10} more")
    _log(f"  Name mismatches 2021 (merged on ac_no, names differ post-canon): {len(mm21)}")
    if mm21:
        for ac_no, master, src in mm21[:10]:
            _log(f"    ac {ac_no}: master='{master}' vs src='{src}'")
        if len(mm21) > 10:
            _log(f"    ... and {len(mm21) - 10} more")

    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("_src_name_")])

    # Apply the 2016 byelection patch for ACs missing from OpenCity's
    # general-poll CSV. These values are not fabricated -- they come from
    # ECI's byelection statistical report.
    byelection_acs = []
    for ac_no, patch in BYELECTION_2016_PATCH.items():
        mask = merged["ac_no"] == ac_no
        if not mask.any():
            continue
        for col, val in patch.items():
            merged.loc[mask, col] = val
        byelection_acs.append(ac_no)
    if byelection_acs:
        _log(f"  2016 byelection patch applied for ACs: {byelection_acs}")

    # Coverage stats
    _log("\nCoverage:")
    cov16 = merged["winner_party_2016"].notna().sum()
    cov21 = merged["winner_party_2021"].notna().sum()
    _log(f"  2016 winner_party populated: {cov16}/234")
    _log(f"  2021 winner_party populated: {cov21}/234")

    # Apply schema: add any missing columns as needs_source / NaN; enforce order.
    for col, group in SCHEMA:
        if col not in merged.columns:
            if col in NUMERIC_COLS:
                merged[col] = pd.NA
            else:
                merged[col] = NEEDS

    ordered = [c for c, _ in SCHEMA]

    # Per-row source tag for 2016: default OpenCity, byelection ACs get the
    # ECI byelection source, ACs with no data get needs_source.
    def _source_2016(row: pd.Series) -> str:
        if int(row["ac_no"]) in BYELECTION_2016_PATCH:
            return BYELECTION_2016_SOURCE
        if pd.notna(row.get("winner_2016")):
            return SOURCE_LABELS[G_2016]
        return NEEDS

    source_cols = {
        f"source_{G_2016}": merged.apply(_source_2016, axis=1),
        f"source_{G_2021}": merged.apply(
            lambda r: SOURCE_LABELS[G_2021] if pd.notna(r.get("winner_2021")) else NEEDS, axis=1),
        f"source_{G_VOTER}": NEEDS,
        f"source_{G_2024}": NEEDS,
        f"source_{G_2026}": NEEDS,
        f"source_{G_DEMO}": NEEDS,
        f"source_{G_TVK}": NEEDS,
        f"source_{G_TREND}": NEEDS,
    }
    for name, val in source_cols.items():
        merged[name] = val

    final = merged[ordered + list(source_cols.keys())].copy()

    # Missing-data report: for each schema column, count cells that are
    # empty (NaN) or literal needs_source.
    missing_rows = []
    for col, group in SCHEMA:
        series = final[col]
        if col in NUMERIC_COLS:
            missing_n = int(series.isna().sum())
        else:
            missing_n = int(((series == NEEDS) | series.isna()).sum())
        if missing_n:
            missing_rows.append({
                "column": col,
                "group": group,
                "missing_cells": missing_n,
                "total_cells": len(final),
                "coverage_pct": round(100.0 * (len(final) - missing_n) / len(final), 2),
                "default_source": SOURCE_LABELS.get(group, NEEDS),
            })
    missing_report = pd.DataFrame(missing_rows).sort_values(
        ["group", "column"]
    ).reset_index(drop=True)

    # Source notes: one row per group with its canonical source.
    source_notes = pd.DataFrame([
        {"group": G_SPINE,
         "source": SOURCE_LABELS[G_SPINE],
         "url_or_path": os.path.relpath(SPINE_PATH, BACKEND_DIR),
         "kind": "verified_official (CEO TN)"},
        {"group": G_2016,
         "source": SOURCE_LABELS[G_2016],
         "url_or_path": OPENCITY_2016_URL,
         "kind": "verified_derived (TCPD, ECI-sourced)"},
        {"group": G_2021,
         "source": SOURCE_LABELS[G_2021],
         "url_or_path": OPENCITY_2021_URL,
         "kind": "verified_derived (TCPD, ECI-sourced)"},
        {"group": G_VOTER,
         "source": NEEDS,
         "url_or_path": "CEO TN electoral roll AC-wise (not ingested here)",
         "kind": "missing"},
        {"group": G_2024,
         "source": NEEDS,
         "url_or_path": (
             "elections.tn.gov.in GELS_2024 Form 20 (polling-station PDFs; "
             "needs AC-segment aggregation)"),
         "kind": "missing"},
        {"group": G_2026,
         "source": NEEDS,
         "url_or_path": "CEO TN 2026 candidate list (publishes after nominations close)",
         "kind": "missing"},
        {"group": G_DEMO,
         "source": NEEDS,
         "url_or_path": "Census 2011 AC-mapped indicators / SECC (not ingested here)",
         "kind": "missing"},
        {"group": G_TVK,
         "source": NEEDS,
         "url_or_path": "Survey / polling data; not set at AC level to avoid fabrication",
         "kind": "missing"},
        {"group": G_TREND,
         "source": NEEDS,
         "url_or_path": "Model output - populate after training",
         "kind": "missing"},
    ])

    return final, missing_report, source_notes


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refetch", action="store_true",
                        help="force re-download even if cache is present")
    args = parser.parse_args(argv)

    _log("=" * 70)
    _log("  TN 2026 - ML-READY DATASET BUILDER")
    _log("=" * 70)

    final, missing_report, source_notes = build_dataset(refetch=args.refetch)

    os.makedirs(DATASET_DIR, exist_ok=True)
    final.to_csv(OUT_MAIN, index=False)
    missing_report.to_csv(OUT_MISSING, index=False)
    source_notes.to_csv(OUT_SOURCES, index=False)

    _log("\nOutputs:")
    _log(f"  {os.path.relpath(OUT_MAIN, BACKEND_DIR)}  "
         f"({len(final)} rows, {len(final.columns)} cols)")
    _log(f"  {os.path.relpath(OUT_MISSING, BACKEND_DIR)}  "
         f"({len(missing_report)} rows)")
    _log(f"  {os.path.relpath(OUT_SOURCES, BACKEND_DIR)}  "
         f"({len(source_notes)} rows)")

    _log("\nGroup coverage (populated_cells / total_cells, across all columns in group):")
    group_cols: dict[str, list[str]] = {}
    for col, g in SCHEMA:
        group_cols.setdefault(g, []).append(col)
    for g in sorted(group_cols.keys()):
        cols = group_cols[g]
        total_cells = len(final) * len(cols)
        populated = 0
        for col in cols:
            s = final[col]
            if col in NUMERIC_COLS:
                populated += int(s.notna().sum())
            else:
                populated += int(((s != NEEDS) & s.notna()).sum())
        pct = 100.0 * populated / total_cells if total_cells else 0.0
        _log(f"  {g:35s} cells={populated:5d}/{total_cells:5d}  coverage={pct:6.2f}%")

    _log("\nSample rows:")
    preview_cols = [
        "ac_no", "ac_name", "district", "region",
        "winner_2016", "winner_party_2016", "margin_pct_2016", "turnout_pct_2016",
        "winner_2021", "winner_party_2021", "margin_pct_2021", "turnout_pct_2021",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(final[preview_cols].head(8).to_string(index=False))

    _log("\nNote: this dataset is wired into training as a sidecar. See")
    _log("data_loader.load_verified_model_dataset() -- columns are prefixed")
    _log("'verified_' in the feature matrix. Disable with TN2026_DISABLE_SIDECAR=1.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
