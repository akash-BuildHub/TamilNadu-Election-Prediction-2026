"""
One-shot utility to transform backend/dataset/*.csv into the pipeline's
CSVs in backend/data_files/. Runs stateless: re-executing overwrites outputs.

This script is explicitly NOT part of the training pipeline. It exists only
to materialise the canonical CSV form the rest of the pipeline consumes
(column renames, alliance-row filters, percentage normalisation, etc.).
The training path reads exclusively from backend/data_files/.
"""

import os

import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BACKEND_DIR, "dataset")
DST_DIR = os.path.join(BACKEND_DIR, "data_files")
os.makedirs(DST_DIR, exist_ok=True)


# District-name spelling mapping:
#   constituency master file (authoritative)   <- state voters list spelling
DISTRICT_SPELLING_FIXES = {
    "Villupuram": "Viluppuram",
    "Kanyakumari": "Kanniyakumari",
    "Kanchipuram": "Kancheepuram",
    "Tiruppattur": "Tirupattur",
    "Nilgiris": "The Nilgiris",
}


# --------------------------------------------------------------------------
# 1. Direct xlsx -> csv dumps (column names kept as-is except minimal cleanup)
# --------------------------------------------------------------------------

def _dump_simple(src_name: str, dst_name: str, rename: dict | None = None) -> None:
    df = pd.read_csv(os.path.join(SRC_DIR, src_name))
    if rename:
        df = df.rename(columns=rename)
    df.to_csv(os.path.join(DST_DIR, dst_name), index=False)
    print(f"  {dst_name}: {df.shape[0]} rows, {df.shape[1]} cols")


# --------------------------------------------------------------------------
# 2. Constituency master (234 ACs)
# --------------------------------------------------------------------------

def build_constituency_master() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_assembly_constituency_2026.csv"))
    # Columns expected: ac_no, ac_name, reservation, district, region_5way,
    # is_reserved, election_cycle
    assert len(df) == 234, f"Expected 234 ACs, got {len(df)}"
    out = df[["ac_no", "ac_name", "district", "region_5way", "reservation", "is_reserved"]].copy()
    out.to_csv(os.path.join(DST_DIR, "tamilnadu_constituency_master_2026.csv"), index=False)
    print(f"  tamilnadu_constituency_master_2026.csv: {len(out)} rows")
    return out


# --------------------------------------------------------------------------
# 3. District demographics (38 districts) built from state voters list +
#    constituency master. Only sources we actually have.
# --------------------------------------------------------------------------

def build_district_demographics(master: pd.DataFrame) -> None:
    voters = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_state_voters_list_2026.csv"))
    voters = voters.rename(columns={
        "District": "district",
        "Total voters": "total_voters",
        "Men": "men_voters",
        "Women": "women_voters",
        "TG": "third_gender_voters",
    })
    voters["district"] = voters["district"].astype(str).str.strip()
    voters["district"] = voters["district"].replace(DISTRICT_SPELLING_FIXES)

    # Join ACs-per-district and reserved-per-district from the master
    per_district = master.groupby("district").agg(
        ac_count=("ac_no", "size"),
        reserved_count=("is_reserved", "sum"),
    ).reset_index()

    demo = voters.merge(per_district, on="district", how="outer")
    # Derived features the model can use directly
    demo["women_pct"] = (demo["women_voters"] / demo["total_voters"] * 100.0).round(4)
    demo["men_pct"] = (demo["men_voters"] / demo["total_voters"] * 100.0).round(4)
    demo["third_gender_pct"] = (demo["third_gender_voters"] / demo["total_voters"] * 100.0).round(6)
    demo["reserved_share_pct"] = (demo["reserved_count"] / demo["ac_count"] * 100.0).round(4)

    missing_in_voters = demo[demo["total_voters"].isna()]["district"].tolist()
    missing_in_master = demo[demo["ac_count"].isna()]["district"].tolist()
    if missing_in_voters:
        raise ValueError(f"Districts in master but missing in voters list: {missing_in_voters}")
    if missing_in_master:
        raise ValueError(f"Districts in voters list but missing in master: {missing_in_master}")

    demo = demo[[
        "district", "total_voters", "men_voters", "women_voters", "third_gender_voters",
        "men_pct", "women_pct", "third_gender_pct",
        "ac_count", "reserved_count", "reserved_share_pct",
    ]]
    demo.to_csv(os.path.join(DST_DIR, "tamilnadu_demographics.csv"), index=False)
    print(f"  tamilnadu_demographics.csv: {len(demo)} rows")


# --------------------------------------------------------------------------
# 4. Historical election results (state-level)
# --------------------------------------------------------------------------

def _normalise_election_file(src: str, dst: str, year: int, kind: str) -> None:
    """
    State-level xlsx -> CSV with columns:
      party, seats_won, votes, vote_share
    vote_share is stored as a decimal fraction (0-1) in the xlsx; we convert
    to percentage-scale (0-100) here to match how data_loader consumes it.
    """
    df = pd.read_csv(os.path.join(SRC_DIR, src))
    df = df.rename(columns={
        "Party / Alliance": "party",
        "Seats won": "seats_won",
        "Votes": "votes",
        "Vote share": "vote_share",
    })
    df["party"] = df["party"].astype(str).str.strip()

    # Filter out alliance roll-up rows ("DMK-led alliance total", etc.)
    # We keep only per-party rows so seat totals don't double-count.
    roll_up_mask = df["party"].str.contains("total", case=False, na=False) | \
                   df["party"].str.contains("alliance", case=False, na=False) | \
                   df["party"].str.contains("bloc", case=False, na=False) | \
                   df["party"].str.contains(r"\bNDA\b", regex=True, na=False) | \
                   df["party"].str.contains(r"\bAIADMK\+", regex=True, na=False)
    # Keep rows whose 'party' is exactly a single party name. Edge case:
    # "AIADMK-led/NDA total" is caught; bare "AIADMK" is not.
    df = df[~roll_up_mask].copy()

    # Normalise vote_share: xlsx stores 0.40 meaning 40%. Multiply to %-scale.
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce") * 100.0
    df["seats_won"] = pd.to_numeric(df["seats_won"], errors="coerce").fillna(0).astype(int)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype("int64")

    df["year"] = year
    df["election_type"] = kind

    df = df[["year", "election_type", "party", "seats_won", "votes", "vote_share"]]
    df.to_csv(os.path.join(DST_DIR, dst), index=False)
    print(f"  {dst}: {len(df)} rows (after roll-up filter)")


# --------------------------------------------------------------------------
# 5. Alliance seat-sharing (2026)
# --------------------------------------------------------------------------

def _normalise_alliance_sharing(src: str, dst: str, alliance_label: str) -> None:
    df = pd.read_csv(os.path.join(SRC_DIR, src))
    df = df.rename(columns={"Party": "party", "Exact seats": "seats"})
    df["party"] = df["party"].astype(str).str.strip()
    # Drop total / placeholder rows
    mask = df["party"].str.contains("total|subtotal|left for allies", case=False, na=False)
    df = df[~mask].copy()
    df["seats"] = pd.to_numeric(df["seats"], errors="coerce").fillna(0).astype(int)
    df["alliance"] = alliance_label
    df = df[["alliance", "party", "seats"]]
    df.to_csv(os.path.join(DST_DIR, dst), index=False)
    print(f"  {dst}: {len(df)} rows")


# --------------------------------------------------------------------------
# 6. Party-wise 2026 seat table (all fronts, one file)
# --------------------------------------------------------------------------

def build_party_seat_table_2026() -> None:
    df = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_assembly_election_party-wise_seat_table_2026.csv"))
    df = df.rename(columns={
        "Front / Status": "front",
        "Party": "party",
        "Seats": "seats",
        "Status": "status",
        "Front total": "front_total",
    })
    df["seats"] = pd.to_numeric(df["seats"], errors="coerce").fillna(0).astype(int)
    df["front_total"] = pd.to_numeric(df["front_total"], errors="coerce").fillna(0).astype(int)
    df.to_csv(os.path.join(DST_DIR, "tamilnadu_party_wise_seat_table_2026.csv"), index=False)
    print(f"  tamilnadu_party_wise_seat_table_2026.csv: {len(df)} rows")


# --------------------------------------------------------------------------
# 7. Alliance-level sentiment / confidence for 2026
#    (not in the dataset -- we build it from the main parties file + a
#    default label scheme so the model has a sentiment feature. This file
#    can be hand-edited later without touching code.)
# --------------------------------------------------------------------------

def build_sentiment_file() -> None:
    rows = [
        # party, confidence_pre_result   <- conservative defaults; ready to override
        {"party": "DMK_ALLIANCE", "confidence_pre_result": "Medium-High"},
        {"party": "AIADMK_NDA",   "confidence_pre_result": "Medium"},
        {"party": "TVK",          "confidence_pre_result": "Low-Medium"},
        {"party": "NTK",          "confidence_pre_result": "Low"},
        {"party": "OTHERS",       "confidence_pre_result": "Low"},
    ]
    pd.DataFrame(rows).to_csv(
        os.path.join(DST_DIR, "tamilnadu_sentiment_analysis_2026.csv"), index=False
    )
    print(f"  tamilnadu_sentiment_analysis_2026.csv: {len(rows)} rows (defaults - hand-editable)")


# --------------------------------------------------------------------------
# 8. Voter aggregates (single-value state-level facts)
# --------------------------------------------------------------------------

def build_voter_aggregates() -> None:
    # Electorate
    elec = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_total_electorate_2026.csv"))
    elec = elec.rename(columns={"Item": "metric", "Exact value": "value"})
    elec.to_csv(os.path.join(DST_DIR, "tamilnadu_electorate_total_2026.csv"), index=False)
    print(f"  tamilnadu_electorate_total_2026.csv: {len(elec)} rows")

    # Polling-day provisional turnout equivalent
    yet = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_people_yet_to_vote_2026.csv"))
    yet = yet.rename(columns={"Item": "metric", "Exact value": "value"})
    yet.to_csv(os.path.join(DST_DIR, "tamilnadu_people_yet_to_vote_2026.csv"), index=False)
    print(f"  tamilnadu_people_yet_to_vote_2026.csv: {len(yet)} rows")

    # First-time voters
    ftv = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_first-time_voters_2026.csv"))
    ftv = ftv.rename(columns={"Item": "metric", "Value": "count"})
    ftv.to_csv(os.path.join(DST_DIR, "tamilnadu_first_time_voters_2026.csv"), index=False)
    print(f"  tamilnadu_first_time_voters_2026.csv: {len(ftv)} rows")

    # Nominations
    nom = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_nominations_and_candidates_2026.csv"))
    nom = nom.rename(columns={"Item": "metric", "Exact value": "count"})
    nom.to_csv(os.path.join(DST_DIR, "tamilnadu_nominations_and_candidates_2026.csv"), index=False)
    print(f"  tamilnadu_nominations_and_candidates_2026.csv: {len(nom)} rows")

    # Gender-wise state aggregate
    gen = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_gender_wise_voters_2026.csv"))
    gen = gen.rename(columns={"Category": "category", "Exact value": "count"})
    gen.to_csv(os.path.join(DST_DIR, "tamilnadu_gender_wise_voters_2026.csv"), index=False)
    print(f"  tamilnadu_gender_wise_voters_2026.csv: {len(gen)} rows")


# --------------------------------------------------------------------------
# 9. Main parties + comparison table + past-10-years (for cross-checks)
# --------------------------------------------------------------------------

def build_cross_check_tables() -> None:
    main = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_main_parties_2026.csv"))
    main = main.rename(columns={"Front / Party": "front_or_party", "Leader": "leader"})
    main.to_csv(os.path.join(DST_DIR, "tamilnadu_main_parties_2026.csv"), index=False)
    print(f"  tamilnadu_main_parties_2026.csv: {len(main)} rows")

    cmp_df = pd.read_csv(os.path.join(SRC_DIR, "comparison_table.csv"))
    cmp_df = cmp_df.rename(columns={
        "Year": "year",
        "Election": "election",
        "Winner": "winner",
        "Winner seats": "winner_seats",
        "Main winner vote share": "winner_vote_share",
        "Runner-up": "runner_up",
        "Runner-up seats": "runner_up_seats",
        "Runner-up vote share": "runner_up_vote_share",
    })
    for col in ("winner_vote_share", "runner_up_vote_share"):
        cmp_df[col] = pd.to_numeric(cmp_df[col], errors="coerce") * 100.0
    cmp_df.to_csv(os.path.join(DST_DIR, "tamilnadu_election_comparison_table.csv"), index=False)
    print(f"  tamilnadu_election_comparison_table.csv: {len(cmp_df)} rows")

    hist = pd.read_csv(os.path.join(SRC_DIR, "tamilnadu_elections_past_10_years.csv"))
    hist = hist.rename(columns={
        "Year": "year",
        "Election": "election_type",
        "Seats": "seats",
        "Winner": "winner",
    })
    hist.to_csv(os.path.join(DST_DIR, "tamilnadu_elections_results_past_10_years.csv"), index=False)
    print(f"  tamilnadu_elections_results_past_10_years.csv: {len(hist)} rows")


def main():
    print(f"Building CSVs in {DST_DIR}")
    master = build_constituency_master()
    build_district_demographics(master)

    print("\nHistorical elections (state-level, Lok Sabha):")
    _normalise_election_file("tamilnadu_lok_sabha_election_2014.csv",
                             "tamilnadu_lok_sabha_election_2014.csv", 2014, "Lok Sabha")
    _normalise_election_file("tamilnadu_lok_sabha_election_2019.csv",
                             "tamilnadu_lok_sabha_election_2019.csv", 2019, "Lok Sabha")
    _normalise_election_file("tamilnadu_lok_sabha_election_2024.csv",
                             "tamilnadu_lok_sabha_election_2024.csv", 2024, "Lok Sabha")

    print("\nHistorical elections (state-level, Assembly):")
    _normalise_election_file("tamilnadu_assembly_election_2016.csv",
                             "tamilnadu_assembly_election_2016.csv", 2016, "Assembly")
    _normalise_election_file("tamilnadu_assembly_election_2021.csv",
                             "tamilnadu_assembly_election_2021.csv", 2021, "Assembly")

    print("\nAlliance seat sharing (2026):")
    _normalise_alliance_sharing(
        "tamilnadu_DMK_led_alliance_seat_sharing_2026.csv",
        "tamilnadu_DMK_ALLIANCE_seat_sharing_2026.csv",
        "DMK_ALLIANCE",
    )
    _normalise_alliance_sharing(
        "tamilnadu_AIADMK_led_NDA_seat_sharing_2026.csv",
        "tamilnadu_AIADMK_NDA_seat_sharing_2026.csv",
        "AIADMK_NDA",
    )

    print("\nParty-wise 2026 seat table:")
    build_party_seat_table_2026()

    print("\nVoter aggregates:")
    build_voter_aggregates()

    print("\nCross-check tables:")
    build_cross_check_tables()

    print("\nSentiment defaults:")
    build_sentiment_file()

    print("\nDone.")


if __name__ == "__main__":
    main()
