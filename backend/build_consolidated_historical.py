"""Consolidate verified Tamil Nadu Assembly results into one file.

Output: backend/dataset/historical_results/tamilnadu_constituency_results_2011_2021.csv

Sources (already in repo, verified against TCPD/Lok Dhaba mirror of ECI Form 21):
  * backend/dataset/opencity_tn_2016.csv  (candidate-level, 232 ACs)
  * backend/dataset/opencity_tn_2021.csv  (candidate-level, 234 ACs)

2011 is intentionally NOT included yet. The user will drop
backend/dataset/opencity_tn_2011.csv later, at which point this script
adds another 234 rows without any other change.

2016 ACs 134 (Aravakurichi) and 178 (Thanjavur) had their general-election
polls countermanded by ECI under the cash-for-votes probe and were held as
bye-elections on 19 Nov 2016. Both were won by AIADMK per ECI's bye-election
notification. TCPD's bulk file does not carry the bye-election rows; we emit
those two with winner_party=AIADMK_NDA / runner_party=DMK_ALLIANCE and
vote counts = 0 (votes intentionally blank: not in our source CSV).
"""

import os
import pandas as pd

from config import DATASET_DIR

OPENCITY = {
    2016: os.path.join(DATASET_DIR, "opencity_tn_2016.csv"),
    2021: os.path.join(DATASET_DIR, "opencity_tn_2021.csv"),
    # 2011 path reserved; uncomment when the file lands:
    # 2011: os.path.join(DATASET_DIR, "opencity_tn_2011.csv"),
}
MASTER_PATH = os.path.join(DATASET_DIR, "tamilnadu_assembly_constituency_2026.csv")
OUT_DIR = os.path.join(DATASET_DIR, "historical_results")
OUT_PATH = os.path.join(OUT_DIR, "tamilnadu_constituency_results_2011_2021.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# TCPD party-code -> canonical short name
TCPD_PARTY = {
    "ADMK": "AIADMK", "AIADMK": "AIADMK",
    "DMK": "DMK",
    "INC": "INC",
    "BJP": "BJP",
    "CPI": "CPI",
    "CPM": "CPI(M)", "CPI(M)": "CPI(M)",
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

# 4-class output spec: DMK_ALLIANCE, AIADMK_NDA, TVK, OTHERS.
# NTK collapses to OTHERS (no NTK winners or runners-up in 2016 or 2021).
ALLIANCE = {
    2011: {  # reserved - finalize when 2011 source arrives
        "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "PMK": "DMK_ALLIANCE",
        "VCK": "DMK_ALLIANCE", "IUML": "DMK_ALLIANCE",
        "AIADMK": "AIADMK_NDA", "DMDK": "AIADMK_NDA",
        "CPI": "AIADMK_NDA", "CPI(M)": "AIADMK_NDA", "MDMK": "AIADMK_NDA",
        "BJP": "OTHERS", "AMMK": "OTHERS", "MNM": "OTHERS",
        "NTK": "OTHERS", "IND": "OTHERS", "TVK": "TVK",
    },
    2016: {
        "AIADMK": "AIADMK_NDA",
        "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "IUML": "DMK_ALLIANCE",
        "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE", "VCK": "DMK_ALLIANCE",
        "PMK": "OTHERS", "DMDK": "OTHERS", "MDMK": "OTHERS", "BJP": "OTHERS",
        "AMMK": "OTHERS", "MNM": "OTHERS", "NTK": "OTHERS", "IND": "OTHERS",
        "TVK": "TVK",
    },
    2021: {
        "AIADMK": "AIADMK_NDA", "PMK": "AIADMK_NDA", "BJP": "AIADMK_NDA",
        "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "CPI": "DMK_ALLIANCE",
        "CPI(M)": "DMK_ALLIANCE", "VCK": "DMK_ALLIANCE", "MDMK": "DMK_ALLIANCE",
        "IUML": "DMK_ALLIANCE",
        "AMMK": "OTHERS", "MNM": "OTHERS", "NTK": "OTHERS", "DMDK": "OTHERS",
        "IND": "OTHERS", "TVK": "TVK",
    },
}


def to_alliance(party_short, year):
    return ALLIANCE[year].get(party_short, "OTHERS")


def norm_party(raw):
    if raw is None or pd.isna(raw):
        return "Other"
    return TCPD_PARTY.get(str(raw).strip().upper(), "Other")


def build_year(year, opencity_path, master):
    df = pd.read_csv(opencity_path)
    # Poll_No == 0 is the general election. Non-zero rows are bye-elections
    # that TCPD bundles under the same year (e.g. 2016 AC 195 has a 2019 by-poll
    # tagged Year=2016, Poll_No=1). Exclude those - we want only the general poll.
    df = df[df["Poll_No"] == 0]
    pos1 = df[df["Position"] == 1].set_index("Constituency_No")
    pos2 = df[df["Position"] == 2].set_index("Constituency_No")

    rows = []
    missing = []
    for _, m in master.iterrows():
        ac_no = int(m["ac_no"])
        ac_name = m["ac_name"]
        district = m["district"]

        if ac_no in pos1.index and ac_no in pos2.index:
            w = pos1.loc[ac_no]
            r = pos2.loc[ac_no]
            wp = norm_party(w["Party"])
            rp = norm_party(r["Party"])
            margin = w["Margin"]
            if pd.isna(margin):
                margin = int(w["Votes"]) - int(r["Votes"])
            rows.append({
                "year": year,
                "ac_no": ac_no,
                "ac_name": ac_name,
                "district": district,
                "winner": str(w["Candidate"]).strip(),
                "winner_party": to_alliance(wp, year),
                "runner_up": str(r["Candidate"]).strip(),
                "runner_party": to_alliance(rp, year),
                "winner_votes": int(w["Votes"]),
                "runner_votes": int(r["Votes"]),
                "margin_votes": int(margin),
                "winner_vote_share": round(float(w["Vote_Share_Percentage"]), 2),
                "runner_vote_share": round(float(r["Vote_Share_Percentage"]), 2),
            })
        else:
            missing.append((ac_no, ac_name))
            # 2016 bye-election fill (verified via ECI bye-election notification).
            if year == 2016 and ac_no in (134, 174):
                rows.append({
                    "year": 2016, "ac_no": ac_no, "ac_name": ac_name, "district": district,
                    "winner": "", "winner_party": "AIADMK_NDA",
                    "runner_up": "", "runner_party": "DMK_ALLIANCE",
                    "winner_votes": 0, "runner_votes": 0, "margin_votes": 0,
                    "winner_vote_share": 0.0, "runner_vote_share": 0.0,
                })

    return pd.DataFrame(rows), missing


def main():
    master = pd.read_csv(MASTER_PATH)[["ac_no", "ac_name", "district"]]
    print(f"Master spine: {len(master)} ACs")

    frames = []
    for year, path in sorted(OPENCITY.items()):
        if not os.path.exists(path):
            print(f"[skip] {year}: source file not present ({path})")
            continue
        df, missing = build_year(year, path, master)
        print(f"{year}: {len(df)} rows  (missing in source: {len(missing)})")
        for ac_no, name in missing:
            note = " -> filled from ECI bye-election fact" if (year == 2016 and ac_no in (134, 174)) else " -> NOT FILLED"
            print(f"   missing AC {ac_no} {name}{note}")
        frames.append(df)

    if not frames:
        raise SystemExit("No source files found - nothing written.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["year", "ac_no"]).reset_index(drop=True)

    # Sanity: no duplicates
    dups = out.duplicated(subset=["year", "ac_no"]).sum()
    assert dups == 0, f"duplicate (year, ac_no) rows: {dups}"

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"Total rows: {len(out)}")
    print("Per-year row counts:")
    print(out.groupby("year").size().to_string())
    print("\nWinner alliance distribution per year:")
    print(out.groupby(["year", "winner_party"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
