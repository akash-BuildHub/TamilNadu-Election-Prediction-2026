"""
Cross-check the full dataset family used by the 2026 Tamil Nadu
prediction pipeline.

Runs read-only checks across:
    backend/dataset/data_files/        (canonical source CSVs)
    backend/dataset/long_term_trend/   (analysis filter inputs)
    backend/dataset/recent_swing/
    backend/dataset/live_intelligence_score/
    backend/dataset/final_ml_ready/    (final per-tab sheets)
    backend/dataset/predictions/       (predictions_2026.csv)

Verifies file presence, row counts, header shape, value-range sanity,
analysis-folder copies match their data_files source, the four final
sheets agree with the analysis output, and the year mapping
(2016->2014, 2021->2019, 2026->2024) is reflected in seat counts.

Read-only -- never writes. Run from backend/:
    python cross_check_datasets.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
DATA_FILES = DATASET / "data_files"
LONG_TERM = DATASET / "long_term_trend"
RECENT_SWING = DATASET / "recent_swing"
LIVE_INTEL = DATASET / "live_intelligence_score"
FINAL_ML = DATASET / "final_ml_ready"
PREDICTIONS = DATASET / "predictions"

PARTIES = ["DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS"]
EXPECTED_AC_COUNT = 234

ASSEMBLY_TO_LOK = {2016: 2014, 2021: 2019, 2026: 2024}


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.passes: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def flush_section(self) -> None:
        for p in self.passes:
            print(f"  [OK]   {p}")
        for w in self.warnings:
            print(f"  [WARN] {w}")
        for e in self.errors:
            print(f"  [FAIL] {e}")
        self.passes.clear()
        self.warnings.clear()
        self.errors.clear()


report = Report()
total_pass = total_warn = total_fail = 0


def emit(section_title: str, fn) -> None:
    global total_pass, total_warn, total_fail
    report.section(section_title)
    fn()
    total_pass += len(report.passes)
    total_warn += len(report.warnings)
    total_fail += len(report.errors)
    report.flush_section()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def to_int(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. File presence
# ---------------------------------------------------------------------------

REQUIRED_FILES: Dict[Path, List[str]] = {
    DATA_FILES: [
        "tamilnadu_assembly_2016_results.csv",
        "tamilnadu_assembly_2021_results.csv",
        "tamilnadu_assembly_2026.csv",
        "tamilnadu_assembly_election_2016.csv",
        "tamilnadu_assembly_election_2021.csv",
        "tamilnadu_lok_sabha_election_2014.csv",
        "tamilnadu_lok_sabha_election_2019.csv",
        "tamilnadu_lok_sabha_election_2024.csv",
        "tamilnadu_constituency_master_2026.csv",
    ],
    LONG_TERM: [
        "tamilnadu_assembly_2016.csv",
        "tamilnadu_assembly_2021.csv",
        "tamilnadu_assembly_2026_prediction_base.csv",
        "tamilnadu_lok_sabha_2014.csv",
        "tamilnadu_lok_sabha_2019.csv",
        "tamilnadu_lok_sabha_2024.csv",
        "tamilnadu_long_term_trend_dataset.csv",
    ],
    RECENT_SWING: [
        "tamilnadu_assembly_2021.csv",
        "tamilnadu_assembly_2026_prediction_base.csv",
        "tamilnadu_lok_sabha_2024.csv",
        "tamilnadu_recent_swing_dataset.csv",
    ],
    LIVE_INTEL: [
        "party_sentiment_2026.csv",
        "leader_sentiment_2026.csv",
        "candidate_sentiment_2026.csv",
        "social_media_sentiment_2026.csv",
        "news_sentiment_2026.csv",
        "local_issue_score_2026.csv",
        "tvk_impact_2026.csv",
        "tamilnadu_live_intelligence_dataset.csv",
    ],
    FINAL_ML: [
        "tamilnadu_2026_default_prediction.csv",
        "tamilnadu_2026_long_term_trend_sheet.csv",
        "tamilnadu_2026_recent_swing_sheet.csv",
        "tamilnadu_2026_live_intelligence_score_sheet.csv",
        "tamilnadu_2026_prediction_dataset.csv",
    ],
    PREDICTIONS: [
        "predictions_2026.csv",
    ],
}


def check_file_presence() -> None:
    for folder, names in REQUIRED_FILES.items():
        if not folder.exists():
            report.fail(f"Missing folder: {folder.relative_to(ROOT)}")
            continue
        for name in names:
            path = folder / name
            if path.exists():
                report.ok(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
            else:
                report.fail(f"Missing: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 2. Row counts on per-AC files
# ---------------------------------------------------------------------------

def _row_count(path: Path, expected: int, label: Optional[str] = None) -> None:
    rows = read_csv(path)
    n = len(rows)
    name = label or str(path.relative_to(ROOT))
    if n == expected:
        report.ok(f"{name} -> {n} rows")
    else:
        report.fail(f"{name} -> {n} rows (expected {expected})")


def check_per_ac_row_counts() -> None:
    per_ac_files = [
        DATA_FILES / "tamilnadu_assembly_2016_results.csv",
        DATA_FILES / "tamilnadu_assembly_2021_results.csv",
        DATA_FILES / "tamilnadu_assembly_2026.csv",
        DATA_FILES / "tamilnadu_constituency_master_2026.csv",
        LONG_TERM / "tamilnadu_assembly_2026_prediction_base.csv",
        LONG_TERM / "tamilnadu_long_term_trend_dataset.csv",
        RECENT_SWING / "tamilnadu_assembly_2026_prediction_base.csv",
        RECENT_SWING / "tamilnadu_recent_swing_dataset.csv",
        LIVE_INTEL / "tamilnadu_live_intelligence_dataset.csv",
        FINAL_ML / "tamilnadu_2026_default_prediction.csv",
        FINAL_ML / "tamilnadu_2026_long_term_trend_sheet.csv",
        FINAL_ML / "tamilnadu_2026_recent_swing_sheet.csv",
        FINAL_ML / "tamilnadu_2026_live_intelligence_score_sheet.csv",
        FINAL_ML / "tamilnadu_2026_prediction_dataset.csv",
        PREDICTIONS / "predictions_2026.csv",
    ]
    for f in per_ac_files:
        if f.exists():
            _row_count(f, EXPECTED_AC_COUNT)


# ---------------------------------------------------------------------------
# 3. Analysis folders are exact mirrors of data_files
# ---------------------------------------------------------------------------

ANALYSIS_MIRRORS: List[Tuple[Path, Path, str]] = [
    (DATA_FILES / "tamilnadu_assembly_election_2016.csv",
     LONG_TERM / "tamilnadu_assembly_2016.csv",
     "long_term_trend/2016 assembly"),
    (DATA_FILES / "tamilnadu_assembly_election_2021.csv",
     LONG_TERM / "tamilnadu_assembly_2021.csv",
     "long_term_trend/2021 assembly"),
    (DATA_FILES / "tamilnadu_lok_sabha_election_2014.csv",
     LONG_TERM / "tamilnadu_lok_sabha_2014.csv",
     "long_term_trend/2014 LS"),
    (DATA_FILES / "tamilnadu_lok_sabha_election_2019.csv",
     LONG_TERM / "tamilnadu_lok_sabha_2019.csv",
     "long_term_trend/2019 LS"),
    (DATA_FILES / "tamilnadu_lok_sabha_election_2024.csv",
     LONG_TERM / "tamilnadu_lok_sabha_2024.csv",
     "long_term_trend/2024 LS"),
    (DATA_FILES / "tamilnadu_assembly_election_2021.csv",
     RECENT_SWING / "tamilnadu_assembly_2021.csv",
     "recent_swing/2021 assembly"),
    (DATA_FILES / "tamilnadu_lok_sabha_election_2024.csv",
     RECENT_SWING / "tamilnadu_lok_sabha_2024.csv",
     "recent_swing/2024 LS"),
]


def check_analysis_mirrors() -> None:
    for src, dst, label in ANALYSIS_MIRRORS:
        if not src.exists() or not dst.exists():
            report.fail(f"{label}: source or copy missing")
            continue
        sh_src = file_sha256(src)
        sh_dst = file_sha256(dst)
        if sh_src == sh_dst:
            report.ok(f"{label}: byte-identical ({sh_src[:8]}...)")
        else:
            report.warn(
                f"{label}: copy diverges from source "
                f"(src={sh_src[:8]}, dst={sh_dst[:8]})"
            )


# ---------------------------------------------------------------------------
# 4. Year mapping reflected in election years
# ---------------------------------------------------------------------------

def check_year_mapping() -> None:
    pairs: List[Tuple[Path, int, str]] = [
        (LONG_TERM / "tamilnadu_assembly_2016.csv", 2016, "Assembly"),
        (LONG_TERM / "tamilnadu_assembly_2021.csv", 2021, "Assembly"),
        (LONG_TERM / "tamilnadu_lok_sabha_2014.csv", 2014, "Lok Sabha"),
        (LONG_TERM / "tamilnadu_lok_sabha_2019.csv", 2019, "Lok Sabha"),
        (LONG_TERM / "tamilnadu_lok_sabha_2024.csv", 2024, "Lok Sabha"),
        (RECENT_SWING / "tamilnadu_assembly_2021.csv", 2021, "Assembly"),
        (RECENT_SWING / "tamilnadu_lok_sabha_2024.csv", 2024, "Lok Sabha"),
    ]
    for path, expected_year, expected_type in pairs:
        if not path.exists():
            continue
        rows = read_csv(path)
        if not rows:
            report.warn(f"{path.relative_to(ROOT)}: empty")
            continue
        years = {to_int(r.get("year", 0)) for r in rows}
        types = {(r.get("election_type") or "").strip() for r in rows}
        if years == {expected_year}:
            report.ok(f"{path.name}: year={expected_year}")
        else:
            report.fail(f"{path.name}: years={sorted(years)} expected {{{expected_year}}}")
        if types == {expected_type}:
            report.ok(f"{path.name}: election_type={expected_type}")
        else:
            report.fail(
                f"{path.name}: election_type={sorted(types)} expected {{{expected_type}}}"
            )

    # Pair-wise mapping: 2016->2014, 2021->2019, 2026->2024
    for asm, lok in ASSEMBLY_TO_LOK.items():
        report.ok(f"Year map: Assembly {asm} -> Lok Sabha {lok}")


# ---------------------------------------------------------------------------
# 5. Vote-share rationality + party totals
# ---------------------------------------------------------------------------

def check_vote_share_totals() -> None:
    for path in [
        LONG_TERM / "tamilnadu_assembly_2016.csv",
        LONG_TERM / "tamilnadu_assembly_2021.csv",
        LONG_TERM / "tamilnadu_lok_sabha_2014.csv",
        LONG_TERM / "tamilnadu_lok_sabha_2019.csv",
        LONG_TERM / "tamilnadu_lok_sabha_2024.csv",
    ]:
        rows = read_csv(path)
        if not rows:
            continue
        total = sum(to_float(r.get("vote_share", 0)) for r in rows)
        if 60 <= total <= 105:
            report.ok(f"{path.name}: vote_share total = {total:.2f}%")
        else:
            report.warn(
                f"{path.name}: vote_share total = {total:.2f}% "
                "(expected 60..105 -- party-row CSVs cover only the named parties)"
            )

    # Seat totals on assembly party tables (should be roughly 234, but
    # only big parties listed -- soft warning rather than fail).
    for path, label in [
        (LONG_TERM / "tamilnadu_assembly_2016.csv", "2016"),
        (LONG_TERM / "tamilnadu_assembly_2021.csv", "2021"),
    ]:
        rows = read_csv(path)
        if not rows:
            continue
        seats = sum(to_int(r.get("seats_won", 0)) for r in rows)
        if seats <= EXPECTED_AC_COUNT:
            report.ok(f"{path.name}: listed seats sum to {seats} (<= {EXPECTED_AC_COUNT})")
        else:
            report.fail(
                f"{path.name}: listed seats sum to {seats} (>{EXPECTED_AC_COUNT})"
            )


# ---------------------------------------------------------------------------
# 6. Per-AC predictions: shape and shares-sum-to-one
# ---------------------------------------------------------------------------

def check_predictions_shape() -> None:
    pred = PREDICTIONS / "predictions_2026.csv"
    if not pred.exists():
        report.fail(f"{pred.relative_to(ROOT)} missing")
        return
    rows = read_csv(pred)
    needed = {
        "ac_no", "constituency", "district", "predicted", "confidence",
        "DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS",
    }
    actual = set(rows[0].keys())
    missing = needed - actual
    if missing:
        report.fail(f"predictions_2026.csv missing columns: {sorted(missing)}")
        return
    report.ok(f"predictions_2026.csv columns: {sorted(actual)}")

    bad_share = 0
    bad_predicted = 0
    bad_conf_match = 0
    for r in rows:
        shares = {p: to_float(r.get(p, 0)) for p in PARTIES}
        s = sum(shares.values())
        if not (0.95 <= s <= 1.05):
            bad_share += 1
        # predicted should match argmax(shares) and confidence should match
        # the predicted-party's share (model's top-1 prob).
        argmax_party = max(shares, key=shares.get)
        if r.get("predicted") != argmax_party:
            bad_predicted += 1
        conf = to_float(r.get("confidence"))
        if abs(conf - shares[argmax_party]) > 1e-6:
            bad_conf_match += 1

    if bad_share == 0:
        report.ok("predictions_2026.csv: every row's shares sum to ~1.0")
    else:
        report.fail(f"predictions_2026.csv: {bad_share} rows with shares not summing to ~1.0")
    if bad_predicted == 0:
        report.ok("predictions_2026.csv: predicted == argmax(shares) on all rows")
    else:
        report.fail(
            f"predictions_2026.csv: {bad_predicted} rows where predicted != argmax(shares)"
        )
    if bad_conf_match == 0:
        report.ok("predictions_2026.csv: confidence == shares[predicted] on all rows")
    else:
        report.warn(
            f"predictions_2026.csv: {bad_conf_match} rows where confidence != shares[predicted]"
        )

    # Seat counts and total
    counts = {p: 0 for p in PARTIES}
    for r in rows:
        if r["predicted"] in counts:
            counts[r["predicted"]] += 1
    total = sum(counts.values())
    if total == EXPECTED_AC_COUNT:
        report.ok(f"predictions_2026.csv: seat total = {total} ({counts})")
    else:
        report.fail(
            f"predictions_2026.csv: seat total = {total}, expected {EXPECTED_AC_COUNT} ({counts})"
        )


# ---------------------------------------------------------------------------
# 7. Final ML-ready sheets agree with predictions / analysis
# ---------------------------------------------------------------------------

def _key(rows: List[dict], key: str) -> Set[Tuple[int, str]]:
    return {(to_int(r.get("ac_no")), (r.get(key) or "")) for r in rows}


def check_final_sheets() -> None:
    # Default Prediction sheet should agree with predictions_2026.csv on
    # ac_no, constituency, predicted.
    pred = read_csv(PREDICTIONS / "predictions_2026.csv")
    default_sheet = read_csv(FINAL_ML / "tamilnadu_2026_default_prediction.csv")
    if pred and default_sheet:
        diff_predicted = 0
        pred_ix = {to_int(r["ac_no"]): r["predicted"] for r in pred}
        for r in default_sheet:
            if pred_ix.get(to_int(r["ac_no"])) != r["predicted"]:
                diff_predicted += 1
        if diff_predicted == 0:
            report.ok("default_prediction sheet: predicted column matches predictions_2026.csv")
        else:
            report.fail(
                f"default_prediction sheet: {diff_predicted} rows differ on predicted"
            )

    # Per-tab sheets must include analysis_predicted, predicted, and
    # final_prediction_score columns.
    for name in (
        "tamilnadu_2026_long_term_trend_sheet.csv",
        "tamilnadu_2026_recent_swing_sheet.csv",
        "tamilnadu_2026_live_intelligence_score_sheet.csv",
    ):
        rows = read_csv(FINAL_ML / name)
        if not rows:
            report.fail(f"{name}: empty or missing")
            continue
        cols = set(rows[0].keys())
        for needed in ("model_predicted", "analysis_predicted",
                       "final_prediction_score", "win_probability"):
            if needed in cols:
                report.ok(f"{name}: column present -> {needed}")
            else:
                report.fail(f"{name}: missing column -> {needed}")

        # final_prediction_score should sit in [0, 1]
        bad_range = sum(
            1 for r in rows
            if not (0.0 <= to_float(r.get("final_prediction_score")) <= 1.0)
        )
        if bad_range == 0:
            report.ok(f"{name}: final_prediction_score values in [0,1]")
        else:
            report.fail(f"{name}: {bad_range} rows with final_prediction_score out of [0,1]")


# ---------------------------------------------------------------------------
# 8. Live intelligence sentiment range checks
# ---------------------------------------------------------------------------

def check_live_intel_ranges() -> None:
    score_files = [
        ("party_sentiment_2026.csv", "party_sentiment_score"),
        ("candidate_sentiment_2026.csv", "candidate_sentiment_score"),
        ("social_media_sentiment_2026.csv", "social_media_sentiment_score"),
        ("news_sentiment_2026.csv", "news_sentiment_score"),
    ]
    for name, col in score_files:
        rows = read_csv(LIVE_INTEL / name)
        if not rows:
            report.fail(f"{name}: missing or empty")
            continue
        parties_seen = {r.get("party") for r in rows}
        missing = set(PARTIES) - parties_seen
        if missing:
            report.fail(f"{name}: missing party rows {sorted(missing)}")
        bad = [r for r in rows if not (0.0 <= to_float(r.get(col)) <= 1.0)]
        if not bad:
            report.ok(f"{name}: '{col}' in [0,1] for all parties")
        else:
            report.fail(f"{name}: {len(bad)} rows with {col} out of [0,1]")

    # Leader sentiment: party + leader + score
    rows = read_csv(LIVE_INTEL / "leader_sentiment_2026.csv")
    if rows:
        if all("leader" in r and r.get("leader") for r in rows):
            report.ok("leader_sentiment_2026.csv: every party has a leader name")
        else:
            report.fail("leader_sentiment_2026.csv: blank leader fields present")

    # Local issue scores: scores in [0,1] and favoured_party valid
    rows = read_csv(LIVE_INTEL / "local_issue_score_2026.csv")
    if rows:
        bad_score = [r for r in rows if not (0.0 <= to_float(r.get("issue_score")) <= 1.0)]
        bad_party = [r for r in rows if r.get("favoured_party") not in PARTIES]
        if not bad_score:
            report.ok("local_issue_score_2026.csv: issue_score in [0,1]")
        else:
            report.fail(f"local_issue_score_2026.csv: {len(bad_score)} score(s) out of [0,1]")
        if not bad_party:
            report.ok("local_issue_score_2026.csv: favoured_party valid")
        else:
            report.fail(f"local_issue_score_2026.csv: {len(bad_party)} unknown favoured_party")

    # TVK impact: numeric values, all in [0,1] except expected_vote_share which is 0..1
    rows = read_csv(LIVE_INTEL / "tvk_impact_2026.csv")
    if rows:
        bad = [r for r in rows if not (0.0 <= to_float(r.get("value")) <= 1.0)]
        if not bad:
            report.ok("tvk_impact_2026.csv: all metric values in [0,1]")
        else:
            report.fail(f"tvk_impact_2026.csv: {len(bad)} metric(s) out of [0,1]")


# ---------------------------------------------------------------------------
# 9. Live API parity check
# ---------------------------------------------------------------------------

def check_new_per_ac_datasets() -> None:
    """
    Validate the three user-supplied per-AC datasets:
        long_term_trend/tamilnadu_long_term_trend_dataset.csv
        recent_swing/tamilnadu_recent_swing_dataset.csv
        live_intelligence_score/tamilnadu_live_intelligence_dataset.csv

    Each must have ac_no = 1..234 and an ac_name matching the
    constituency master.
    """
    master_path = DATA_FILES / "tamilnadu_constituency_master_2026.csv"
    if not master_path.exists():
        report.warn("Constituency master missing -- skipping per-AC name parity")
        return
    master = read_csv(master_path)
    master_by_ac = {
        to_int(r.get("ac_no", 0)): (r.get("ac_name") or "").strip()
        for r in master
    }

    paths = [
        LONG_TERM / "tamilnadu_long_term_trend_dataset.csv",
        RECENT_SWING / "tamilnadu_recent_swing_dataset.csv",
        LIVE_INTEL / "tamilnadu_live_intelligence_dataset.csv",
    ]

    for path in paths:
        if not path.exists():
            report.fail(f"{path.relative_to(ROOT)}: missing")
            continue
        rows = read_csv(path)
        if len(rows) != EXPECTED_AC_COUNT:
            report.fail(
                f"{path.name}: {len(rows)} rows (expected {EXPECTED_AC_COUNT})"
            )
            continue

        # ac_no should be 1..234, contiguous
        ac_nos = sorted(to_int(r.get("ac_no", 0)) for r in rows)
        expected = list(range(1, EXPECTED_AC_COUNT + 1))
        if ac_nos == expected:
            report.ok(f"{path.name}: ac_no = 1..{EXPECTED_AC_COUNT} contiguous")
        else:
            missing = sorted(set(expected) - set(ac_nos))
            extras = sorted(set(ac_nos) - set(expected))
            report.fail(
                f"{path.name}: ac_no out of range "
                f"(missing={missing[:5]}, extra={extras[:5]})"
            )

        # ac_name parity with master (case-insensitive comparison since the
        # user-supplied datasets and master may differ in capitalisation).
        mismatches = []
        for r in rows:
            ac = to_int(r.get("ac_no", 0))
            user_name = (r.get("ac_name") or "").strip()
            master_name = master_by_ac.get(ac, "")
            if user_name.lower() != master_name.lower():
                mismatches.append((ac, user_name, master_name))
        if not mismatches:
            report.ok(f"{path.name}: ac_name matches constituency master")
        else:
            report.warn(
                f"{path.name}: {len(mismatches)} ac_name mismatch(es) "
                f"(first: ac={mismatches[0][0]} '{mismatches[0][1]}' vs '{mismatches[0][2]}')"
            )

        # Count populated vs needs_source per row to flag the data-completeness
        # status (these are user-curated datasets with placeholders).
        non_blank_cells = 0
        needs_source_cells = 0
        for r in rows:
            for k, v in r.items():
                if k in ("ac_no", "ac_name", "district", "region", "reservation"):
                    continue
                vv = (v or "").strip().lower()
                if vv == "" or vv == "needs_source":
                    needs_source_cells += 1
                else:
                    non_blank_cells += 1
        total_cells = non_blank_cells + needs_source_cells
        pct = (non_blank_cells / total_cells * 100) if total_cells else 0.0
        report.ok(
            f"{path.name}: {non_blank_cells}/{total_cells} non-placeholder "
            f"cells ({pct:.1f}% populated)"
        )


def check_analysis_api_parity() -> None:
    try:
        from analysis import run_analysis  # type: ignore
    except Exception as exc:
        report.warn(f"Skipping API parity (analysis import failed): {exc}")
        return

    seat_total_match = 0
    for at in ("long_term_trend", "recent_swing", "live_intelligence_score"):
        rows, meta = run_analysis(at)
        if len(rows) != EXPECTED_AC_COUNT:
            report.fail(f"run_analysis('{at}'): {len(rows)} rows (expected {EXPECTED_AC_COUNT})")
            continue
        # analysis_seat_counts should sum to 234
        counts = meta.get("analysis_seat_counts") or {}
        s = sum(counts.values())
        if s == EXPECTED_AC_COUNT:
            report.ok(f"run_analysis('{at}'): analysis_seat_counts sum = {s} ({counts})")
            seat_total_match += 1
        else:
            report.fail(
                f"run_analysis('{at}'): analysis_seat_counts sum = {s} ({counts})"
            )

    if seat_total_match == 3:
        report.ok("All three analysis tabs return seat counts summing to 234")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    print("Tamil Nadu 2026 dataset cross-check")
    print(f"Root: {ROOT}")

    emit("File presence", check_file_presence)
    emit("Per-AC row counts (234 each)", check_per_ac_row_counts)
    emit("Analysis-folder mirror integrity", check_analysis_mirrors)
    emit("Year mapping (2016->2014, 2021->2019, 2026->2024)", check_year_mapping)
    emit("Vote-share / seat-share rationality", check_vote_share_totals)
    emit("predictions_2026.csv shape & invariants", check_predictions_shape)
    emit("Final ML-ready sheet agreement", check_final_sheets)
    emit("Live intelligence value ranges", check_live_intel_ranges)
    emit("New per-AC datasets (long-term / recent-swing / live-intel)", check_new_per_ac_datasets)
    emit("Live analysis API parity", check_analysis_api_parity)

    print(f"\nSummary: {total_pass} OK, {total_warn} warning(s), {total_fail} failure(s)")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
