"""
Historical backtest: train on 2016 AC-level features, predict the real
2021 winner party at each of 234 constituencies.

This breaks the circular validation in train.py by:
  - Using ONLY 2016 columns as input features (vote shares, turnout,
    margin, winner/runner-up party, plus structural columns:
    district / region / reservation).
  - Using the real, ECI-sourced winner_party_2021 as the label.
  - Never exposing any 2021 vote-share, margin, or turnout to the model.

Standalone. Does NOT modify the main training pipeline.

Outputs written to backend/dataset/backtests/:
  backtest_2021_predictions.csv       - per-AC true vs predicted, correct flag
  backtest_2021_metrics.json          - accuracy, F1, per-class report, cv + holdout
  backtest_2021_feature_importance.csv - RF importances, ranked
  backtest_2021_confusion_matrix.csv  - labels x labels matrix from CV predictions

Usage:
  python backend/backtest_2021.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from config import BACKTESTS_DIR, DATASET_DIR

BACKEND_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(DATASET_DIR) / "tn_model_dataset_updated.csv"
OUT_DIR = Path(BACKTESTS_DIR)

# Input features -- strictly 2016 + structural. No 2021 leakage.
NUMERIC_FEATS = [
    "turnout_pct_2016", "margin_pct_2016",
    "dmk_vote_share_2016", "aiadmk_vote_share_2016", "bjp_vote_share_2016",
    "congress_vote_share_2016", "ntk_vote_share_2016", "others_vote_share_2016",
]
CATEG_FEATS = [
    "winner_party_2016", "runner_up_party_2016",
    "district", "region", "reservation",
]
TARGET_COL = "winner_party_2021"

RANDOM_STATE = 42


def _log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Run: python backend/build_model_dataset.py"
        )
    df = pd.read_csv(DATASET_PATH)
    want = ["ac_no", "ac_name"] + NUMERIC_FEATS + CATEG_FEATS + [TARGET_COL]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. "
                         f"Rebuild with build_model_dataset.py.")
    return df[want].copy()


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns (X, y, ac_info) where ac_info carries ac_no + ac_name for
    row-tracing in the prediction CSV.

    Missing-value policy:
      - numeric cells: per-column median (handles the 2 byelection ACs in 2016)
      - categorical cells (including literal "needs_source" or empty): "UNKNOWN"
      - target cells that are "needs_source" or NaN: dropped before training
    """
    ac_info = df[["ac_no", "ac_name"]].copy()

    # Target cleanup: drop rows without a real 2021 winner label
    raw_target = df[TARGET_COL].astype("object")
    valid_mask = raw_target.notna() & (raw_target.astype(str).str.strip() != "needs_source")
    valid_mask &= raw_target.astype(str).str.strip() != ""
    dropped = int((~valid_mask).sum())
    if dropped:
        _log(f"  dropped {dropped} rows with missing/needs_source target")
    df = df.loc[valid_mask].reset_index(drop=True)
    ac_info = ac_info.loc[valid_mask].reset_index(drop=True)

    # Numeric
    X_num = df[NUMERIC_FEATS].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    # Any column that is entirely NaN -> 0 (shouldn't happen for our cols)
    X_num = X_num.fillna(0.0)

    # Categorical
    X_cat = df[CATEG_FEATS].copy()
    for c in CATEG_FEATS:
        s = X_cat[c].astype("object")
        s = s.where(~s.isna(), "UNKNOWN")
        s = s.replace({"needs_source": "UNKNOWN", "": "UNKNOWN"})
        X_cat[c] = s.astype(str).str.strip().replace({"": "UNKNOWN"})

    X_cat_oh = pd.get_dummies(X_cat, prefix=CATEG_FEATS, dummy_na=False)

    X = pd.concat([X_num.reset_index(drop=True),
                   X_cat_oh.reset_index(drop=True)], axis=1)

    y = df[TARGET_COL].astype(str).str.strip()

    return X, y, ac_info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _pick_cv_k(y: pd.Series, max_k: int = 5) -> int:
    """
    Stratified-K-Fold requires every class to have at least K samples. We
    pick the largest K in [2, max_k] that satisfies that, falling back to
    2 if the smallest class still doesn't fit.
    """
    min_class = y.value_counts().min()
    k = min(max_k, int(min_class))
    return max(2, k)


def cv_evaluate(X: pd.DataFrame, y: pd.Series, k: int) -> dict:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
    )
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return {
        "strategy": f"StratifiedKFold(K={k}, shuffle=True, seed={RANDOM_STATE})",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "y_pred": y_pred,
    }


def holdout_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
    """
    Stratified 80/20 holdout. Classes with only 1 member can't be stratified,
    so in that edge case we fall back to a random split with a warning.
    """
    min_class = y.value_counts().min()
    stratify = y if min_class >= 2 else None
    if stratify is None:
        _log("  [holdout] min class = 1, falling back to non-stratified split")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        random_state=RANDOM_STATE, stratify=stratify,
    )
    clf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "strategy": f"Stratified 80/20 holdout (seed={RANDOM_STATE})",
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_te)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _log("=" * 70)
    _log("  HISTORICAL BACKTEST - 2016 features -> 2021 winner_party")
    _log("=" * 70)

    OUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    _log(f"\nRows loaded: {len(df)}")

    X, y, ac_info = preprocess(df)
    _log(f"Feature matrix: {X.shape}  (numeric={len(NUMERIC_FEATS)}, "
         f"categorical one-hots={X.shape[1] - len(NUMERIC_FEATS)})")
    _log(f"Labels: {y.nunique()} unique classes")
    _log(f"\nClass distribution (winner_party_2021):")
    vc = y.value_counts()
    for k_, v_ in vc.items():
        _log(f"  {k_:10s} {v_:3d}")

    # CV
    k = _pick_cv_k(y, max_k=5)
    _log(f"\n--- Cross-validated evaluation (K={k}) ---")
    cv_result = cv_evaluate(X, y, k=k)
    _log(f"  accuracy     : {cv_result['accuracy']:.4f}")
    _log(f"  macro F1     : {cv_result['macro_f1']:.4f}")
    _log(f"  weighted F1  : {cv_result['weighted_f1']:.4f}")

    # Holdout
    _log(f"\n--- Holdout evaluation ---")
    ho_result = holdout_evaluate(X, y, test_size=0.2)
    _log(f"  accuracy     : {ho_result['accuracy']:.4f}")
    _log(f"  macro F1     : {ho_result['macro_f1']:.4f}")
    _log(f"  weighted F1  : {ho_result['weighted_f1']:.4f}")
    _log(f"  test size    : {ho_result['n_test']}")

    # Train a final model on full data for feature importance
    _log(f"\n--- Feature importance (trained on all 234 rows) ---")
    final_clf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
    )
    final_clf.fit(X, y)
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": final_clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    _log("Top 15 features:")
    for _, row in fi.head(15).iterrows():
        _log(f"  {int(row['rank']):2d}. {row['feature']:40s} {row['importance']:.4f}")

    # Per-class metrics table
    _log(f"\n--- Per-class CV metrics ---")
    _log(f"  {'class':12s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}")
    report = cv_result["classification_report"]
    for cls in sorted(y.unique()):
        r = report.get(cls, {})
        _log(f"  {cls:12s} "
             f"{r.get('precision', 0):10.3f} "
             f"{r.get('recall', 0):10.3f} "
             f"{r.get('f1-score', 0):10.3f} "
             f"{int(r.get('support', 0)):8d}")

    # Confusion matrix from CV predictions
    labels = sorted(y.unique())
    cm = confusion_matrix(y, cv_result["y_pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true"

    # Per-AC predictions
    preds_df = pd.DataFrame({
        "ac_no": ac_info["ac_no"].values,
        "ac_name": ac_info["ac_name"].values,
        "true_party_2021": y.values,
        "predicted_party_2021_cv": cv_result["y_pred"],
        "correct": (y.values == cv_result["y_pred"]).astype(int),
    })

    miscls = preds_df[preds_df["correct"] == 0].copy()
    _log(f"\nMisclassified constituencies: {len(miscls)} / {len(preds_df)}")
    if len(miscls):
        _log("First 15 misclassifications:")
        for _, row in miscls.head(15).iterrows():
            _log(f"  ac_no={int(row['ac_no']):3d}  {row['ac_name']:20s}  "
                 f"true={row['true_party_2021']:8s}  "
                 f"predicted={row['predicted_party_2021_cv']}")

    # Write outputs
    preds_df.to_csv(OUT_DIR / "backtest_2021_predictions.csv", index=False)
    fi.to_csv(OUT_DIR / "backtest_2021_feature_importance.csv", index=False)
    cm_df.to_csv(OUT_DIR / "backtest_2021_confusion_matrix.csv")
    metrics_payload = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "class_distribution": {str(k_): int(v_) for k_, v_ in vc.items()},
        "cv": {
            "strategy": cv_result["strategy"],
            "accuracy": cv_result["accuracy"],
            "macro_f1": cv_result["macro_f1"],
            "weighted_f1": cv_result["weighted_f1"],
        },
        "holdout": ho_result,
        "per_class_metrics": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in sorted(y.unique()) if cls in report
        },
        "top_features": fi.head(20).to_dict(orient="records"),
        "n_misclassified": int(len(miscls)),
    }
    with open(OUT_DIR / "backtest_2021_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2, default=float)

    _log("\nOutputs:")
    for name in (
        "backtest_2021_predictions.csv",
        "backtest_2021_metrics.json",
        "backtest_2021_feature_importance.csv",
        "backtest_2021_confusion_matrix.csv",
    ):
        p = OUT_DIR / name
        _log(f"  {p.relative_to(BACKEND_DIR)}  ({p.stat().st_size} bytes)")

    _log("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
