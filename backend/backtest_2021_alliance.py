"""
Alliance-level historical backtest: train on 2016 + structural features,
predict the real 2021 winning alliance at each of 234 constituencies.

Why alliance-level:
  Party-level prediction (see backtest_2021.py) is unrealistically harsh
  because parties move across fronts between elections (BJP joined AIADMK
  in 2021, MDMK moved into DMK front, etc.). For seat-level forecasting,
  the useful target is which alliance's candidate won, not which party
  branded the ticket.

Alliance mapping (applied to the real winner_party_2021):
  DMK_ALLIANCE   -> DMK, INC, VCK, CPI, CPI(M), MDMK, IUML
  AIADMK_ALLIANCE -> AIADMK, BJP, PMK, DMDK
  OTHERS          -> NTK, AMMK, MNM, IND, everything else

Input features (STRICTLY no 2021 leakage):
  Numeric: 2016 turnout, 2016 margin, 6 party vote shares from 2016
  Categorical: 2016 winner_party, 2016 runner_up_party, district, region,
               reservation
  Target:  winner_alliance_2021 (derived from the real winner_party_2021)

Three models compared with identical feature matrix + splits:
  - RandomForestClassifier (baseline)
  - GradientBoostingClassifier (usually strong on tabular)
  - LogisticRegression with scaling (interpretable linear baseline)

Outputs to backend/dataset/backtests/:
  backtest_2021_alliance_predictions.csv          - best-model per-AC preds
  backtest_2021_alliance_metrics.json             - all models, CV + holdout
  backtest_2021_alliance_feature_importance.csv   - RF importance, ranked
  backtest_2021_alliance_confusion_matrix.csv     - labels x labels, best model

Standalone. Does NOT modify train.py.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from config import BACKTESTS_DIR, DATASET_DIR

BACKEND_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(DATASET_DIR) / "tn_model_dataset_updated.csv"
OUT_DIR = Path(BACKTESTS_DIR)

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

# Alliance mapping used to derive the 2021 target label.
ALLIANCE_2021 = {
    "DMK": "DMK_ALLIANCE",
    "INC": "DMK_ALLIANCE",
    "VCK": "DMK_ALLIANCE",
    "CPI": "DMK_ALLIANCE",
    "CPI(M)": "DMK_ALLIANCE",
    "MDMK": "DMK_ALLIANCE",
    "IUML": "DMK_ALLIANCE",
    "AIADMK": "AIADMK_ALLIANCE",
    "BJP": "AIADMK_ALLIANCE",
    "PMK": "AIADMK_ALLIANCE",
    "DMDK": "AIADMK_ALLIANCE",   # DMDK was a fringe ally; 0 seats in 2021
    "NTK": "OTHERS",
    "AMMK": "OTHERS",
    "MNM": "OTHERS",
    "IND": "OTHERS",
    "Other": "OTHERS",
}
ALLIANCE_CLASSES = ["DMK_ALLIANCE", "AIADMK_ALLIANCE", "OTHERS"]

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
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df[want].copy()


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    ac_info = df[["ac_no", "ac_name"]].copy()

    raw_target = df[TARGET_COL].astype("object")
    valid = raw_target.notna() & (raw_target.astype(str).str.strip() != "needs_source")
    valid &= raw_target.astype(str).str.strip() != ""
    dropped = int((~valid).sum())
    if dropped:
        _log(f"  dropped {dropped} rows with missing/needs_source target")
    df = df.loc[valid].reset_index(drop=True)
    ac_info = ac_info.loc[valid].reset_index(drop=True)

    # Numeric: per-column median fill (covers the 2 byelection ACs)
    X_num = df[NUMERIC_FEATS].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0.0)

    # Categorical: UNKNOWN for missing / needs_source / empty
    X_cat = df[CATEG_FEATS].copy()
    for c in CATEG_FEATS:
        s = X_cat[c].astype("object")
        s = s.where(~s.isna(), "UNKNOWN")
        s = s.replace({"needs_source": "UNKNOWN", "": "UNKNOWN"})
        X_cat[c] = s.astype(str).str.strip().replace({"": "UNKNOWN"})

    X_cat_oh = pd.get_dummies(X_cat, prefix=CATEG_FEATS, dummy_na=False)
    X = pd.concat([X_num.reset_index(drop=True),
                   X_cat_oh.reset_index(drop=True)], axis=1)

    # Map party -> alliance label
    raw_party = df[TARGET_COL].astype(str).str.strip()
    y_alliance = raw_party.map(ALLIANCE_2021).fillna("OTHERS")

    return X, y_alliance, ac_info


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def make_models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                max_iter=3000, C=1.0, class_weight="balanced",
                random_state=RANDOM_STATE, solver="lbfgs",
            )),
        ]),
    }


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series, k: int) -> dict:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "y_pred": y_pred,
    }


def evaluate_holdout(model, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_te)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _log("=" * 72)
    _log("  ALLIANCE-LEVEL BACKTEST - 2016 features -> winner_alliance_2021")
    _log("=" * 72)

    OUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    _log(f"\nRows loaded: {len(df)}")

    X, y, ac_info = preprocess(df)
    _log(f"Feature matrix: {X.shape}  "
         f"(numeric={len(NUMERIC_FEATS)}, "
         f"categorical one-hots={X.shape[1] - len(NUMERIC_FEATS)})")

    vc = y.value_counts()
    _log(f"\nClass distribution (winner_alliance_2021):")
    for cls in ALLIANCE_CLASSES:
        _log(f"  {cls:16s} {int(vc.get(cls, 0)):3d}")

    baseline_cls = vc.idxmax()
    baseline_acc = vc.max() / len(y)
    _log(f"\nBaseline (always-{baseline_cls}): "
         f"{vc.max()}/{len(y)} = {baseline_acc:.4f}")

    # Stratified K -- minimum class size sets the ceiling on K.
    min_class = vc[vc > 0].min()
    k = min(5, int(min_class))
    _log(f"\nStratified CV: K={k}  (min non-empty class size = {min_class})")

    models = make_models()
    results: dict = {}
    for name, model in models.items():
        _log(f"\n--- Model: {name} ---")
        # Fresh instance per eval to avoid cross_val_predict mutations
        cv_res = evaluate_cv(models[name], X, y, k=k)
        ho_res = evaluate_holdout(make_models()[name], X, y, test_size=0.2)
        _log(f"  CV:       acc={cv_res['accuracy']:.4f}  "
             f"macro_f1={cv_res['macro_f1']:.4f}  "
             f"weighted_f1={cv_res['weighted_f1']:.4f}")
        _log(f"  Holdout:  acc={ho_res['accuracy']:.4f}  "
             f"macro_f1={ho_res['macro_f1']:.4f}  "
             f"weighted_f1={ho_res['weighted_f1']:.4f}  "
             f"(n_test={ho_res['n_test']})")
        results[name] = {"cv": cv_res, "holdout": ho_res}

    # Pick best model by CV accuracy for the detailed report
    best_name = max(results, key=lambda n: results[n]["cv"]["accuracy"])
    _log(f"\nBest model by CV accuracy: {best_name}")
    best_cv = results[best_name]["cv"]

    # Per-class report for the best model
    labels = sorted(y.unique())
    best_report = classification_report(
        y, best_cv["y_pred"], labels=labels, output_dict=True, zero_division=0,
    )

    _log(f"\n--- Per-class CV metrics ({best_name}) ---")
    _log(f"  {'alliance':16s} {'precision':>10s} {'recall':>10s} "
         f"{'f1':>10s} {'support':>8s}")
    for cls in labels:
        r = best_report.get(cls, {})
        _log(f"  {cls:16s} "
             f"{r.get('precision', 0):10.3f} "
             f"{r.get('recall', 0):10.3f} "
             f"{r.get('f1-score', 0):10.3f} "
             f"{int(r.get('support', 0)):8d}")

    # RF feature importance -- consistent across runs for interpretability
    rf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y)
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    _log(f"\n--- Feature importance (RandomForest on full data) ---")
    _log("Top 15:")
    for _, row in fi.head(15).iterrows():
        _log(f"  {int(row['rank']):2d}. {row['feature']:40s} {row['importance']:.4f}")

    # Confusion matrix from best model's CV predictions
    cm = confusion_matrix(y, best_cv["y_pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true"

    _log(f"\n--- Confusion matrix ({best_name}, CV predictions) ---")
    _log(cm_df.to_string())

    # Per-AC predictions using the best model
    preds_df = pd.DataFrame({
        "ac_no": ac_info["ac_no"].values,
        "ac_name": ac_info["ac_name"].values,
        "true_alliance_2021": y.values,
        "predicted_alliance_2021_cv": best_cv["y_pred"],
        "correct": (y.values == best_cv["y_pred"]).astype(int),
    })
    miscls = preds_df[preds_df["correct"] == 0]
    _log(f"\nMisclassified constituencies ({best_name}, CV): "
         f"{len(miscls)} / {len(preds_df)}")
    if len(miscls):
        _log("First 20:")
        for _, row in miscls.head(20).iterrows():
            _log(f"  ac_no={int(row['ac_no']):3d}  {row['ac_name']:20s}  "
                 f"true={row['true_alliance_2021']:16s}  "
                 f"predicted={row['predicted_alliance_2021_cv']}")

    # Write outputs
    preds_df.to_csv(OUT_DIR / "backtest_2021_alliance_predictions.csv", index=False)
    fi.to_csv(OUT_DIR / "backtest_2021_alliance_feature_importance.csv", index=False)
    cm_df.to_csv(OUT_DIR / "backtest_2021_alliance_confusion_matrix.csv")

    metrics_payload = {
        "target": "winner_alliance_2021 (derived from real winner_party_2021)",
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "alliance_classes": ALLIANCE_CLASSES,
        "class_distribution": {
            cls: int(vc.get(cls, 0)) for cls in ALLIANCE_CLASSES
        },
        "baseline_always_majority_accuracy": float(baseline_acc),
        "cv_strategy": f"StratifiedKFold(K={k}, shuffle=True, seed={RANDOM_STATE})",
        "holdout_strategy": f"Stratified 80/20 (seed={RANDOM_STATE})",
        "models": {
            name: {
                "cv": {kk: v for kk, v in r["cv"].items() if kk != "y_pred"},
                "holdout": r["holdout"],
            }
            for name, r in results.items()
        },
        "best_model_by_cv_accuracy": best_name,
        "best_model_per_class_metrics": {
            cls: {
                "precision": float(best_report[cls]["precision"]),
                "recall":    float(best_report[cls]["recall"]),
                "f1":        float(best_report[cls]["f1-score"]),
                "support":   int(best_report[cls]["support"]),
            }
            for cls in labels if cls in best_report
        },
        "top_features": fi.head(20).to_dict(orient="records"),
        "n_misclassified_cv": int(len(miscls)),
    }
    with open(OUT_DIR / "backtest_2021_alliance_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2, default=float)

    _log("\nOutputs:")
    for name in (
        "backtest_2021_alliance_predictions.csv",
        "backtest_2021_alliance_metrics.json",
        "backtest_2021_alliance_feature_importance.csv",
        "backtest_2021_alliance_confusion_matrix.csv",
    ):
        p = OUT_DIR / name
        _log(f"  {p.relative_to(BACKEND_DIR)}  ({p.stat().st_size} bytes)")

    _log("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
