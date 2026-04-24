"""
Tamil Nadu Assembly Election 2026 - Training & Prediction Pipeline.

Flow:
  backend/data_files/*.csv  (merged by data_loader.py)
    -> feature extraction (categorical one-hots + numeric scaling)
    -> RepeatedKFold (5x3 = 15 folds)
    -> MLP ensemble with dual heads (classification + vote-share regression)
    -> ensemble_predict across all 15 models
    -> predictions_2026.csv

Usage:
    python train.py
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config import DISTRICTS, PARTIES, REGIONS
from data_loader import (
    VERIFIED_CATEG_COLS,
    VERIFIED_NUMERIC_COLS,
    VERIFIED_PARTY_VOCAB,
    load_training_dataframe,
)

warnings.filterwarnings("ignore")
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    parties: List[str] = field(default_factory=lambda: list(PARTIES))
    num_classes: int = len(PARTIES)

    # Architecture
    hidden_dim: int = 160
    num_layers: int = 2
    dropout: float = 0.22

    # Training
    batch_size: int = 48
    lr: float = 7e-4
    weight_decay: float = 0.02
    epochs: int = 260
    warmup_epochs: int = 18
    patience: int = 32
    label_smoothing: float = 0.1

    cls_weight: float = 0.55
    reg_weight: float = 0.45

    n_splits: int = 5
    n_repeats: int = 3

    max_class_weight: float = 5.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    # Per-AC historical
    "vote_share_2021", "margin_pct_2021",
    # District-level voter aggregates
    "total_voters", "men_pct", "women_pct", "third_gender_pct",
    "ac_count", "reserved_count", "reserved_share_pct",
    # Reservation flag
    "is_reserved",
    # Incumbent / runner-up state-level momentum
    "incumbent_ls_swing_2024_2019", "incumbent_ls_swing_2019_2014",
    "incumbent_as_swing_2021_2016",
    "runnerup_ls_swing_2024_2019", "runnerup_as_swing_2021_2016",
    # Sentiment + alliance structure
    "incumbent_sentiment", "challenger_sentiment",
    "incumbent_concentration", "challenger_concentration",
    "incumbent_breadth", "challenger_breadth",
    # State-level aggregates (broadcast constants)
    "state_turnout_pct", "state_first_time_voter_pct",
    "state_candidates_per_seat",
]

PERCENT_SCALE_COLS = {
    "men_pct", "women_pct", "third_gender_pct", "reserved_share_pct",
    # Sidecar (all prefixed "verified_") vote-shares and turnout are 0-100 scale
    "verified_turnout_pct_2016", "verified_turnout_pct_2021",
    "verified_margin_pct_2016", "verified_margin_pct_2021",
    "verified_dmk_vote_share_2016", "verified_aiadmk_vote_share_2016",
    "verified_bjp_vote_share_2016", "verified_congress_vote_share_2016",
    "verified_ntk_vote_share_2016", "verified_others_vote_share_2016",
    "verified_dmk_vote_share_2021", "verified_aiadmk_vote_share_2021",
    "verified_bjp_vote_share_2021", "verified_congress_vote_share_2021",
    "verified_ntk_vote_share_2021", "verified_others_vote_share_2021",
}
VOTER_COUNT_DIVISOR = 2_000_000.0
AC_COUNT_DIVISOR = 10.0
RESERVED_COUNT_DIVISOR = 5.0

RESERVATION_CATEGORIES = ["GEN", "SC", "ST"]


class ElectionDataset:
    """Builds the model-ready feature matrix from the CSV-driven DataFrame."""

    def __init__(self):
        self.config = Config()
        self.party_to_idx = {p: i for i, p in enumerate(self.config.parties)}

        print("Loading data from backend/data_files/ ...")
        self.df = load_training_dataframe()

        # Detect whether the verified sidecar is actually present on the
        # merged DataFrame. If yes, we append its features; if no (user set
        # TN2026_DISABLE_SIDECAR or the CSV was absent), we skip them --
        # so the same code works in both A/B legs.
        self._sidecar_present = all(
            c in self.df.columns for c in VERIFIED_NUMERIC_COLS + VERIFIED_CATEG_COLS
        )
        if self._sidecar_present:
            print("  sidecar: ACTIVE (verified historical features included)")
        else:
            print("  sidecar: INACTIVE (baseline feature set only)")

        self.features, self.labels, self.vote_shares, self.meta = self._build()

        n = len(self.features)
        dist = {p: int((self.labels == i).sum()) for i, p in enumerate(self.config.parties)}
        print(f"  Samples: {n} | Features: {self.features.shape[1]} | Classes: {dist}")

    def _row_features(self, row: pd.Series) -> list[float]:
        f: list[float] = []

        for party in self.config.parties:
            f.append(1.0 if row["winner_alliance_2016"] == party else 0.0)
        for party in self.config.parties:
            f.append(1.0 if row["winner_alliance_2021"] == party else 0.0)
        for party in self.config.parties:
            f.append(1.0 if row["runner_up_alliance_2021"] == party else 0.0)
        for r in REGIONS:
            f.append(1.0 if row["region_5way"] == r else 0.0)
        for d in DISTRICTS:
            f.append(1.0 if row["district"] == d else 0.0)
        for res in RESERVATION_CATEGORIES:
            f.append(1.0 if row["reservation"] == res else 0.0)

        for col in NUMERIC_FEATURES:
            v = float(row[col])
            if col == "total_voters":
                v /= VOTER_COUNT_DIVISOR
            elif col == "ac_count":
                v /= AC_COUNT_DIVISOR
            elif col == "reserved_count":
                v /= RESERVED_COUNT_DIVISOR
            elif col in PERCENT_SCALE_COLS:
                v /= 100.0
            f.append(v)

        # Sidecar features -- appended AFTER the baseline features so the
        # baseline feature indices are unchanged.
        if self._sidecar_present:
            # Numeric: same percent-scaling as the baseline path.
            for col in VERIFIED_NUMERIC_COLS:
                v = float(row[col])
                if col in PERCENT_SCALE_COLS:
                    v /= 100.0
                f.append(v)
            # Categorical: one-hot against VERIFIED_PARTY_VOCAB.
            for col in VERIFIED_CATEG_COLS:
                val = str(row[col])
                for party in VERIFIED_PARTY_VOCAB:
                    f.append(1.0 if val == party else 0.0)

        return f

    def _build(self):
        features, labels, vote_shares, meta = [], [], [], []

        for _, row in self.df.iterrows():
            features.append(self._row_features(row))

            label_name = row["proj_2026_winner"]
            if label_name not in self.party_to_idx:
                raise ValueError(
                    f"Unknown proj_2026_winner '{label_name}' for "
                    f"'{row['constituency']}'. Expected one of {list(self.party_to_idx)}."
                )
            labels.append(self.party_to_idx[label_name])

            vote_shares.append([
                row["proj_2026_dmk_alliance_pct"],
                row["proj_2026_aiadmk_nda_pct"],
                row["proj_2026_tvk_pct"],
                row["proj_2026_ntk_pct"],
                row["proj_2026_others_pct"],
            ])

            meta.append({
                "ac_no": int(row["ac_no"]),
                "constituency": row["constituency"],
                "district": row["district"],
            })

        features_arr = np.array(features, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int64)
        vote_shares_arr = np.array(vote_shares, dtype=np.float32)

        if np.isnan(features_arr).any():
            raise ValueError("NaN detected in feature matrix after CSV merge.")
        if np.isnan(vote_shares_arr).any():
            raise ValueError("NaN detected in vote-share targets after CSV merge.")

        sums = vote_shares_arr.sum(axis=1, keepdims=True)
        vote_shares_arr = vote_shares_arr / sums

        return features_arr, labels_arr, vote_shares_arr, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.LayerNorm(dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class ElectionModel(nn.Module):
    """Dual-head MLP: classifier + vote-share regressor."""

    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        h = config.hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(h, config.dropout) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(h)
        self.classifier = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, config.num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, config.num_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.classifier(x)
        vs_logits = self.regressor(x)
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "vs_logits": vs_logits,
            "vote_shares": F.softmax(vs_logits, dim=-1),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compute_class_weights(labels, num_classes, max_w=5.0):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = np.where(counts == 0, max_w, len(labels) / (num_classes * counts + 1e-6))
    return torch.FloatTensor(np.minimum(w, max_w))


def safe_save(state, path, retries=5):
    for i in range(retries):
        try:
            torch.save(state, path)
            return
        except RuntimeError:
            if i == retries - 1:
                raise
            time.sleep(0.5)


def train_fold(fold_idx, train_idx, val_idx, data: ElectionDataset, config: Config):
    dev = config.device
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data.features[train_idx]).astype(np.float32)
    X_va = scaler.transform(data.features[val_idx]).astype(np.float32)
    y_tr, y_va = data.labels[train_idx], data.labels[val_idx]
    vs_tr, vs_va = data.vote_shares[train_idx], data.vote_shares[val_idx]

    cw_np = compute_class_weights(y_tr, config.num_classes, config.max_class_weight).numpy()
    sampler = WeightedRandomSampler(
        torch.FloatTensor([cw_np[l] for l in y_tr]), len(y_tr), replacement=True
    )

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(vs_tr)),
        batch_size=config.batch_size, sampler=sampler, num_workers=0,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va), torch.from_numpy(vs_va)),
        batch_size=len(X_va), shuffle=False,
    )

    model = ElectionModel(X_tr.shape[1], config).to(dev)
    cls_criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y_tr, config.num_classes, config.max_class_weight).to(dev)
    )

    def reg_criterion(vs_logits, vs_target):
        log_probs = F.log_softmax(vs_logits, dim=-1)
        return -(vs_target * log_probs).sum(dim=-1).mean()

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_acc = 0.0
    wait = 0
    ckpt_path = os.path.join(_BACKEND_DIR, f"checkpoints/model_fold_{fold_idx}.pt")

    for epoch in range(config.epochs):
        model.train()
        for xb, yb, vsb in train_dl:
            xb, yb, vsb = xb.to(dev), yb.to(dev), vsb.to(dev)
            optimizer.zero_grad()
            out = model(xb)
            loss = (
                config.cls_weight * cls_criterion(out["logits"], yb)
                + config.reg_weight * reg_criterion(out["vs_logits"], vsb)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for xb, yb, vsb in val_dl:
                xb, yb, vsb = xb.to(dev), yb.to(dev), vsb.to(dev)
                out = model(xb)
                v_cls = cls_criterion(out["logits"], yb).item()
                v_reg = reg_criterion(out["vs_logits"], vsb).item()
                val_loss = config.cls_weight * v_cls + config.reg_weight * v_reg
                acc = (out["logits"].argmax(-1) == yb).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = acc
            wait = 0
            safe_save({"model": model.state_dict(), "scaler": scaler}, ckpt_path)
        else:
            wait += 1

        if wait >= config.patience:
            break

    best_epoch = epoch + 1 - wait
    print(f"  Fold {fold_idx + 1:2d}: val_loss={best_val_loss:.4f}  acc={best_acc:.4f}  "
          f"(best @ epoch {best_epoch})")
    return best_acc


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------


def ensemble_predict(
    data: ElectionDataset, config: Config, global_scaler: StandardScaler
) -> pd.DataFrame:
    print("\nGenerating ensemble predictions...")
    dev = config.device
    n_models = config.n_splits * config.n_repeats
    n_samples = len(data.labels)

    X_scaled = global_scaler.transform(data.features).astype(np.float32)
    X_tensor = torch.from_numpy(X_scaled).to(dev)

    all_probs = np.zeros((n_samples, config.num_classes))
    all_vs = np.zeros((n_samples, config.num_classes))

    for i in range(n_models):
        ckpt = torch.load(
            os.path.join(_BACKEND_DIR, f"checkpoints/model_fold_{i}.pt"),
            weights_only=False,
        )
        model = ElectionModel(X_scaled.shape[1], config).to(dev)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            out = model(X_tensor)
            all_probs += out["probs"].cpu().numpy()
            all_vs += out["vote_shares"].cpu().numpy()

    all_probs /= n_models
    all_vs /= n_models

    assert not np.isnan(all_probs).any(), "NaN in ensemble class probabilities"
    assert not np.isnan(all_vs).any(), "NaN in ensemble vote shares"
    assert np.allclose(all_probs.sum(axis=1), 1.0, atol=1e-4), \
        "Class probabilities do not sum to 1"
    assert np.allclose(all_vs.sum(axis=1), 1.0, atol=1e-4), \
        "Vote shares do not sum to 1"

    results = []
    for i in range(n_samples):
        probs = all_probs[i]
        vs = all_vs[i]
        pred_idx = int(np.argmax(probs))

        results.append({
            "ac_no": data.meta[i]["ac_no"],
            "constituency": data.meta[i]["constituency"],
            "district": data.meta[i]["district"],
            "predicted": config.parties[pred_idx],
            "confidence": float(probs[pred_idx]),
            "DMK_ALLIANCE": float(probs[0]),
            "AIADMK_NDA": float(probs[1]),
            "TVK": float(probs[2]),
            "NTK": float(probs[3]),
            "OTHERS": float(probs[4]),
            "vs_DMK_ALLIANCE": float(vs[0]),
            "vs_AIADMK_NDA": float(vs[1]),
            "vs_TVK": float(vs[2]),
            "vs_NTK": float(vs[3]),
            "vs_OTHERS": float(vs[4]),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: pd.DataFrame, config: Config):
    total = len(results)
    counts = results["predicted"].value_counts()
    majority = 118  # 234/2 + 1

    print("\n" + "=" * 60)
    print("  FINAL SEAT PROJECTION (234 constituencies)")
    print("=" * 60)

    for party in config.parties:
        n = counts.get(party, 0)
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {party:14s}: {n:3d} seats ({pct:5.1f}%) {bar}")

    winner = counts.idxmax()
    winner_n = counts.max()
    status = "MAJORITY" if winner_n >= majority else "HUNG ASSEMBLY"
    print(f"\n  Projected winner: {winner} ({winner_n} seats) - {status}")

    avg_conf = results["confidence"].mean()
    high_conf = (results["confidence"] >= 0.60).sum()
    print(f"\n  Avg winning-party probability: {avg_conf:.1%}")
    print(f"  High-confidence seats (>=60% probability): {high_conf}/{total}")

    print("\n  DISTRICT-WISE:")
    print("  " + "-" * 56)
    pivot = results.groupby(["district", "predicted"]).size().unstack(fill_value=0)
    for p in config.parties:
        if p not in pivot.columns:
            pivot[p] = 0
    pivot = pivot[config.parties]
    pivot["Total"] = pivot.sum(axis=1)
    pivot["Winner"] = pivot[config.parties].idxmax(axis=1)
    print(pivot.to_string(header=True))

    print("\n" + "=" * 60)


def main():
    config = Config()
    os.makedirs(os.path.join(_BACKEND_DIR, "checkpoints"), exist_ok=True)

    print("=" * 60)
    print("  TAMIL NADU ELECTION 2026 - MODEL TRAINING")
    print(f"  Device: {config.device}")
    print("=" * 60)

    data = ElectionDataset()

    print(f"\nTraining {config.n_splits}x{config.n_repeats} = "
          f"{config.n_splits * config.n_repeats} fold ensemble...\n")

    cv = RepeatedKFold(n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=42)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data.features)):
        acc = train_fold(fold_idx, train_idx, val_idx, data, config)
        fold_accs.append(acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n  CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    global_scaler = StandardScaler().fit(data.features)
    results = ensemble_predict(data, config, global_scaler)

    out_path = os.path.join(_BACKEND_DIR, "predictions_2026.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    print_summary(results, config)


if __name__ == "__main__":
    main()
