# Model Validation Summary — Tamil Nadu 2026 Prediction

_This file is the single source of truth for how to interpret
prediction confidence. Re-run `python backend/write_model_validation.py`
after retraining._

## Headline numbers

| Metric | Value | What it measures |
|---|---|---|
| `train.py` synthetic-label CV accuracy | **0.9104 ± 0.1037** | How well the neural net reproduces `proj_2026_winner`, a synthetic label built by `create_dataset.py` using 2021 AC-level winner as its strong prior. |
| Party-level historical backtest CV accuracy | **0.6325** | 2016 features → real 2021 winner party (8 classes, RandomForest, K=2 stratified). |
| Party-level holdout accuracy | 0.6596 | stratified 80/20 holdout, same setup. |
| Alliance-level historical backtest CV accuracy | **0.7607** | 2016 features → real 2021 winning alliance (3 classes, RandomForest, K=5 stratified). |
| Alliance-level holdout accuracy | 0.7234 | stratified 80/20 holdout, same setup. |
| Alliance majority baseline (always DMK-led) | 0.6795 | naive ‘always predict the majority class’ baseline, alliance level. |
| Party majority baseline (always DMK) | 0.5684 | naive ‘always predict DMK’ baseline, party level. |

## Why the 91% number is not real forecast accuracy

`train.py` reports its cross-validation accuracy against the label
`proj_2026_winner`, which is **not** a real election outcome. That
label is synthesised inside `create_dataset.py` using 2021
constituency-level winners as the anchor, then perturbed by state
swing, alliance supply caps, and a small amount of noise. The
verified sidecar features added by `data_loader.py` also carry 2021
AC-level data. Since the input features and the label both derive
from the same 2021 base, the model is effectively being graded on
reproducing projection logic — a reproduction-of-heuristics score,
not a prediction score.

## Use these numbers for real-world reliability

- Historical backtest (party level): **63.25%**
  — about **+6.4 pp** over the always-DMK baseline.
- Historical backtest (alliance level): **76.07%**
  — about **+8.1 pp** over the always-DMK-alliance baseline.
- Recommended range to cite in any report: **63–76%**,
  depending on whether the target is party-level or alliance-level.

## One-line validation note (used by the API and the validated CSV)

> Internal synthetic-label CV accuracy: 91.04%. Historical backtest accuracy: 63.25% party-level and 76.07% alliance-level. Treat constituency-level predictions as directional, not guaranteed.

## Confidence columns in dataset/predictions/predictions_2026.csv

The `confidence` column is the top-1 predicted-party probability
from the model's softmax. It is a relative model-confidence score,
**not a calibrated probability of the real-world event**. The
validated CSV (`dataset/predictions/predictions_2026_validated.csv`)
explicitly tags every row with
`confidence_type = "relative_model_confidence_not_true_probability"` to make this unambiguous
to downstream consumers.

## Changing these numbers

Any of the three validation numbers can be refreshed by re-running
the corresponding script, then editing the constants at the top of
`backend/write_model_validation.py` and re-running this script.

- `train.py` → `TRAIN_CV_ACCURACY`, `TRAIN_CV_STD`
- `backtest_2021.py` → `PARTY_BACKTEST_CV`, `PARTY_BACKTEST_HOLDOUT`
- `backtest_2021_alliance.py` → `ALLIANCE_BACKTEST_CV`, `ALLIANCE_BACKTEST_HOLDOUT`
