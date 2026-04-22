# Tamil Nadu Assembly Election Prediction 2026

Constituency-level seat projection for all **234 Tamil Nadu Assembly constituencies** in the 2026 election, across five alliance classes:

- **DMK Alliance** (DMK + Congress + CPI + CPI(M) + VCK + DMDK + MDMK + IUML)
- **AIADMK NDA** (AIADMK + BJP + PMK + AMMK)
- **TVK** (Tamilaga Vettri Kazhagam — Vijay)
- **NTK** (Naam Tamilar Katchi — Seeman)
- **OTHERS** (independents, rebels, minor parties)

## Stack

- **Backend**: Python 3.11+, PyTorch, scikit-learn, pandas
- **Frontend**: React + TypeScript + Vite + framer-motion
- **Model**: Dual-head MLP ensemble (15 folds, `RepeatedKFold(5, 3)`), trained on 234 constituencies with classification + vote-share regression heads

## Project layout

```
backend/
  config.py                    # Parties, districts, regions, keywords
  build_data_files.py          # dataset/*.csv -> data_files/*.csv (schema normalise)
  build_historical_results.py  # scrapes real per-AC 2016 & 2021 results
  create_dataset.py            # TN 2026 projection engine (real historical + swing)
  data_loader.py               # Merges all CSVs into 234-row training frame
  train.py                     # Training + ensemble prediction
  server.py                    # JSON API on :8001
  data/
    live_collectors.py         # X / YouTube / NewsAPI sentiment collectors
    sentiment_extractor.py     # Multilingual rule-based + model-based sentiment
  dataset/                     # Raw input CSVs (2016/2021/2024 results, electorate)
  data_files/                  # Pipeline-normalised CSVs + projection targets
  checkpoints/                 # Trained model weights (gitignored)
  predictions_2026.csv         # 234-row output with per-AC alliance probabilities
frontend/
  src/App.tsx                  # Dashboard UI
  src/services/api.ts          # Typed API client
  src/types/prediction.ts      # Party / PredictionRow / PredictionsMeta types
  src/components/              # PartyBadge, AnimatedKpiGrid
```

## How to run

### Backend

```bash
cd backend
python build_data_files.py           # only needed when dataset/ changes
python build_historical_results.py   # refresh real per-AC historical (optional)
python create_dataset.py             # regenerate 2026 projection targets
python train.py                      # train 15-fold ensemble -> predictions_2026.csv
python server.py                     # serve API on http://127.0.0.1:8001
```

API endpoints:
- `GET /api/health` — status + meta
- `GET /api/predictions` — 234 rows of per-AC probabilities
- `GET /api/predictions/meta` — seat counts + file integrity hash

### Frontend

```bash
cd frontend
npm install
npm run dev                          # vite dev server on :5173
```

Create `frontend/.env` from `.env.example` and set `VITE_API_BASE_URL=http://127.0.0.1:8001` for local development.

## Data flow

```
dataset/*.csv  (real historical + 2026 inputs)
     │
     ├─▶ build_data_files.py        (normalise -> data_files/)
     ├─▶ build_historical_results.py (scrape real per-AC 2016/2021 -> data_files/)
     │
     ▼
create_dataset.py  (real historical + 2024 LS swing + TVK/NTK/Others wildcards)
     │
     ▼
data_files/tamilnadu_assembly_2026.csv  (training targets)
     │
     ▼
train.py  (15-fold ensemble)
     │
     ▼
predictions_2026.csv  -> server.py  -> frontend
```

The pipeline cleanly separates **real historical data** (2016/2021 per-AC results), **synthesised 2026 projection** (swing + structural signals), and **live sentiment** (optional, via API keys). A curated `tamilnadu_assembly_2026.csv` can replace the generated projection targets without any code changes.

## Environment variables (backend)

Optional — only required if you want live sentiment ingestion:

```
X_BEARER_TOKEN=
YOUTUBE_API_KEY=
NEWS_API_KEY=
```

Put them in `backend/.env` (never committed — see `.gitignore`).
