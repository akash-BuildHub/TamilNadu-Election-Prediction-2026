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
backend/                       # Source code only
  config.py                    # Path constants + parties/districts/regions/keywords
  build_data_files.py          # dataset/*.csv -> dataset/data_files/*.csv
  build_historical_results.py  # real per-AC 2016 & 2021 results
  build_model_dataset.py       # TCPD/OpenCity -> tn_model_dataset_updated.csv
  create_dataset.py            # TN 2026 projection engine (real historical + swing)
  data_loader.py               # Merges CSVs into 234-row training frame
  train.py                     # Training + ensemble prediction
  backtest_2021.py             # Historical party-level backtest
  backtest_2021_alliance.py    # Historical alliance-level backtest
  write_model_validation.py    # Validation summary + validated predictions
  server.py                    # JSON API on :8001
  data/                        # Live sentiment collectors (optional feature)
  utils/                       # Visualisation helpers
  checkpoints/                 # Trained model weights (gitignored)
  dataset/                     # ALL data artifacts live here
    tamilnadu_*.csv            # Hand-curated source CSVs
    opencity_tn_201{6,1}.csv   # Downloaded TCPD-schema CSVs
    tn_model_dataset_*.csv     # build_model_dataset.py outputs
    data_files/                # Pipeline-normalised CSVs + projection targets
    predictions/               # predictions_2026.csv, predictions_2026_validated.csv
    backtests/                 # backtest_2021_* outputs (metrics, CM, preds, feat imp)
    validation/                # model_validation_summary.md + .json
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
python build_model_dataset.py        # verified dataset (TCPD/OpenCity -> dataset/)
python build_data_files.py           # normalise source CSVs -> dataset/data_files/
python build_historical_results.py   # real per-AC 2016/2021 -> dataset/data_files/
python create_dataset.py             # 2026 projection targets -> dataset/data_files/
python train.py                      # 15-fold ensemble -> dataset/predictions/predictions_2026.csv
python write_model_validation.py     # validation summary + validated CSV
python server.py                     # serve API on http://127.0.0.1:8001
```

API endpoints:
- `GET /api/health` — status + meta
- `GET /api/predictions` — 234 rows of per-AC probabilities
- `GET /api/predictions/meta` — seat counts + file integrity hash + validation note
- `GET /api/sentiment` — optional live social sentiment (see below)
- `GET /api/sentiment/health` — diagnostic: module import, key presence, flag state

### Frontend

```bash
cd frontend
npm install
npm run dev                          # vite dev server on :5173
```

Create `frontend/.env` from `.env.example` and set `VITE_API_BASE_URL=http://127.0.0.1:8001` for local development.

## Data flow

```
dataset/*.csv  (hand-curated + TCPD/OpenCity-downloaded)
     │
     ├─▶ build_data_files.py         (normalise -> dataset/data_files/)
     ├─▶ build_historical_results.py (scrape real per-AC 2016/2021 -> dataset/data_files/)
     ├─▶ build_model_dataset.py      (verified sidecar -> dataset/tn_model_dataset_updated.csv)
     │
     ▼
create_dataset.py  (real historical + 2024 LS swing + TVK/NTK/Others wildcards)
     │
     ▼
dataset/data_files/tamilnadu_assembly_2026.csv  (training targets)
     │
     ▼
train.py  (15-fold ensemble)
     │
     ▼
dataset/predictions/predictions_2026.csv  -> server.py  -> frontend
                          │
                          └─▶ write_model_validation.py (+confidence disclaimer)
                              -> dataset/predictions/predictions_2026_validated.csv
                              -> dataset/validation/model_validation_summary.{md,json}
```

The pipeline cleanly separates **real historical data** (2016/2021 per-AC results), **synthesised 2026 projection** (swing + structural signals), and **live sentiment** (optional, via API keys). A curated `tamilnadu_assembly_2026.csv` can replace the generated projection targets without any code changes.

## Optional: live sentiment

Live social sentiment (X / YouTube / NewsAPI) is **optional and future-ready**.
It is **not used by training or prediction** — the model trains and serves
without any API keys. Enabling it only affects the `/api/sentiment` endpoint.

To enable:

```
# backend/.env
ENABLE_LIVE_SENTIMENT=1
X_BEARER_TOKEN=...       # optional — any one key is enough
YOUTUBE_API_KEY=...
NEWS_API_KEY=...
```

Behaviour when any of these are missing:

| `ENABLE_LIVE_SENTIMENT` | keys present | `/api/sentiment` returns |
|---|---|---|
| `0` (default) | — | HTTP 200, `enabled: false`, warning with setup instructions |
| `1` | none | HTTP 200, `enabled: false`, warning about missing keys |
| `1` | at least one | HTTP 200, `enabled: true`, data from those providers |
| anything | import broken | HTTP 200, `enabled: false`, warning with import error |

The endpoint **never** returns 5xx for sentiment-specific failures — all
errors surface inside the payload so the frontend can handle them gracefully.

### Test commands

```bash
python backend/server.py                           # starts on :8001 by default
curl http://localhost:8001/api/sentiment/health    # always 200
curl http://localhost:8001/api/sentiment           # always 200, shape depends on flags
```

## Deployment

This repo deploys as two independent services: the **backend on Railway**
and the **frontend on Vercel**. Both platforms read from the same GitHub
repo — you just point each at a different subdirectory.

### Backend → Railway

Railway config lives in [backend/railway.json](backend/railway.json).
Python version is pinned in [backend/runtime.txt](backend/runtime.txt)
(3.11). Runtime-only Python deps are in [backend/requirements.txt](backend/requirements.txt)
— torch / sklearn / matplotlib are NOT installed on the deployed server
(they're only needed for training, which runs locally).

Steps:

1. On Railway → **New Project** → **Deploy from GitHub repo** → select this repo.
2. In the service Settings → **Root Directory** set to `backend`.
   Railway will pick up `backend/railway.json`, `backend/requirements.txt`,
   and `backend/runtime.txt` automatically.
3. In **Variables**, set (all optional, defaults work):

   ```
   CORS_ALLOW_ORIGIN=https://your-app.vercel.app    # tighten from "*"
   ENABLE_LIVE_SENTIMENT=0                           # leave off unless you have keys
   ALLOW_ASSEMBLY_FALLBACK=0
   # X_BEARER_TOKEN=... YOUTUBE_API_KEY=... NEWS_API_KEY=...   # only if sentiment on
   ```

   `PORT` and `HOST` are provided by Railway automatically.

4. Deploy. The health check hits `/api/health`. First build takes 1-2 min
   (small deps only — no torch).

5. Grab the public URL from Railway (e.g.
   `https://<service-name>.up.railway.app`) and use it in the frontend step.

**What gets served**: [backend/dataset/predictions/predictions_2026.csv](backend/dataset/predictions/predictions_2026.csv)
is committed to the repo and shipped with the deploy. Re-train locally,
commit the new CSV, and Railway redeploys with the fresh predictions.
No training happens on Railway.

### Frontend → Vercel

Vercel config lives in [frontend/vercel.json](frontend/vercel.json).
It currently rewrites `/api/*` to an existing Railway backend URL
(`owlytics-election-prediction-production.up.railway.app`). **If your
Railway URL is different, edit that URL in `frontend/vercel.json` and
commit.**

Steps:

1. On Vercel → **Add New Project** → import this repo.
2. In project settings → **Root Directory** set to `frontend`.
   Framework preset: **Vite**. Build command `npm run build`. Output `dist`.
3. In **Environment Variables**, set at least:

   ```
   VITE_API_BASE_URL=https://<your-railway-service>.up.railway.app
   ```

   Optional integrity checks (see [frontend/.env.example](frontend/.env.example)):

   ```
   VITE_EXPECTED_API_VERSION=tn-2026.1
   VITE_EXPECTED_PREDICTIONS_SHA256=<sha from /api/predictions/meta>
   ```

4. Deploy. First build takes ~30s.

### Picking an API strategy (pick one)

The frontend supports two ways to reach the backend:

| strategy | how it works | set |
|---|---|---|
| **Direct CORS** (recommended) | Browser → Railway directly. Needs `CORS_ALLOW_ORIGIN` on the backend to include the Vercel URL. | `VITE_API_BASE_URL=https://...up.railway.app` |
| **Vercel rewrite proxy** | Browser → Vercel → Railway. Same-origin, no CORS needed. `VITE_API_BASE_URL` must be empty so the client hits relative `/api/...`. | Edit `frontend/vercel.json` rewrite destination to your Railway URL. Leave `VITE_API_BASE_URL` unset. |

Don't mix both — if `VITE_API_BASE_URL` is set, the client uses direct calls
and the Vercel rewrite becomes dead code. Pick one and stick with it.

### Verifying the deploy

```bash
# Backend health:
curl https://<your-service>.up.railway.app/api/health

# Predictions (should return 234 rows):
curl https://<your-service>.up.railway.app/api/predictions | head -c 400

# Meta with validation note:
curl https://<your-service>.up.railway.app/api/predictions/meta

# Sentiment (returns 200 with enabled:false by default):
curl https://<your-service>.up.railway.app/api/sentiment/health
curl https://<your-service>.up.railway.app/api/sentiment
```

## Environment variables (backend)

All variables are optional. Defaults are safe for local development
without any API keys.

```
PORT=8001
HOST=0.0.0.0
CORS_ALLOW_ORIGIN=*                   # tighten to your Vercel URL in prod
ALLOW_ASSEMBLY_FALLBACK=0
ENABLE_LIVE_SENTIMENT=0
X_BEARER_TOKEN=
YOUTUBE_API_KEY=
NEWS_API_KEY=
```

Put them in `backend/.env` (never committed — see `.gitignore`). See
[backend/.env.example](backend/.env.example) for the canonical template.
