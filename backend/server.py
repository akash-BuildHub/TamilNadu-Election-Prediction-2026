import csv
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from config import (
    DATA_FILES_DIR,
    PREDICTIONS_DIR,
    VALIDATION_DIR,
)

# Analysis-filter system (long_term_trend / recent_swing / live_intelligence_score).
# Defensive import so /api/predictions stays up even if the analysis CSVs
# are missing -- the analysis-typed branch surfaces the error explicitly.
_ANALYSIS_IMPORT_OK = False
_ANALYSIS_IMPORT_ERROR = ""
try:
    from analysis import (  # type: ignore
        ANALYSIS_TYPES,
        build_analysis_context,
        run_analysis,
    )
    _ANALYSIS_IMPORT_OK = True
except Exception as _exc:  # pragma: no cover - diagnostic only
    ANALYSIS_TYPES = ("long_term_trend", "recent_swing", "live_intelligence_score")
    _ANALYSIS_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# Optional sentiment feature. Defensive import so the server ALWAYS boots,
# even if requests/pandas/torch or the sentiment modules themselves fail to
# load. _SENTIMENT_IMPORT_ERROR is surfaced via /api/sentiment/health.
_SENTIMENT_IMPORT_OK = False
_SENTIMENT_IMPORT_ERROR = ""
try:
    from data.live_collectors import (  # type: ignore
        create_sentiment_data_live,
        create_social_media_details_live,
        get_api_context,
        load_env_file,
    )
    from data.sentiment_extractor import SentimentExtractor  # type: ignore
    _SENTIMENT_IMPORT_OK = True
except Exception as _exc:  # pragma: no cover - diagnostic only
    _SENTIMENT_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

ROOT = Path(__file__).resolve().parent
PREDICTIONS_FILE = Path(PREDICTIONS_DIR) / "predictions_2026.csv"
# Legacy location (kept for read-only backward compatibility). We never
# write new files here; see _load_rows_from_predictions_file() below.
LEGACY_PREDICTIONS_FILE = ROOT / "predictions_2026.csv"
# If predictions_2026.csv has not been generated yet, the server can fall
# back to the projection-layer CSV (the model's training target). This is
# for bootstrap only; see ALLOW_ASSEMBLY_FALLBACK below.
ASSEMBLY_FALLBACK_FILE = Path(DATA_FILES_DIR) / "tamilnadu_assembly_2026.csv"
# Validation summary produced by write_model_validation.py. Surfaces in the
# /api/health and /api/predictions/meta payloads so consumers know the
# confidence field is relative model confidence, not a calibrated probability.
VALIDATION_SUMMARY_FILE = Path(VALIDATION_DIR) / "model_validation_summary.json"
PARTIES = ("DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS")
NO_STORE_CACHE_HEADER = "no-store, no-cache, must-revalidate, max-age=0"
CORS_ALLOWED_HEADERS = "Content-Type, Cache-Control, Pragma"
# CORS_ALLOW_ORIGIN controls the Access-Control-Allow-Origin response header.
# Default "*" stays permissive for local development. In production, set it
# to the exact Vercel URL, e.g.
#   CORS_ALLOW_ORIGIN=https://your-app.vercel.app
# Comma-separated list is also accepted; the first value is used in the
# header and the others are kept as an echo match for the request Origin.
CORS_ALLOW_ORIGIN = os.getenv("CORS_ALLOW_ORIGIN", "*").strip() or "*"
API_VERSION = "tn-2026.1"


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ALLOW_ASSEMBLY_FALLBACK = _env_flag("ALLOW_ASSEMBLY_FALLBACK", default=False)

# Load backend/.env before reading optional feature flags so local dev
# works without the user exporting env vars manually. Safe no-op if the
# file is absent.
if _SENTIMENT_IMPORT_OK:
    try:
        load_env_file(ROOT / ".env")
    except Exception:
        # load_env_file is already defensive, but guard the call site too.
        pass

# Live sentiment is OFF by default. Training and prediction do NOT depend
# on it. See /api/sentiment and /api/sentiment/health.
ENABLE_LIVE_SENTIMENT = _env_flag("ENABLE_LIVE_SENTIMENT", default=False)


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_rows_from_predictions_file(path: Path = PREDICTIONS_FILE):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                {
                    "ac_no": _to_int(row.get("ac_no", 0)),
                    "constituency": row.get("constituency", ""),
                    "district": row.get("district", ""),
                    "predicted": row.get("predicted", ""),
                    "confidence": _to_float(row.get("confidence", 0)),
                    "DMK_ALLIANCE": _to_float(row.get("DMK_ALLIANCE", 0)),
                    "AIADMK_NDA": _to_float(row.get("AIADMK_NDA", 0)),
                    "TVK": _to_float(row.get("TVK", 0)),
                    "NTK": _to_float(row.get("NTK", 0)),
                    "OTHERS": _to_float(row.get("OTHERS", 0)),
                }
            )
    return rows


def _load_rows_from_assembly_fallback():
    if not ASSEMBLY_FALLBACK_FILE.exists():
        raise FileNotFoundError(
            f"Neither {PREDICTIONS_FILE.name} nor {ASSEMBLY_FALLBACK_FILE} was found. "
            "Run build_data_files.py, create_dataset.py, and train.py before starting "
            "the server."
        )

    rows = []
    with ASSEMBLY_FALLBACK_FILE.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            shares = {
                "DMK_ALLIANCE": _to_float(row.get("proj_2026_dmk_alliance_pct", 0)),
                "AIADMK_NDA":   _to_float(row.get("proj_2026_aiadmk_nda_pct", 0)),
                "TVK":          _to_float(row.get("proj_2026_tvk_pct", 0)),
                "NTK":          _to_float(row.get("proj_2026_ntk_pct", 0)),
                "OTHERS":       _to_float(row.get("proj_2026_others_pct", 0)),
            }
            predicted = row.get("proj_2026_winner", "")
            if predicted not in shares:
                predicted = max(shares, key=shares.get)
            confidence = shares.get(predicted, 0.0)

            rows.append(
                {
                    "ac_no": _to_int(row.get("ac_no", 0)),
                    "constituency": row.get("ac_name", row.get("constituency", "")),
                    "district": row.get("district", ""),
                    "predicted": predicted,
                    "confidence": confidence,
                    "DMK_ALLIANCE": shares["DMK_ALLIANCE"],
                    "AIADMK_NDA": shares["AIADMK_NDA"],
                    "TVK": shares["TVK"],
                    "NTK": shares["NTK"],
                    "OTHERS": shares["OTHERS"],
                }
            )
    return rows


def _iso_mtime_utc(path: Path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except FileNotFoundError:
        return None


def _file_sha256(path: Path):
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError:
        return None


def _seat_counts(rows):
    counts = {party: 0 for party in PARTIES}
    for row in rows:
        predicted = row.get("predicted")
        if predicted in counts:
            counts[predicted] += 1
    return counts


def _load_validation_summary():
    """
    Return the validation summary dict written by write_model_validation.py.
    Degrades to a minimal stub if the file is missing so the server still
    starts (but consumers still see the disclaimer wording).
    """
    try:
        with VALIDATION_SUMMARY_FILE.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError:
        return {
            "validation_note": (
                "Model validation summary not generated. "
                "Run: python backend/write_model_validation.py"
            ),
            "confidence_type": "relative_model_confidence_not_true_probability",
        }


def _api_key_presence() -> dict:
    """Report which sentiment API keys are present, without revealing values."""
    present = {"x_bearer_token": False, "youtube_api_key": False, "news_api_key": False}
    if not _SENTIMENT_IMPORT_OK:
        return present
    try:
        ctx = get_api_context()
        present["x_bearer_token"] = bool(ctx.x_bearer_token)
        present["youtube_api_key"] = bool(ctx.youtube_api_key)
        present["news_api_key"] = bool(ctx.news_api_key)
    except Exception:
        pass
    return present


def _build_sentiment_health() -> dict:
    """Diagnostic payload for /api/sentiment/health. Always returns 200."""
    keys = _api_key_presence()
    any_key = any(keys.values())
    live_enabled = ENABLE_LIVE_SENTIMENT and _SENTIMENT_IMPORT_OK and any_key

    return {
        "module_import_ok": _SENTIMENT_IMPORT_OK,
        "module_import_error": _SENTIMENT_IMPORT_ERROR or None,
        "enable_live_sentiment_flag": ENABLE_LIVE_SENTIMENT,
        "api_keys_present": keys,
        "any_api_key_present": any_key,
        "live_collection_enabled": live_enabled,
        "training_dependency": False,
        "last_checked_utc": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "sentiment": "/api/sentiment",
            "health": "/api/sentiment/health",
        },
    }


def _build_sentiment_payload() -> dict:
    """User-facing /api/sentiment response. Always returns 200."""
    payload = {
        "enabled": False,
        "source": "live_collectors",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": [],
        "warning": "",
        "training_dependency": False,
    }

    if not ENABLE_LIVE_SENTIMENT:
        payload["warning"] = (
            "Live sentiment is disabled. Set ENABLE_LIVE_SENTIMENT=true and "
            "configure API keys (X_BEARER_TOKEN, YOUTUBE_API_KEY, NEWS_API_KEY) "
            "to enable it."
        )
        return payload

    if not _SENTIMENT_IMPORT_OK:
        payload["warning"] = (
            "Sentiment module failed to import: "
            f"{_SENTIMENT_IMPORT_ERROR or 'unknown error'}. "
            "Check that backend/data/ is intact."
        )
        return payload

    keys = _api_key_presence()
    if not any(keys.values()):
        payload["warning"] = (
            "Live collection is enabled but no API keys are configured. "
            "Set X_BEARER_TOKEN, YOUTUBE_API_KEY, or NEWS_API_KEY in backend/.env "
            "and restart the server."
        )
        return payload

    payload["enabled"] = True
    try:
        extractor = SentimentExtractor()
        social_df = create_social_media_details_live(extractor)
        summary_df = create_sentiment_data_live(social_df)
        payload["data"] = summary_df.to_dict(orient="records")
        payload["source_row_count"] = int(len(social_df))
        if social_df.empty:
            payload["warning"] = (
                "Live collection returned no rows. API keys may be invalid, "
                "rate-limited, or the queries returned nothing."
            )
    except Exception as exc:
        # Never crash the endpoint -- report the failure as a warning.
        payload["enabled"] = False
        payload["warning"] = (
            f"Live collection failed: {type(exc).__name__}: {exc}"
        )
        payload["error_trace"] = traceback.format_exc().splitlines()[-1]
    return payload


def _build_predictions_meta(rows, source_file: Path, fallback_in_use: bool):
    counts = _seat_counts(rows)
    projected_winner = "-"
    if rows:
        projected_winner = max(PARTIES, key=lambda party: counts[party])
    validation = _load_validation_summary()
    return {
        "api_version": API_VERSION,
        "state": "Tamil Nadu",
        "election_year": 2026,
        "source_file": source_file.name,
        "source_path": str(source_file),
        "source_last_modified_utc": _iso_mtime_utc(source_file),
        "source_sha256": _file_sha256(source_file),
        "fallback_in_use": fallback_in_use,
        "allow_assembly_fallback": ALLOW_ASSEMBLY_FALLBACK,
        "total_constituencies": len(rows),
        "seat_counts": counts,
        "projected_winner": projected_winner,
        "majority_threshold": 118,
        # Honest-interpretation disclaimer. Any frontend that wants to display
        # "accuracy" or "confidence" semantics should read these two fields.
        "confidence_type": validation.get("confidence_type"),
        "validation_note": validation.get("validation_note"),
        "validation": {
            "train_py_synthetic_cv_accuracy":
                validation.get("train_py_synthetic_cv_accuracy"),
            "party_level_backtest_cv_accuracy":
                (validation.get("party_level_backtest") or {}).get("cv_accuracy"),
            "alliance_level_backtest_cv_accuracy":
                (validation.get("alliance_level_backtest") or {}).get("cv_accuracy"),
        },
    }


class ElectionAPIHandler(BaseHTTPRequestHandler):
    server_version = "TN-ElectionAPI/1.0"

    def _cors_origin(self):
        """
        Resolve the Access-Control-Allow-Origin value for the current request.
        If CORS_ALLOW_ORIGIN is "*", echo "*". Otherwise, if the request's
        Origin matches any entry in the comma-separated allowlist, echo it;
        else return the first allowlist entry as a conservative default.
        """
        if CORS_ALLOW_ORIGIN == "*":
            return "*"
        allowed = [x.strip() for x in CORS_ALLOW_ORIGIN.split(",") if x.strip()]
        if not allowed:
            return "*"
        req_origin = self.headers.get("Origin", "") or ""
        if req_origin and req_origin in allowed:
            return req_origin
        return allowed[0]

    def _send_json(self, payload, status=200, extra_headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", CORS_ALLOWED_HEADERS)
        self.send_header("Cache-Control", NO_STORE_CACHE_HEADER)
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def _load_predictions(self):
        # Primary: new canonical location.
        if PREDICTIONS_FILE.exists():
            return _load_rows_from_predictions_file(PREDICTIONS_FILE), PREDICTIONS_FILE, False

        # Backward compatibility: read (but never write) from the legacy
        # path if someone has stale artefacts from before the reorg.
        if LEGACY_PREDICTIONS_FILE.exists():
            return (
                _load_rows_from_predictions_file(LEGACY_PREDICTIONS_FILE),
                LEGACY_PREDICTIONS_FILE,
                False,
            )

        if ALLOW_ASSEMBLY_FALLBACK:
            return _load_rows_from_assembly_fallback(), ASSEMBLY_FALLBACK_FILE, True

        raise FileNotFoundError(
            f"{PREDICTIONS_FILE.name} not found at {PREDICTIONS_FILE}. "
            f"Run `python backend/train.py`. "
            "To intentionally use heuristic fallback data, set ALLOW_ASSEMBLY_FALLBACK=1."
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", CORS_ALLOWED_HEADERS)
        self.end_headers()

    def _get_query_param(self, key: str) -> str:
        qs = parse_qs(urlparse(self.path).query)
        values = qs.get(key) or []
        return values[0].strip() if values else ""

    def _handle_analysis_predictions(self, analysis_type: str) -> None:
        """
        Serve `/api/predictions?analysis_type=...`. Returns an envelope:
            {
                "analysis_type": ...,
                "meta": {...},
                "rows": [...],
            }
        Default `/api/predictions` (no analysis_type) is unchanged and
        continues to return the bare rows list -- existing frontend code
        keeps working.
        """
        if not _ANALYSIS_IMPORT_OK:
            self._send_json(
                {
                    "error": "Analysis module failed to import",
                    "import_error": _ANALYSIS_IMPORT_ERROR,
                },
                status=500,
            )
            return
        if analysis_type not in ANALYSIS_TYPES:
            self._send_json(
                {
                    "error": f"Unknown analysis_type '{analysis_type}'",
                    "supported_analysis_types": list(ANALYSIS_TYPES),
                },
                status=400,
            )
            return
        try:
            rows, meta = run_analysis(analysis_type)
            self._send_json(
                {
                    "analysis_type": analysis_type,
                    "meta": meta,
                    "rows": rows,
                },
                extra_headers={
                    "X-API-Version": API_VERSION,
                    "X-Analysis-Type": analysis_type,
                },
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json(
                {"error": f"Unexpected server error: {exc}"}, status=500
            )

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            try:
                rows, source_file, fallback_in_use = self._load_predictions()
                self._send_json({
                    "status": "ok",
                    "api_version": API_VERSION,
                    "meta": _build_predictions_meta(rows, source_file, fallback_in_use),
                })
            except FileNotFoundError as exc:
                self._send_json({"status": "error", "error": str(exc)}, status=500)
            except Exception as exc:
                self._send_json({"status": "error", "error": f"Unexpected server error: {exc}"}, status=500)
            return

        if path == "/api/predictions":
            analysis_type = self._get_query_param("analysis_type")
            if analysis_type:
                self._handle_analysis_predictions(analysis_type)
                return
            try:
                rows, source_file, fallback_in_use = self._load_predictions()
                self._send_json(
                    rows,
                    extra_headers={
                        "X-API-Version": API_VERSION,
                        "X-Predictions-Source": source_file.name,
                        "X-Predictions-Last-Modified-Utc": _iso_mtime_utc(source_file),
                        "X-Predictions-SHA256": _file_sha256(source_file),
                        "X-Predictions-Fallback": "1" if fallback_in_use else "0",
                    },
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            except Exception as exc:
                self._send_json({"error": f"Unexpected server error: {exc}"}, status=500)
            return

        if path == "/api/predictions/analysis/meta":
            analysis_type = self._get_query_param("analysis_type")
            if not analysis_type:
                self._send_json(
                    {
                        "supported_analysis_types": list(ANALYSIS_TYPES),
                        "weights": {
                            "long_term_trend": 0.40,
                            "recent_swing": 0.35,
                            "live_intelligence_score": 0.25,
                        },
                        "election_year_mapping": {
                            "2016_assembly": "2014_lok_sabha",
                            "2021_assembly": "2019_lok_sabha",
                            "2026_assembly": "2024_lok_sabha",
                        },
                        "prediction_mode": True,
                    }
                )
                return
            if not _ANALYSIS_IMPORT_OK:
                self._send_json(
                    {
                        "error": "Analysis module failed to import",
                        "import_error": _ANALYSIS_IMPORT_ERROR,
                    },
                    status=500,
                )
                return
            try:
                self._send_json(build_analysis_context(analysis_type))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"error": f"Unexpected server error: {exc}"}, status=500)
            return

        if path == "/api/predictions/meta":
            try:
                rows, source_file, fallback_in_use = self._load_predictions()
                self._send_json(_build_predictions_meta(rows, source_file, fallback_in_use))
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            except Exception as exc:
                self._send_json({"error": f"Unexpected server error: {exc}"}, status=500)
            return

        if path == "/api/sentiment/health":
            # Always 200 -- this endpoint IS the diagnostic.
            self._send_json(_build_sentiment_health())
            return

        if path == "/api/sentiment":
            # Always 200 unless something genuinely unexpected breaks the
            # response builder itself. Live collection failures are
            # surfaced inside the payload, not as HTTP errors.
            try:
                self._send_json(_build_sentiment_payload())
            except Exception as exc:
                self._send_json(
                    {"error": f"Unexpected server error: {exc}"}, status=500
                )
            return

        self._send_json(
            {
                "error": "Not found",
                "available_routes": [
                    "/api/health",
                    "/api/predictions",
                    "/api/predictions?analysis_type=long_term_trend",
                    "/api/predictions?analysis_type=recent_swing",
                    "/api/predictions?analysis_type=live_intelligence_score",
                    "/api/predictions/meta",
                    "/api/predictions/analysis/meta",
                    "/api/sentiment",
                    "/api/sentiment/health",
                ],
            },
            status=404,
        )


def main(host=None, port=None):
    bind_host = host if host is not None else os.getenv("HOST", "0.0.0.0")
    bind_port = int(port) if port is not None else int(os.getenv("PORT", "8001"))
    server = ThreadingHTTPServer((bind_host, bind_port), ElectionAPIHandler)
    print(f"Tamil Nadu backend API running on http://{bind_host}:{bind_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
