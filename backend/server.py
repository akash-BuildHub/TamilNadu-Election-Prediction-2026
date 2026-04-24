import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PREDICTIONS_FILE = ROOT / "predictions_2026.csv"
# If predictions_2026.csv has not been generated yet, the server can fall
# back to the projection-layer CSV (the model's training target). This is
# for bootstrap only; see ALLOW_ASSEMBLY_FALLBACK below.
ASSEMBLY_FALLBACK_FILE = ROOT / "data_files" / "tamilnadu_assembly_2026.csv"
# Validation summary produced by write_model_validation.py. Surfaces in the
# /api/health and /api/predictions/meta payloads so consumers know the
# confidence field is relative model confidence, not a calibrated probability.
VALIDATION_SUMMARY_FILE = ROOT / "backtests" / "model_validation_summary.json"
PARTIES = ("DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS")
NO_STORE_CACHE_HEADER = "no-store, no-cache, must-revalidate, max-age=0"
CORS_ALLOWED_HEADERS = "Content-Type, Cache-Control, Pragma"
API_VERSION = "tn-2026.1"


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ALLOW_ASSEMBLY_FALLBACK = _env_flag("ALLOW_ASSEMBLY_FALLBACK", default=False)


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


def _load_rows_from_predictions_file():
    rows = []
    with PREDICTIONS_FILE.open("r", encoding="utf-8", newline="") as fp:
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

    def _send_json(self, payload, status=200, extra_headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
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
        if PREDICTIONS_FILE.exists():
            return _load_rows_from_predictions_file(), PREDICTIONS_FILE, False

        if ALLOW_ASSEMBLY_FALLBACK:
            return _load_rows_from_assembly_fallback(), ASSEMBLY_FALLBACK_FILE, True

        raise FileNotFoundError(
            f"{PREDICTIONS_FILE.name} not found. Run `python backend/train.py`. "
            "To intentionally use heuristic fallback data, set ALLOW_ASSEMBLY_FALLBACK=1."
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", CORS_ALLOWED_HEADERS)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

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

        if path == "/api/predictions/meta":
            try:
                rows, source_file, fallback_in_use = self._load_predictions()
                self._send_json(_build_predictions_meta(rows, source_file, fallback_in_use))
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            except Exception as exc:
                self._send_json({"error": f"Unexpected server error: {exc}"}, status=500)
            return

        self._send_json(
            {
                "error": "Not found",
                "available_routes": ["/api/health", "/api/predictions", "/api/predictions/meta"],
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
