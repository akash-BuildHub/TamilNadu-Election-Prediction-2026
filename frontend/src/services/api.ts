import {
  AnalysisResponse,
  AnalysisType,
  HealthResponse,
  PredictionRow,
  PredictionsMeta,
} from "../types/prediction";

/**
 * API Base URL Configuration
 * Priority:
 * 1. VITE_API_BASE_URL (preferred)
 * 2. VITE_API_URL (fallback)
 * 3. Inferred from current browser host (LAN and production safe default)
 * 4. Local dev fallback: http://127.0.0.1:8001
 * Trailing slashes are removed automatically to avoid `//api/...` requests.
 *
 * For production (Railway): Set VITE_API_BASE_URL to your Railway backend URL
 * For local dev: Use .env with VITE_API_BASE_URL=http://127.0.0.1:8001
 */
function normalizeApiBase(rawValue: string | undefined): string {
  if (!rawValue) return "";
  return rawValue.trim().replace(/\/+$/, "");
}

const LOCAL_API_BASE = "http://127.0.0.1:8001";

function isPrivateIpv4(hostname: string): boolean {
  const octets = hostname.split(".").map((part) => Number(part));
  if (octets.length !== 4 || octets.some((octet) => Number.isNaN(octet))) {
    return false;
  }

  const [first, second] = octets;
  if (first === 10) return true;
  if (first === 192 && second === 168) return true;
  if (first === 172 && second >= 16 && second <= 31) return true;
  return false;
}

function inferApiBaseFromWindow(): string {
  if (typeof window === "undefined") return "";

  const { hostname, protocol, origin } = window.location;
  if (hostname === "127.0.0.1" || hostname === "localhost") {
    return LOCAL_API_BASE;
  }

  if (isPrivateIpv4(hostname)) {
    const inferredPort = String(import.meta.env.VITE_API_PORT || "8001").trim();
    return `${protocol}//${hostname}:${inferredPort}`;
  }

  return import.meta.env.PROD ? origin : "";
}

function isVercelHostname(hostname: string): boolean {
  return hostname === "vercel.app" || hostname.endsWith(".vercel.app");
}

const EXPLICIT_API_BASE =
  normalizeApiBase(import.meta.env.VITE_API_BASE_URL) ||
  normalizeApiBase(import.meta.env.VITE_API_URL);

const INFERRED_API_BASE = normalizeApiBase(inferApiBaseFromWindow());

const WINDOW_HOSTNAME = typeof window !== "undefined" ? window.location.hostname : "";
const SHOULD_USE_SAME_ORIGIN_PROXY =
  import.meta.env.PROD &&
  isVercelHostname(WINDOW_HOSTNAME) &&
  String(import.meta.env.VITE_FORCE_DIRECT_API || "").trim() !== "1";

const API_BASE =
  (SHOULD_USE_SAME_ORIGIN_PROXY
    ? INFERRED_API_BASE || EXPLICIT_API_BASE
    : EXPLICIT_API_BASE || INFERRED_API_BASE) ||
  (import.meta.env.DEV ? LOCAL_API_BASE : "");

const API_BASE_FALLBACK =
  SHOULD_USE_SAME_ORIGIN_PROXY && EXPLICIT_API_BASE && EXPLICIT_API_BASE !== API_BASE
    ? EXPLICIT_API_BASE
    : "";

const EXPECTED_PREDICTIONS_SHA256 =
  import.meta.env.VITE_EXPECTED_PREDICTIONS_SHA256?.trim().toLowerCase() || "";

const EXPECTED_API_VERSION =
  import.meta.env.VITE_EXPECTED_API_VERSION?.trim() || "";

function withCacheBuster(path: string): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}_ts=${Date.now()}`;
}

// Debug logging (helpful for troubleshooting)
if (import.meta.env.DEV) {
  console.log(
    "[API Config] API_BASE_URL:",
    API_BASE,
    "| SHOULD_USE_SAME_ORIGIN_PROXY:",
    SHOULD_USE_SAME_ORIGIN_PROXY,
    "| API_BASE_FALLBACK:",
    API_BASE_FALLBACK || "(none)",
    "| EXPLICIT_API_BASE:",
    EXPLICIT_API_BASE || "(none)",
    "| INFERRED_API_BASE:",
    INFERRED_API_BASE || "(none)",
    "| VITE_API_BASE_URL:",
    import.meta.env.VITE_API_BASE_URL,
    "| VITE_API_URL:",
    import.meta.env.VITE_API_URL
  );
}

/**
 * Validates that the API_BASE URL is set
 * Logs a warning if using default localhost
 */
function validateApiConfig(): void {
  if (API_BASE === LOCAL_API_BASE && import.meta.env.PROD) {
    console.warn(
      "[API Config] WARNING: Using localhost API_BASE in production. Set VITE_API_BASE_URL to your Railway backend URL."
    );
  }

  if (!EXPLICIT_API_BASE && import.meta.env.PROD) {
    console.warn(
      "[API Config] WARNING: VITE_API_BASE_URL is not set in production. Using inferred API base:",
      API_BASE || "(same-origin /api)"
    );
  }

  if (SHOULD_USE_SAME_ORIGIN_PROXY) {
    console.info(
      "[API Config] INFO: Using same-origin API proxy on Vercel host to improve cross-network reachability."
    );
  }

  if (import.meta.env.PROD && !EXPECTED_PREDICTIONS_SHA256) {
    console.warn(
      "[API Config] WARNING: VITE_EXPECTED_PREDICTIONS_SHA256 is not set. Deployment may show stale backend data without detection."
    );
  }
}

// Validate on module load
validateApiConfig();

function withApiBase(path: string, base: string): string {
  return withCacheBuster(`${base}${path}`);
}

async function fetchWithApiFallback(
  path: string,
  init: RequestInit
): Promise<Response> {
  const primaryUrl = withApiBase(path, API_BASE);
  const fallbackUrl = API_BASE_FALLBACK ? withApiBase(path, API_BASE_FALLBACK) : "";

  try {
    const response = await fetch(primaryUrl, init);
    if (
      fallbackUrl &&
      (response.status === 404 ||
        response.status === 502 ||
        response.status === 503 ||
        response.status === 504)
    ) {
      console.warn(
        `[API] Primary endpoint returned ${response.status} (${API_BASE}). Retrying via fallback ${API_BASE_FALLBACK}.`
      );
      return fetch(fallbackUrl, init);
    }
    return response;
  } catch (error) {
    if (!(error instanceof TypeError) || !fallbackUrl) {
      throw error;
    }

    console.warn(
      `[API] Primary endpoint unreachable (${API_BASE}). Retrying via fallback ${API_BASE_FALLBACK}.`
    );
    return fetch(fallbackUrl, init);
  }
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  const body = await fetchHealth(signal);
  return body.status === "ok";
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    const response = await fetchWithApiFallback("/api/health", {
      signal,
      cache: "no-store",
    });
    if (!response.ok) {
      console.warn(
        `[API] Health check failed: ${response.status} ${response.statusText}`
      );
      return {
        status: "error",
        error: `Health check failed: ${response.status} ${response.statusText}`,
      };
    }
    const body = (await response.json()) as HealthResponse;
    if (!body?.status) {
      return { status: "error", error: "Invalid health response payload." };
    }
    return body;
  } catch (error) {
    console.error("[API] Health check error:", error);
    return {
      status: "error",
      error: error instanceof Error ? error.message : "Health check request failed.",
    };
  }
}

export async function fetchPredictionsMeta(signal?: AbortSignal): Promise<PredictionsMeta> {
  const response = await fetchWithApiFallback("/api/predictions/meta", {
    signal,
    cache: "no-store",
  });

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(
        `Backend API is outdated: ${API_BASE}/api/predictions/meta is missing (404). Redeploy the Railway backend from latest main so it serves /api/predictions/meta.`
      );
    }
    throw new Error(
      `Failed to load prediction metadata (${response.status} ${response.statusText}) from ${API_BASE}`
    );
  }

  return (await response.json()) as PredictionsMeta;
}

export async function fetchPredictions(signal?: AbortSignal): Promise<PredictionRow[]> {
  try {
    const response = await fetchWithApiFallback("/api/predictions", {
      signal,
      cache: "no-store",
    });
    if (!response.ok) {
      const errorMsg = `Failed to load predictions (${response.status} ${response.statusText}) from ${API_BASE}`;
      console.error("[API]", errorMsg);
      throw new Error(errorMsg);
    }
    const source = response.headers.get("X-Predictions-Source");
    const fallback = response.headers.get("X-Predictions-Fallback") === "1";
    if (source) {
      console.info(
        `[API] Predictions source: ${source}${fallback ? " (fallback mode)" : ""}`
      );
    }
    const data: PredictionRow[] = await response.json();
    return data;
  } catch (error) {
    if (error instanceof TypeError) {
      console.error(
        "[API] Network error - unable to reach backend at",
        API_BASE,
        error
      );
    }
    throw error;
  }
}

export async function fetchAnalysisPredictions(
  analysisType: AnalysisType,
  signal?: AbortSignal
): Promise<AnalysisResponse> {
  const response = await fetchWithApiFallback(
    `/api/predictions?analysis_type=${encodeURIComponent(analysisType)}`,
    { signal, cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error(
      `Failed to load ${analysisType} analysis (${response.status} ${response.statusText}) from ${API_BASE}`
    );
  }
  const body = (await response.json()) as AnalysisResponse;
  if (!body || !Array.isArray(body.rows) || !body.meta) {
    throw new Error(
      `Malformed analysis response for ${analysisType}: missing rows/meta.`
    );
  }
  return body;
}

export { API_BASE };
export { EXPECTED_API_VERSION, EXPECTED_PREDICTIONS_SHA256 };
