export type Party = "DMK_ALLIANCE" | "AIADMK_NDA" | "TVK" | "NTK" | "OTHERS";

export const PARTIES: Party[] = [
  "DMK_ALLIANCE",
  "AIADMK_NDA",
  "TVK",
  "NTK",
  "OTHERS",
];

// Human-readable labels for UI chrome (badges, filter dropdowns, legends).
export const PARTY_LABELS: Record<Party, string> = {
  DMK_ALLIANCE: "DMK",
  AIADMK_NDA: "AIADMK",
  TVK: "TVK",
  NTK: "NTK",
  OTHERS: "Others",
};

export type PredictionRow = {
  ac_no: number;
  constituency: string;
  district: string;
  predicted: Party;
  confidence: number;
  DMK_ALLIANCE: number;
  AIADMK_NDA: number;
  TVK: number;
  NTK: number;
  OTHERS: number;
};

export type SeatCounts = Record<Party, number>;

export type PredictionsMeta = {
  api_version?: string;
  state?: string;
  election_year?: number;
  source_file: string;
  source_path?: string;
  source_last_modified_utc?: string | null;
  source_sha256?: string | null;
  fallback_in_use: boolean;
  allow_assembly_fallback?: boolean;
  total_constituencies: number;
  seat_counts: SeatCounts;
  projected_winner: Party | "-";
  majority_threshold?: number;
};

export type HealthResponse = {
  status: "ok" | "error";
  api_version?: string;
  meta?: PredictionsMeta;
  error?: string;
};

// ---- Analysis filter system ------------------------------------------------
export type AnalysisType =
  | "long_term_trend"
  | "recent_swing"
  | "live_intelligence_score";

export const ANALYSIS_TYPES: AnalysisType[] = [
  "long_term_trend",
  "recent_swing",
  "live_intelligence_score",
];

export const ANALYSIS_LABELS: Record<AnalysisType, string> = {
  long_term_trend: "Long-Term Trend",
  recent_swing: "Recent Swing",
  live_intelligence_score: "Live Intelligence Score",
};

// Per-row payload returned when an analysis filter is selected. All
// analysis-specific fields are optional so the same type covers all
// three analysis_type values plus the bare default response.
export type AnalysisPredictionRow = PredictionRow & {
  analysis_predicted?: Party;
  final_prediction_score?: number;
  win_probability?: number;
  confidence_level?: string;

  // long_term_trend
  long_term_trend_score?: number;
  historical_strength?: number;
  winner_2016?: string;
  winner_2021?: string;
  party_growth_score?: number;
  vote_share_trend_for_predicted?: number[];

  // recent_swing
  recent_swing_score?: number;
  runner_up_2021?: string;
  winner_party_2021?: string;
  runner_up_party_2021?: string;
  winning_margin_2021?: number;
  vote_share_2021?: number;
  incumbency_status?: string;
  constituency_swing?: number;
  seat_retention_probability?: number;
  opposition_gain_probability?: number;

  // live_intelligence_score
  live_intelligence_score?: number;
  sentiment_adjusted_prediction?: number;
  tvk_impact_score?: number;
};

export type AnalysisMeta = {
  analysis_type: AnalysisType;
  cm_election_year: number;
  lok_sabha_reference_years: number[];
  gap_years: number;
  gap_category: string;
  prediction_mode: boolean;
  weights: Record<AnalysisType, number>;
  party_seat_counts: SeatCounts;
  analysis_seat_counts: SeatCounts;
  party_average_score: SeatCounts;
  party_state_share_2026: SeatCounts;
  confidence_buckets: Record<string, number>;
  analysis_specific: Record<string, unknown>;
};

export type AnalysisResponse = {
  analysis_type: AnalysisType;
  meta: AnalysisMeta;
  rows: AnalysisPredictionRow[];
};
