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
