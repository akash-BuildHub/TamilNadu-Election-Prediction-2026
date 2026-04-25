import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { fetchAnalysisPredictions } from "../services/api";
import {
  AnalysisMeta,
  AnalysisPredictionRow,
  AnalysisResponse,
  AnalysisType,
  ANALYSIS_LABELS,
  PARTIES,
  PARTY_LABELS,
  Party,
} from "../types/prediction";
import { asPercentSmart, asSeatPercent } from "../utils/format";
import { PartyBadge } from "./PartyBadge";

const DISPLAY_PARTIES: Party[] = [
  "DMK_ALLIANCE",
  "AIADMK_NDA",
  "TVK",
  "NTK",
  "OTHERS",
];

type AnalysisPanelProps = {
  analysisType: AnalysisType;
};

const SCORE_KEY: Record<AnalysisType, keyof AnalysisPredictionRow> = {
  long_term_trend: "long_term_trend_score",
  recent_swing: "recent_swing_score",
  live_intelligence_score: "live_intelligence_score",
};

function fmtScore(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(3);
}

function fmtPct(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return asPercentSmart(value);
}

type HeaderSummary = {
  totalConstituencies: number;
  dataReference: string;
  projectedWinner: string;
  averageWinningScore: string;
};

const EMPTY_HEADER_SUMMARY: HeaderSummary = {
  totalConstituencies: 0,
  dataReference: "-",
  projectedWinner: "-",
  averageWinningScore: "-",
};

function buildHeaderSummary(meta: AnalysisMeta, rows: AnalysisPredictionRow[]): HeaderSummary {
  const totalConstituencies = rows.length;

  let dataReference: string;
  if (meta.analysis_type === "live_intelligence_score") {
    dataReference = "Live data";
  } else {
    const refYears = meta.lok_sabha_reference_years;
    const earliest = refYears.length > 0 ? Math.min(...refYears) : meta.cm_election_year;
    dataReference = `${earliest} - ${meta.cm_election_year}`;
  }

  const seatCounts: Record<Party, number> = {
    DMK_ALLIANCE: 0,
    AIADMK_NDA: 0,
    TVK: 0,
    NTK: 0,
    OTHERS: 0,
  };
  let winningScoreSum = 0;

  for (const row of rows) {
    const winner = row.analysis_predicted ?? row.predicted;
    if (winner in seatCounts) {
      seatCounts[winner as Party] += 1;
    }
    winningScoreSum += row.win_probability ?? row.confidence ?? 0;
  }

  const maxSeats = Math.max(...PARTIES.map((p) => seatCounts[p]));
  const topParties = PARTIES.filter((p) => seatCounts[p] === maxSeats);
  const projectedWinner = maxSeats > 0
    ? topParties.map((p) => PARTY_LABELS[p]).join(" / ")
    : "-";

  const avgWinningScore = totalConstituencies > 0
    ? winningScoreSum / totalConstituencies
    : 0;

  return {
    totalConstituencies,
    dataReference,
    projectedWinner,
    averageWinningScore: asPercentSmart(avgWinningScore),
  };
}

type FilterState = {
  district: string;
  party: Party | "ALL";
  query: string;
};

type CenterCardProps = {
  rows: AnalysisPredictionRow[];
  districts: string[];
  filter: FilterState;
  onFilterChange: (next: FilterState) => void;
  constituencyOptions: string[];
};

function CenterCard({
  rows,
  districts,
  filter,
  onFilterChange,
  constituencyOptions,
}: CenterCardProps) {
  const [districtOpen, setDistrictOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const districtRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!searchOpen && !districtOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (searchOpen && searchRef.current && !searchRef.current.contains(target)) {
        setSearchOpen(false);
      }
      if (districtOpen && districtRef.current && !districtRef.current.contains(target)) {
        setDistrictOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [searchOpen, districtOpen]);

  const counts = useMemo(() => {
    const c: Record<Party, number> = {
      DMK_ALLIANCE: 0,
      AIADMK_NDA: 0,
      TVK: 0,
      NTK: 0,
      OTHERS: 0,
    };
    for (const row of rows) {
      // Use the analysis-specific winner so each tab produces its own
      // distribution. Fallback to the base predicted winner if the
      // backend hasn't added the analysis_predicted field.
      const winner = row.analysis_predicted ?? row.predicted;
      if (winner in c) {
        c[winner as Party] += 1;
      }
    }
    return c;
  }, [rows]);
  const total = rows.length;
  const safeTotal = total || 1;
  const sortedParties = useMemo(
    () => [...DISPLAY_PARTIES].sort((a, b) => counts[b] - counts[a]),
    [counts],
  );

  return (
    <article className="panel center-card">
      <div className="center-inner-grid">
        <section className="inner-block filters-block">
          <h2>Filters</h2>
          <div className="filters-grid">
            <div className="combo-wrap" ref={districtRef}>
              <label htmlFor="analysis-district">District</label>
              <button
                id="analysis-district"
                type="button"
                className="combo-toggle"
                aria-haspopup="listbox"
                aria-expanded={districtOpen}
                onClick={() => setDistrictOpen((o) => !o)}
              >
                <span>{filter.district === "ALL" ? "All Districts" : filter.district}</span>
                <span className="combo-chevron" aria-hidden="true">▾</span>
              </button>
              {districtOpen && (
                <ul className="combo-list" role="listbox">
                  <li
                    role="option"
                    aria-selected={filter.district === "ALL"}
                    className="combo-item"
                    onClick={() => {
                      onFilterChange({ ...filter, district: "ALL" });
                      setDistrictOpen(false);
                    }}
                  >
                    All Districts
                  </li>
                  {districts.map((d) => (
                    <li
                      key={d}
                      role="option"
                      aria-selected={filter.district === d}
                      className="combo-item"
                      onClick={() => {
                        onFilterChange({ ...filter, district: d });
                        setDistrictOpen(false);
                      }}
                    >
                      {d}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div>
              <label htmlFor="analysis-party">Predicted Party</label>
              <select
                id="analysis-party"
                value={filter.party}
                onChange={(e) =>
                  onFilterChange({ ...filter, party: e.target.value as Party | "ALL" })
                }
              >
                <option value="ALL">All Parties</option>
                {DISPLAY_PARTIES.map((p) => (
                  <option key={p} value={p}>
                    {PARTY_LABELS[p]}
                  </option>
                ))}
              </select>
            </div>

            <div className="search-wrap" ref={searchRef}>
              <label htmlFor="analysis-search">Search Constituency</label>
              <input
                id="analysis-search"
                type="text"
                value={filter.query}
                placeholder="Type or select..."
                autoComplete="off"
                role="combobox"
                aria-expanded={searchOpen}
                aria-controls="analysis-constituency-list"
                onChange={(e) => {
                  onFilterChange({ ...filter, query: e.target.value });
                  setSearchOpen(true);
                }}
                onFocus={() => setSearchOpen(true)}
              />
              {searchOpen && constituencyOptions.length > 0 && (
                <ul
                  id="analysis-constituency-list"
                  className="combo-list"
                  role="listbox"
                >
                  {constituencyOptions.map((name) => (
                    <li
                      key={name}
                      role="option"
                      aria-selected={filter.query === name}
                      className="combo-item"
                      onClick={() => {
                        onFilterChange({ ...filter, query: name });
                        setSearchOpen(false);
                      }}
                    >
                      {name}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </section>

        <section className="inner-block seat-distribution-block">
          <h2>Seat Distribution</h2>
          <div className="bar-list">
            {sortedParties.map((p) => (
              <div className="bar-item" key={p}>
                <div className="bar-top">
                  <span>{PARTY_LABELS[p]}</span>
                  <span>
                    {counts[p]} ({asSeatPercent(counts[p], safeTotal)})
                  </span>
                </div>
                <div className="bar-track">
                  <div
                    className={`bar-fill fill-${p}`}
                    style={{ width: `${(counts[p] / safeTotal) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </article>
  );
}

function DistrictBreakdownCard({ rows }: { rows: AnalysisPredictionRow[] }) {
  const districtBreakdown = useMemo(() => {
    const map = new Map<string, Record<Party, number>>();
    for (const row of rows) {
      const winner = (row.analysis_predicted ?? row.predicted) as Party;
      if (!map.has(row.district)) {
        map.set(row.district, {
          DMK_ALLIANCE: 0,
          AIADMK_NDA: 0,
          TVK: 0,
          NTK: 0,
          OTHERS: 0,
        });
      }
      const counts = map.get(row.district)!;
      if (winner in counts) counts[winner] += 1;
    }
    return [...map.entries()]
      .map(([name, counts]) => ({
        name,
        ...counts,
        total:
          counts.DMK_ALLIANCE +
          counts.AIADMK_NDA +
          counts.TVK +
          counts.NTK +
          counts.OTHERS,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [rows]);

  return (
    <article className="panel">
      <h2>District Breakdown</h2>
      <div className="district-list">
        {districtBreakdown.map((d) => (
          <div key={d.name} className="district-item">
            <div className="district-head">
              <strong>{d.name}</strong>
              <span>{d.total} seats</span>
            </div>
            <div className="district-bars">
              {DISPLAY_PARTIES.map((p) => (
                <div
                  key={p}
                  className={`district-segment segment-${p}`}
                  style={{ width: `${(d[p] / (d.total || 1)) * 100}%` }}
                  title={`${PARTY_LABELS[p]}: ${d[p]}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function CompetitiveSeatsCard({
  rows,
  scoreKey,
}: {
  rows: AnalysisPredictionRow[];
  scoreKey: keyof AnalysisPredictionRow;
}) {
  const closestSeats = useMemo(() => {
    return [...rows]
      .filter((r) => typeof r[scoreKey] === "number")
      .sort((a, b) => (a[scoreKey] as number) - (b[scoreKey] as number))
      .slice(0, 8);
  }, [rows, scoreKey]);

  return (
    <article className="panel">
      <h2>Most Competitive Seats</h2>
      <ul className="tight-list">
        {closestSeats.map((seat) => {
          const score = seat[scoreKey] as number;
          const winner = (seat.analysis_predicted ?? seat.predicted) as Party;
          return (
            <li key={`${seat.ac_no}-${seat.constituency}`}>
              <div>
                <strong>{seat.constituency}</strong>
                <small>{seat.district}</small>
              </div>
              <div className="right-inline">
                <PartyBadge party={winner} />
                <span>{asPercentSmart(score)} Score</span>
              </div>
            </li>
          );
        })}
      </ul>
    </article>
  );
}

function MetaHeader({ summary }: { summary: HeaderSummary }) {
  // Reuse the same .kpi-grid / .kpi-card classes the Default Prediction
  // header uses (AnimatedKpiGrid), so the analysis-tab header is visually
  // identical -- just with four tiles instead of three. No section title
  // is rendered inside the card; the active tab pill above already names it.
  return (
    <section className="kpi-grid analysis-kpi-grid">
      <article className="panel kpi-card">
        <h3>Total Constituencies</h3>
        <strong>{summary.totalConstituencies}</strong>
      </article>
      <article className="panel kpi-card">
        <h3>Data Reference</h3>
        <strong>{summary.dataReference}</strong>
      </article>
      <article className="panel kpi-card">
        <h3>Projected Winner</h3>
        <strong>{summary.projectedWinner}</strong>
      </article>
      <article className="panel kpi-card">
        <h3>Average Winning Score</h3>
        <strong>{summary.averageWinningScore}</strong>
      </article>
    </section>
  );
}

function VoteShareTrendCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as {
    vote_share_trend?: Partial<Record<Party, number[]>>;
    long_term_party_trend_score?: Partial<Record<Party, number>>;
    party_growth_score?: Partial<Record<Party, number>>;
  };
  const vsTrend: Partial<Record<Party, number[]>> = specific.vote_share_trend ?? {};
  const longTermScore: Partial<Record<Party, number>> =
    specific.long_term_party_trend_score ?? {};
  const growth: Partial<Record<Party, number>> = specific.party_growth_score ?? {};

  return (
    <article className="panel analysis-card">
      <h3>Vote-share Trend (2016 - 2021 - 2026)</h3>
      <table className="compact-table">
        <thead>
          <tr>
            <th>Party</th>
            <th>2016</th>
            <th>2021</th>
            <th>2026</th>
            <th>Trend</th>
            <th>Growth</th>
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((p) => {
            const trend = vsTrend[p] || [0, 0, 0];
            return (
              <tr key={p}>
                <td>{PARTY_LABELS[p]}</td>
                <td>{trend[0]?.toFixed(2)}%</td>
                <td>{trend[1]?.toFixed(2)}%</td>
                <td>{trend[2]?.toFixed(2)}%</td>
                <td>{fmtScore(longTermScore[p])}</td>
                <td>
                  {typeof growth[p] === "number"
                    ? `${(growth[p]! * 100).toFixed(1)}%`
                    : "-"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </article>
  );
}

function SeatSwingCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as {
    seat_swing_trend?: Partial<Record<Party, Record<string, number>>>;
  };
  const seatSwing: Partial<Record<Party, Record<string, number>>> =
    specific.seat_swing_trend ?? {};

  return (
    <article className="panel analysis-card">
      <h3>Seat Swing (2016 - 2021 - 2026)</h3>
      <table className="compact-table">
        <thead>
          <tr>
            <th>Party</th>
            <th>2016</th>
            <th>2021</th>
            <th>2026 (Pred)</th>
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((p) => {
            const sw = seatSwing[p] || {};
            return (
              <tr key={p}>
                <td>{PARTY_LABELS[p]}</td>
                <td>{sw["2016"] ?? "-"}</td>
                <td>{sw["2021"] ?? "-"}</td>
                <td>{sw["2026_predicted"] ?? "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </article>
  );
}


type RecentSwingSpecific = {
  recent_swing_party_score?: Partial<Record<Party, number>>;
  party_recent_momentum?: Partial<Record<Party, number>>;
  assembly_state_share?: {
    "2021"?: Partial<Record<Party, number>>;
    "2026_predicted"?: Partial<Record<Party, number>>;
  };
  lok_sabha_state_share_2024?: Partial<Record<Party, number>>;
};

function AssemblyShareCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as RecentSwingSpecific;
  const share2021: Partial<Record<Party, number>> =
    specific.assembly_state_share?.["2021"] ?? {};
  const share2026: Partial<Record<Party, number>> =
    specific.assembly_state_share?.["2026_predicted"] ?? {};

  return (
    <article className="panel analysis-card">
      <h3>2021 Assembly vs 2026 Prediction</h3>
      <table className="compact-table">
        <thead>
          <tr>
            <th>Party</th>
            <th>2021 Share</th>
            <th>2026 Share</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((p) => {
            const a21 = share2021[p] ?? 0;
            const a26 = share2026[p] ?? 0;
            const delta = a26 - a21;
            return (
              <tr key={p}>
                <td>{PARTY_LABELS[p]}</td>
                <td>{a21.toFixed(2)}%</td>
                <td>{a26.toFixed(2)}%</td>
                <td>
                  <strong>
                    {delta >= 0 ? "+" : ""}
                    {delta.toFixed(2)}%
                  </strong>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </article>
  );
}

function SwingMomentumCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as RecentSwingSpecific;
  const swingScore: Partial<Record<Party, number>> = specific.recent_swing_party_score ?? {};
  const momentum: Partial<Record<Party, number>> = specific.party_recent_momentum ?? {};
  const ls2024: Partial<Record<Party, number>> = specific.lok_sabha_state_share_2024 ?? {};

  return (
    <article className="panel analysis-card">
      <h3>2024 Lok Sabha Influence &amp; Swing</h3>
      <table className="compact-table">
        <thead>
          <tr>
            <th>Party</th>
            <th>2024 LS</th>
            <th>Momentum</th>
            <th>Swing Score</th>
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((p) => (
            <tr key={p}>
              <td>{PARTY_LABELS[p]}</td>
              <td>{(ls2024[p] ?? 0).toFixed(2)}%</td>
              <td>
                {typeof momentum[p] === "number"
                  ? `${(momentum[p]! * 100).toFixed(2)}%`
                  : "-"}
              </td>
              <td>
                <strong>{fmtScore(swingScore[p])}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

type LiveIntelSpecific = {
  party_sentiment_score?: Partial<Record<Party, number>>;
  leader_sentiment_score?: Partial<Record<Party, number>>;
  leader_names?: Partial<Record<Party, string>>;
  candidate_sentiment_score?: Partial<Record<Party, number>>;
  social_media_sentiment_score?: Partial<Record<Party, number>>;
  news_sentiment_score?: Partial<Record<Party, number>>;
  local_issue_party_score?: Partial<Record<Party, number>>;
  party_live_intelligence_score?: Partial<Record<Party, number>>;
  tvk_impact_score?: number;
  tvk_impact_metrics?: Record<string, number>;
};

function SentimentScoresCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as LiveIntelSpecific;
  const partySent: Partial<Record<Party, number>> = specific.party_sentiment_score ?? {};
  const leaderSent: Partial<Record<Party, number>> = specific.leader_sentiment_score ?? {};
  const candidateSent: Partial<Record<Party, number>> = specific.candidate_sentiment_score ?? {};
  const socialSent: Partial<Record<Party, number>> = specific.social_media_sentiment_score ?? {};
  const newsSent: Partial<Record<Party, number>> = specific.news_sentiment_score ?? {};
  const localIssue: Partial<Record<Party, number>> = specific.local_issue_party_score ?? {};
  const liveScore: Partial<Record<Party, number>> = specific.party_live_intelligence_score ?? {};

  return (
    <article className="panel analysis-card">
      <h3>Sentiment Scores by Party</h3>
      <table className="compact-table">
        <thead>
          <tr>
            <th>Party</th>
            <th>Party</th>
            <th>Leader</th>
            <th>Cand.</th>
            <th>Social</th>
            <th>News</th>
            <th>Issues</th>
            <th>Live</th>
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((p) => (
            <tr key={p}>
              <td>{PARTY_LABELS[p]}</td>
              <td>{fmtScore(partySent[p])}</td>
              <td>{fmtScore(leaderSent[p])}</td>
              <td>{fmtScore(candidateSent[p])}</td>
              <td>{fmtScore(socialSent[p])}</td>
              <td>{fmtScore(newsSent[p])}</td>
              <td>{fmtScore(localIssue[p])}</td>
              <td>
                <strong>{fmtScore(liveScore[p])}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

function TvkImpactCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as LiveIntelSpecific;
  const leaderNames: Partial<Record<Party, string>> = specific.leader_names ?? {};
  const tvkMetrics: Record<string, number> = specific.tvk_impact_metrics ?? {};

  return (
    <article className="panel analysis-card">
      <h3>TVK / Vijay Impact</h3>
      {leaderNames.TVK && (
        <p className="muted" style={{ marginTop: 0 }}>
          Leader: <strong>{leaderNames.TVK}</strong>
        </p>
      )}
      <table className="compact-table">
        <tbody>
          {Object.entries(tvkMetrics).map(([k, v]) => (
            <tr key={k}>
              <td>{k.split("_").join(" ")}</td>
              <td>
                <strong>{Number(v).toFixed(3)}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

function ConstituencyTable({
  rows,
  analysisType,
}: {
  rows: AnalysisPredictionRow[];
  analysisType: AnalysisType;
}) {
  const columns: { header: string; key: keyof AnalysisPredictionRow }[] = (() => {
    if (analysisType === "long_term_trend") {
      return [
        { header: "2016 Winner", key: "winner_2016" },
        { header: "2021 Winner", key: "winner_2021" },
        { header: "Historical Strength", key: "historical_strength" },
        { header: "Party Growth", key: "party_growth_score" },
        { header: "Long-Term Score", key: "long_term_trend_score" },
      ];
    }
    if (analysisType === "recent_swing") {
      return [
        { header: "2021 Winner", key: "winner_party_2021" },
        { header: "2021 Runner-Up", key: "runner_up_party_2021" },
        { header: "Constituency Swing", key: "constituency_swing" },
        { header: "Retention Prob.", key: "seat_retention_probability" },
        { header: "Swing Score", key: "recent_swing_score" },
      ];
    }
    return [
      { header: "Sentiment-Adjusted", key: "sentiment_adjusted_prediction" },
      { header: "TVK Impact", key: "tvk_impact_score" },
      { header: "Confidence Level", key: "confidence_level" },
      { header: "Live Score", key: "live_intelligence_score" },
    ];
  })();

  const sortedRows = [...rows].sort((a, b) => a.ac_no - b.ac_no);

  return (
    <article className="panel table-panel analysis-table-panel">
      <div className="table-head">
        <h3>Constituency-level {ANALYSIS_LABELS[analysisType]}</h3>
        <div className="table-head-actions">
          <span className="table-meta">{sortedRows.length} constituencies analysed</span>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>AC No</th>
              <th>Constituency</th>
              <th>District</th>
              <th>Predicted</th>
              <th>Win Prob.</th>
              {columns.map((c) => (
                <th key={String(c.key)}>{c.header}</th>
              ))}
              <th>Final Score</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={`${row.ac_no}-${row.constituency}`}>
                <td>{row.ac_no}</td>
                <td>{row.constituency}</td>
                <td>{row.district}</td>
                <td>
                  <PartyBadge party={row.predicted} />
                </td>
                <td>{fmtPct(row.win_probability ?? row.confidence)}</td>
                {columns.map((c) => {
                  const value = row[c.key];
                  let display: string;
                  if (typeof value === "number") {
                    display = c.key === "constituency_swing" || c.key === "party_growth_score"
                      ? `${(value * 100).toFixed(2)}%`
                      : fmtScore(value);
                  } else if (Array.isArray(value)) {
                    display = value.join(", ");
                  } else {
                    display = value !== undefined && value !== "" ? String(value) : "-";
                  }
                  return <td key={String(c.key)}>{display}</td>;
                })}
                <td>
                  <strong>{fmtScore(row.final_prediction_score)}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export function AnalysisPanel({ analysisType }: AnalysisPanelProps) {
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterState>({
    district: "ALL",
    party: "ALL",
    query: "",
  });
  const deferredQuery = useDeferredValue(filter.query);

  const allRows = data?.rows ?? [];

  const districts = useMemo(
    () =>
      [...new Set(allRows.map((r) => r.district))].sort((a, b) =>
        a.localeCompare(b),
      ),
    [allRows],
  );

  const constituencyOptions = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    const base = q
      ? allRows.filter((r) => r.constituency.toLowerCase().includes(q))
      : allRows;
    return [...base]
      .sort((a, b) => a.ac_no - b.ac_no)
      .map((r) => r.constituency);
  }, [allRows, deferredQuery]);

  const filteredRows = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    return allRows.filter((r) => {
      const districtOk = filter.district === "ALL" || r.district === filter.district;
      const winner = r.analysis_predicted ?? r.predicted;
      const partyOk = filter.party === "ALL" || winner === filter.party;
      const queryOk = q.length === 0 || r.constituency.toLowerCase().includes(q);
      return districtOk && partyOk && queryOk;
    });
  }, [allRows, filter.district, filter.party, deferredQuery]);

  const headerSummary = useMemo<HeaderSummary>(
    () => (data ? buildHeaderSummary(data.meta, filteredRows) : EMPTY_HEADER_SUMMARY),
    [data, filteredRows],
  );

  // Reset filters when switching analysis tabs.
  useEffect(() => {
    setFilter({ district: "ALL", party: "ALL", query: "" });
  }, [analysisType]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);

    fetchAnalysisPredictions(analysisType, controller.signal)
      .then((res) => {
        if (controller.signal.aborted) return;
        setData(res);
      })
      .catch((err) => {
        if (
          controller.signal.aborted ||
          (err instanceof DOMException && err.name === "AbortError")
        ) {
          return;
        }
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [analysisType]);

  if (loading) {
    return <div className="panel loading">Loading {ANALYSIS_LABELS[analysisType]}...</div>;
  }
  if (error) {
    return <div className="error-banner">{error}</div>;
  }
  if (!data) return null;

  const { meta } = data;

  const centerCardProps: CenterCardProps = {
    rows: filteredRows,
    districts,
    filter,
    onFilterChange: setFilter,
    constituencyOptions,
  };
  const scoreKey = SCORE_KEY[analysisType];

  return (
    <section className="analysis-section">
      <MetaHeader summary={headerSummary} />

      <section className="middle-stage analysis-middle-stage">
        <aside className="left-stack">
          <DistrictBreakdownCard rows={filteredRows} />
        </aside>

        <CenterCard {...centerCardProps} />

        <aside className="right-stack">
          <CompetitiveSeatsCard rows={filteredRows} scoreKey={scoreKey} />
        </aside>
      </section>

      <div className="analysis-detail-grid">
        {analysisType === "long_term_trend" && (
          <>
            <VoteShareTrendCard meta={meta} />
            <SeatSwingCard meta={meta} />
          </>
        )}
        {analysisType === "recent_swing" && (
          <>
            <AssemblyShareCard meta={meta} />
            <SwingMomentumCard meta={meta} />
          </>
        )}
        {analysisType === "live_intelligence_score" && (
          <>
            <SentimentScoresCard meta={meta} />
            <TvkImpactCard meta={meta} />
          </>
        )}
      </div>

      <ConstituencyTable rows={filteredRows} analysisType={analysisType} />
    </section>
  );
}
