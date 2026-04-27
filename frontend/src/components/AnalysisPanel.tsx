import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
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
import { asPercentPrecise, asPercentSmart, asSeatPercent } from "../utils/format";
import { AnimatedKpiGrid } from "./AnimatedKpiGrid";
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

function fmtScore(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(3);
}

// Used in every analysis-tab section so all four cards (KPI, bar chart,
// district breakdown, competitive list, table) report the same number for
// the same seat. Mirrors the Default view's reliance on `row.confidence`.
function rowWinningScore(row: AnalysisPredictionRow): number {
  return row.win_probability ?? row.confidence ?? 0;
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
    dataReference = "LIVE DATA";
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
    () =>
      [...DISPLAY_PARTIES].sort((a, b) => {
        // OTHERS always pinned to the last position.
        if (a === "OTHERS") return 1;
        if (b === "OTHERS") return -1;
        return counts[b] - counts[a];
      }),
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

function CompetitiveSeatsCard({ rows }: { rows: AnalysisPredictionRow[] }) {
  const closestSeats = useMemo(() => {
    return [...rows]
      .sort((a, b) => rowWinningScore(a) - rowWinningScore(b))
      .slice(0, 8);
  }, [rows]);

  return (
    <article className="panel">
      <h2>Most Competitive Seats</h2>
      <ul className="tight-list">
        {closestSeats.map((seat) => {
          const winner = (seat.analysis_predicted ?? seat.predicted) as Party;
          return (
            <li key={`${seat.ac_no}-${seat.constituency}`}>
              <div>
                <strong>{seat.constituency}</strong>
                <small>{seat.district}</small>
              </div>
              <div className="right-inline">
                <PartyBadge party={winner} />
                <span>{asPercentSmart(rowWinningScore(seat))} Score</span>
              </div>
            </li>
          );
        })}
      </ul>
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
      <p className="analysis-card-tagline">
        Real-time sentiment analysis transforming public opinion into actionable
        election insights
      </p>
      <div className="compact-table-wrap">
        <table className="compact-table">
        <thead>
          <tr>
            <th>Alliance</th>
            <th title="Sentiment about the party itself">Party</th>
            <th title="Sentiment about the alliance leader">Leader</th>
            <th title="Sentiment about local candidates">Candidate</th>
            <th title="Social-media sentiment">Social</th>
            <th title="News-media sentiment">News</th>
            <th title="Local issues / governance sentiment">Issues</th>
            <th title="Predicted vote strength: weighted blend of the columns to the left (leader and social-media signals carry the most weight, followed by news, party, candidate, and local-issue sentiment)">Vote Strength</th>
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
      </div>
    </article>
  );
}

const TVK_METRIC_LABELS: Record<string, string> = {
  tvk_state_buzz_score: "TVK statewide buzz",
  vijay_personal_rating: "Vijay personal rating",
  youth_pull_score: "Youth pull",
  first_time_voter_score: "First-time voter pull",
  media_coverage_score: "Media coverage",
  vote_split_risk_dmk: "Vote-split risk (DMK)",
  vote_split_risk_aiadmk: "Vote-split risk (AIADMK)",
  expected_vote_share_2026: "Expected 2026 vote share",
};

const TVK_PERCENT_KEYS = new Set([
  "expected_vote_share_2026",
  "vote_split_risk_dmk",
  "vote_split_risk_aiadmk",
]);

function prettyTvkLabel(key: string): string {
  return (
    TVK_METRIC_LABELS[key] ||
    key
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  );
}

function formatTvkValue(key: string, value: number): string {
  if (TVK_PERCENT_KEYS.has(key)) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(2);
}

function TvkImpactCard({ meta }: { meta: AnalysisMeta }) {
  const specific = meta.analysis_specific as LiveIntelSpecific;
  const tvkMetrics: Record<string, number> = specific.tvk_impact_metrics ?? {};

  return (
    <article className="panel analysis-card">
      <h3>TVK / Vijay Impact</h3>
      <p className="analysis-card-tagline">
        From fanbase to vote base &mdash; decoding Vijay&rsquo;s political impact.
      </p>
      <div className="tvk-impact-grid">
        {Object.entries(tvkMetrics).map(([k, v]) => (
          <div className="tvk-impact-cell" key={k}>
            <span className="tvk-impact-label">{prettyTvkLabel(k)}</span>
            <strong className="tvk-impact-value">
              {formatTvkValue(k, Number(v))}
            </strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function ConstituencyTable({
  rows,
  analysisType,
  prefersReducedMotion,
}: {
  rows: AnalysisPredictionRow[];
  analysisType: AnalysisType;
  prefersReducedMotion: boolean | null;
}) {
  const sortedRows = [...rows].sort((a, b) => a.ac_no - b.ac_no);

  return (
    <motion.article
      className="panel table-panel explorer-section"
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <motion.div
        className="table-head"
        initial={prefersReducedMotion ? false : { opacity: 0, y: 10 }}
        animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="explorer-title-block">
          <h2 className="explorer-title">
            Constituency-level {ANALYSIS_LABELS[analysisType]}
          </h2>
          <p className="explorer-subtitle">
            {sortedRows.length} constituencies analysed
          </p>
        </div>
      </motion.div>
      <motion.div
        className="table-wrap"
        initial={prefersReducedMotion ? false : { opacity: 0, y: 14 }}
        animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
        transition={{ duration: 0.38, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
      >
        <table>
          <thead>
            <tr>
              <th>AC No</th>
              <th>Constituency</th>
              <th>District</th>
              <th>Predicted</th>
              <th>Winning Score</th>
              <th>DMK</th>
              <th>AIADMK</th>
              <th>TVK</th>
              <th>NTK</th>
              <th>Others</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => {
              const winner = (row.analysis_predicted ?? row.predicted) as Party;
              return (
                <tr key={`${row.ac_no}-${row.constituency}`}>
                  <td>{row.ac_no}</td>
                  <td>{row.constituency}</td>
                  <td>{row.district}</td>
                  <td>
                    <PartyBadge party={winner} />
                  </td>
                  <td>{asPercentSmart(rowWinningScore(row))}</td>
                  <td>{asPercentPrecise(row.DMK_ALLIANCE)}</td>
                  <td>{asPercentPrecise(row.AIADMK_NDA)}</td>
                  <td>{asPercentPrecise(row.TVK)}</td>
                  <td>{asPercentPrecise(row.NTK)}</td>
                  <td>{asPercentPrecise(row.OTHERS)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </motion.div>
    </motion.article>
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
  const middleStageRef = useRef<HTMLElement | null>(null);
  const prefersReducedMotion = useReducedMotion();

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

  // Re-fire the middle-stage card entrance every time the analysis panel
  // mounts (i.e. on every tab switch via `key={analysisType}` from App.tsx)
  // and again as soon as data lands. Mirrors the Default tab so all four
  // sections share the same entrance animation.
  useEffect(() => {
    const element = middleStageRef.current;
    if (!element || loading || error) return;
    element.classList.remove("animate-cards");
    void element.offsetWidth;
    element.classList.add("animate-cards");
  }, [loading, error, !!data]);

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

  return (
    <section className="analysis-section">
      <AnimatedKpiGrid
        animateToken={`${analysisType}-${headerSummary.totalConstituencies}-${headerSummary.projectedWinner}`}
        cards={[
          {
            heading: "Total Constituencies",
            value: headerSummary.totalConstituencies,
          },
          { heading: "Data Reference", value: headerSummary.dataReference },
          { heading: "Projected Winner", value: headerSummary.projectedWinner },
          {
            heading: "Average Winning Score",
            value: headerSummary.averageWinningScore,
          },
        ]}
      />

      <section className="middle-stage" ref={middleStageRef}>
        <aside className="left-stack">
          <DistrictBreakdownCard rows={filteredRows} />
        </aside>

        <CenterCard {...centerCardProps} />

        <aside className="right-stack">
          <CompetitiveSeatsCard rows={filteredRows} />
        </aside>
      </section>

      {analysisType === "live_intelligence_score" && (
        <div className="analysis-detail-grid">
          <SentimentScoresCard meta={meta} />
          <TvkImpactCard meta={meta} />
        </div>
      )}

      <ConstituencyTable
        rows={filteredRows}
        analysisType={analysisType}
        prefersReducedMotion={prefersReducedMotion}
      />
    </section>
  );
}
