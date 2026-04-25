import { useEffect, useState } from "react";
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

function PartyScoreBars({
  scores,
  totalSeats,
}: {
  scores: Record<Party, number>;
  totalSeats: number;
}) {
  const safeTotal = totalSeats || 1;
  const sorted = [...PARTIES].sort((a, b) => scores[b] - scores[a]);
  return (
    <div className="bar-list">
      {sorted.map((p) => (
        <div className="bar-item" key={p}>
          <div className="bar-top">
            <span>{PARTY_LABELS[p]}</span>
            <span>
              {scores[p].toFixed(3)} ({asSeatPercent(scores[p] * 100, safeTotal)})
            </span>
          </div>
          <div className="bar-track">
            <div
              className={`bar-fill fill-${p}`}
              style={{ width: `${Math.min(100, scores[p] * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function MetaHeader({ meta }: { meta: AnalysisMeta }) {
  return (
    <div className="analysis-meta-grid">
      <div>
        <span className="muted">Selected Analysis</span>
        <strong>{ANALYSIS_LABELS[meta.analysis_type]}</strong>
      </div>
      <div>
        <span className="muted">CM Election Year</span>
        <strong>{meta.cm_election_year}</strong>
      </div>
      <div>
        <span className="muted">PM / Lok Sabha Reference</span>
        <strong>{meta.lok_sabha_reference_years.join(", ")}</strong>
      </div>
      <div>
        <span className="muted">Gap Years</span>
        <strong>{meta.gap_years}</strong>
      </div>
      <div>
        <span className="muted">Gap Category</span>
        <strong>{meta.gap_category}</strong>
      </div>
      <div>
        <span className="muted">Mode</span>
        <strong>{meta.prediction_mode ? "Prediction" : "Final Result"}</strong>
      </div>
    </div>
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
    <article className="panel">
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

function PartyScorePanel({
  meta,
  totalSeats,
}: {
  meta: AnalysisMeta;
  totalSeats: number;
}) {
  return (
    <article className="panel">
      <h3>Party-wise Score (state level)</h3>
      <PartyScoreBars
        scores={meta.party_average_score as Record<Party, number>}
        totalSeats={totalSeats}
      />
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
    <article className="panel">
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
    <article className="panel">
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
    <article className="panel">
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
    <article className="panel">
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
    <article className="panel">
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
        { header: "Incumbency", key: "incumbency_status" },
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
    <article className="panel table-panel">
      <div className="table-head">
        <h3>Constituency-level {ANALYSIS_LABELS[analysisType]}</h3>
        <span className="table-meta">{sortedRows.length} constituencies analysed</span>
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

  const { meta, rows } = data;

  return (
    <section className="analysis-section">
      <article className="panel">
        <h2>{ANALYSIS_LABELS[analysisType]}</h2>
        <MetaHeader meta={meta} />
      </article>

      {analysisType === "long_term_trend" && (
        <div className="analysis-trio-grid">
          <VoteShareTrendCard meta={meta} />
          <PartyScorePanel meta={meta} totalSeats={rows.length} />
          <SeatSwingCard meta={meta} />
        </div>
      )}

      {analysisType === "live_intelligence_score" && (
        <div className="analysis-trio-grid">
          <SentimentScoresCard meta={meta} />
          <PartyScorePanel meta={meta} totalSeats={rows.length} />
          <TvkImpactCard meta={meta} />
        </div>
      )}

      {analysisType === "recent_swing" && (
        <div className="analysis-trio-grid">
          <AssemblyShareCard meta={meta} />
          <PartyScorePanel meta={meta} totalSeats={rows.length} />
          <SwingMomentumCard meta={meta} />
        </div>
      )}

      <ConstituencyTable rows={rows} analysisType={analysisType} />
    </section>
  );
}
