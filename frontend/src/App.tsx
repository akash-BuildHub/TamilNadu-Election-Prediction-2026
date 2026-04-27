import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { PartyBadge } from "./components/PartyBadge";
import { AnimatedKpiGrid } from "./components/AnimatedKpiGrid";
import { AnalysisPanel } from "./components/AnalysisPanel";
import {
  API_BASE,
  EXPECTED_API_VERSION,
  EXPECTED_PREDICTIONS_SHA256,
  fetchHealth,
  fetchPredictions,
  fetchPredictionsMeta,
} from "./services/api";
import {
  ANALYSIS_LABELS,
  ANALYSIS_TYPES,
  AnalysisType,
  PARTIES,
  PARTY_LABELS,
  PredictionRow,
  Party,
  PredictionsMeta,
} from "./types/prediction";
import { asPercentPrecise, asPercentSmart, asSeatPercent } from "./utils/format";

// Parties displayed in bars / filters / table columns.
const DISPLAY_PARTIES: Party[] = ["DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS"];
const EXPECTED_TOTAL_CONSTITUENCIES = 234;
let hasAnimatedMiddleStageInSession = false;

function zeroCounts(): Record<Party, number> {
  return { DMK_ALLIANCE: 0, AIADMK_NDA: 0, TVK: 0, NTK: 0, OTHERS: 0 };
}

function getSeatCounts(rows: PredictionRow[]) {
  const counts = zeroCounts();
  for (const row of rows) counts[row.predicted] += 1;
  return counts;
}

function toCountsLine(counts: Record<Party, number>): string {
  return PARTIES.map((p) => `${PARTY_LABELS[p]} ${counts[p]}`).join(", ");
}

function seatCountsMatch(a: Record<Party, number>, b: Record<Party, number>): boolean {
  return PARTIES.every((p) => a[p] === b[p]);
}

function validatePredictionMeta(meta: PredictionsMeta): string | null {
  if (import.meta.env.PROD && !EXPECTED_PREDICTIONS_SHA256) {
    return "Missing VITE_EXPECTED_PREDICTIONS_SHA256 in production frontend environment.";
  }

  if (meta.fallback_in_use) {
    return "Backend is serving fallback data (tamilnadu_assembly_2026.csv), not final predictions_2026.csv.";
  }

  if (meta.source_file !== "predictions_2026.csv") {
    return `Unexpected prediction source file: ${meta.source_file}. Expected predictions_2026.csv.`;
  }

  if (meta.total_constituencies !== EXPECTED_TOTAL_CONSTITUENCIES) {
    return `Unexpected constituency count: ${meta.total_constituencies}. Expected ${EXPECTED_TOTAL_CONSTITUENCIES}.`;
  }

  if (EXPECTED_API_VERSION && meta.api_version && meta.api_version !== EXPECTED_API_VERSION) {
    return `Backend API version mismatch. Expected ${EXPECTED_API_VERSION}, got ${meta.api_version}.`;
  }

  if (
    EXPECTED_PREDICTIONS_SHA256 &&
    (!meta.source_sha256 ||
      meta.source_sha256.toLowerCase() !== EXPECTED_PREDICTIONS_SHA256.toLowerCase())
  ) {
    const actualHash = meta.source_sha256 ? meta.source_sha256.toLowerCase() : "missing";
    return `Backend predictions hash mismatch. Expected ${EXPECTED_PREDICTIONS_SHA256}, got ${actualHash}. Redeploy backend and frontend from the same commit.`;
  }

  return null;
}

export function App() {
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [analysisType, setAnalysisType] = useState<AnalysisType | "default">("default");
  const [district, setDistrict] = useState("ALL");
  const [party, setParty] = useState<Party | "ALL">("ALL");
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [districtOpen, setDistrictOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement | null>(null);
  const districtRef = useRef<HTMLDivElement | null>(null);
  const middleStageRef = useRef<HTMLElement | null>(null);
  const hasAnimatedMiddleStageRef = useRef(hasAnimatedMiddleStageInSession);
  const prefersReducedMotion = useReducedMotion();
  const deferredQuery = useDeferredValue(query);

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

  useEffect(() => {
    const controller = new AbortController();

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const health = await fetchHealth(controller.signal);
        if (health.status !== "ok") {
          throw new Error(health.error || "Backend health check failed.");
        }

        let meta = health.meta;
        if (!meta) {
          meta = await fetchPredictionsMeta(controller.signal);
        }

        const contractError = validatePredictionMeta(meta);
        if (contractError) throw new Error(contractError);

        const predictions = await fetchPredictions(controller.signal);
        if (controller.signal.aborted) return;

        const fetchedSeatCounts = getSeatCounts(predictions);
        if (!seatCountsMatch(fetchedSeatCounts, meta.seat_counts)) {
          throw new Error(
            `Backend metadata and predictions differ. Meta: ${toCountsLine(meta.seat_counts)}. Payload: ${toCountsLine(fetchedSeatCounts)}.`
          );
        }

        setRows(predictions);
      } catch (err) {
        if (
          controller.signal.aborted ||
          (err instanceof DOMException && err.name === "AbortError") ||
          (err instanceof Error && err.name === "AbortError")
        ) {
          return;
        }
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(`${message} Ensure backend is running on ${API_BASE}.`);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      controller.abort();
    };
  }, []);

  const districts = useMemo(() => {
    return [...new Set(rows.map((r) => r.district))].sort((a, b) => a.localeCompare(b));
  }, [rows]);

  const constituencyOptions = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    const base = q
      ? rows.filter((r) => r.constituency.toLowerCase().includes(q))
      : rows;
    return [...base]
      .sort((a, b) => a.ac_no - b.ac_no)
      .map((r) => r.constituency);
  }, [rows, deferredQuery]);

  const filteredRows = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();

    const next = rows.filter((r) => {
      const districtOk = district === "ALL" || r.district === district;
      const partyOk = party === "ALL" || r.predicted === party;
      const queryOk = q.length === 0 || r.constituency.toLowerCase().includes(q);
      return districtOk && partyOk && queryOk;
    });

    next.sort((a, b) => a.ac_no - b.ac_no);

    return next;
  }, [rows, district, party, deferredQuery]);

  const seatCounts = useMemo(() => getSeatCounts(filteredRows), [filteredRows]);
  const sortedParties = useMemo(() => {
    // Sort by seat count desc, but always pin OTHERS to the last position.
    return [...DISPLAY_PARTIES].sort((a, b) => {
      if (a === "OTHERS") return 1;
      if (b === "OTHERS") return -1;
      return seatCounts[b] - seatCounts[a];
    });
  }, [seatCounts]);
  const total = filteredRows.length;
  const safeTotal = total || 1;
  const projectedWinner = useMemo<Party | "-">(() => {
    if (total === 0) return "-";
    let winner: Party = PARTIES[0];
    for (const partyName of PARTIES) {
      if (seatCounts[partyName] > seatCounts[winner]) {
        winner = partyName;
      }
    }
    return winner;
  }, [total, seatCounts]);
  const averageWinMargin = useMemo(() => {
    return filteredRows.reduce((sum, r) => sum + r.confidence, 0) / safeTotal;
  }, [filteredRows, safeTotal]);
  const closestSeats = useMemo(() => {
    return [...filteredRows]
      .sort((a, b) => a.confidence - b.confidence)
      .slice(0, 8);
  }, [filteredRows]);

  const districtBreakdown = useMemo(() => {
    const map = new Map<string, Record<Party, number>>();

    for (const row of filteredRows) {
      if (!map.has(row.district)) {
        map.set(row.district, zeroCounts());
      }
      map.get(row.district)![row.predicted] += 1;
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
  }, [filteredRows]);

  useEffect(() => {
    const element = middleStageRef.current;
    if (!element || loading || error || hasAnimatedMiddleStageRef.current) return;

    const triggerStageAnimationOnce = () => {
      if (hasAnimatedMiddleStageRef.current) return;
      hasAnimatedMiddleStageRef.current = true;
      hasAnimatedMiddleStageInSession = true;
      element.classList.add("animate-cards");
    };

    if (typeof IntersectionObserver === "undefined") {
      triggerStageAnimationOnce();
      return;
    }

    const observer = new IntersectionObserver((entries, observerInstance) => {
      const shouldAnimate = entries.some(
        (entry) => entry.isIntersecting && entry.intersectionRatio >= 0.12,
      );
      if (!shouldAnimate) return;
      triggerStageAnimationOnce();
      observerInstance.disconnect();
    }, { threshold: [0, 0.12] });

    observer.observe(element);
    const fallbackTimerId = window.setTimeout(triggerStageAnimationOnce, 300);
    return () => {
      window.clearTimeout(fallbackTimerId);
      observer.disconnect();
    };
  }, [loading, error]);

  const projectedWinnerLabel =
    projectedWinner === "-" ? "-" : PARTY_LABELS[projectedWinner];

  return (
    <div className="app-shell">
      <div className="bg-blur bg-blur-a" />
      <div className="bg-blur bg-blur-b" />

      <main className="container">
        <header className="hero">
          <div className="hero-inner">
            <div className="brand-line" aria-label="QVotelytics">
              <img
                src="/assets/owlytics"
                alt="Q logo"
                className="q-logo"
                width={56}
                height={56}
                decoding="async"
              />
              <h1 className="brand-title">Election Prediction</h1>
            </div>
            <p className="hero-tagline">
              Our <span className="accent">Intelligent AI</span> tracked every
              vote across <span className="accent">Tamil Nadu&rsquo;s</span>{" "}
              constituencies, uncovered key trends, and predicted who will form
              the next government.
            </p>
          </div>
        </header>

        {error && <div className="error-banner">{error}</div>}
        {loading && <div className="panel loading">Loading predictions...</div>}

        {!loading && !error && (
          <>
            <nav className="analysis-tabs" aria-label="Prediction analysis filter">
              <button
                type="button"
                className={`analysis-tab ${analysisType === "default" ? "active" : ""}`}
                onClick={() => setAnalysisType("default")}
              >
                Historical Projection
              </button>
              {ANALYSIS_TYPES.map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`analysis-tab ${analysisType === t ? "active" : ""}`}
                  onClick={() => setAnalysisType(t)}
                >
                  {ANALYSIS_LABELS[t]}
                </button>
              ))}
            </nav>

            {analysisType !== "default" && (
              <AnalysisPanel analysisType={analysisType} />
            )}
          </>
        )}

        {!loading && !error && analysisType === "default" && (
          <>
            <AnimatedKpiGrid
              totalConstituencies={total}
              projectedWinner={projectedWinnerLabel}
              averageWinMargin={averageWinMargin}
            />

            <section className="middle-stage" ref={middleStageRef}>
              <aside className="left-stack">
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
              </aside>

              <article className="panel center-card">
                <div className="center-inner-grid">
                  <section className="inner-block filters-block">
                    <h2>Filters</h2>
                    <div className="filters-grid">
                      <div className="combo-wrap" ref={districtRef}>
                        <label htmlFor="district">District</label>
                        <button
                          id="district"
                          type="button"
                          className="combo-toggle"
                          aria-haspopup="listbox"
                          aria-expanded={districtOpen}
                          onClick={() => setDistrictOpen((o) => !o)}
                        >
                          <span>{district === "ALL" ? "All Districts" : district}</span>
                          <span className="combo-chevron" aria-hidden="true">
                            ▾
                          </span>
                        </button>
                        {districtOpen && (
                          <ul className="combo-list" role="listbox">
                            <li
                              role="option"
                              aria-selected={district === "ALL"}
                              className="combo-item"
                              onClick={() => {
                                setDistrict("ALL");
                                setDistrictOpen(false);
                              }}
                            >
                              All Districts
                            </li>
                            {districts.map((d) => (
                              <li
                                key={d}
                                role="option"
                                aria-selected={district === d}
                                className="combo-item"
                                onClick={() => {
                                  setDistrict(d);
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
                        <label htmlFor="party">Predicted Party</label>
                        <select
                          id="party"
                          value={party}
                          onChange={(e) => setParty(e.target.value as Party | "ALL")}
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
                        <label htmlFor="search">Search Constituency</label>
                        <input
                          id="search"
                          type="text"
                          value={query}
                          placeholder="Type or select..."
                          autoComplete="off"
                          role="combobox"
                          aria-expanded={searchOpen}
                          aria-controls="constituency-list"
                          onChange={(e) => {
                            setQuery(e.target.value);
                            setSearchOpen(true);
                          }}
                          onFocus={() => setSearchOpen(true)}
                        />
                        {searchOpen && constituencyOptions.length > 0 && (
                          <ul
                            id="constituency-list"
                            className="combo-list"
                            role="listbox"
                          >
                            {constituencyOptions.map((name) => (
                              <li
                                key={name}
                                role="option"
                                aria-selected={query === name}
                                className="combo-item"
                                onClick={() => {
                                  setQuery(name);
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
                              {seatCounts[p]} ({asSeatPercent(seatCounts[p], safeTotal)})
                            </span>
                          </div>
                          <div className="bar-track">
                            <div
                              className={`bar-fill fill-${p}`}
                              style={{ width: `${(seatCounts[p] / safeTotal) * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>
              </article>

              <aside className="right-stack">
                <article className="panel">
                  <h2>Most Competitive Seats</h2>
                  <ul className="tight-list">
                    {closestSeats.map((seat) => (
                      <li key={`${seat.ac_no}-${seat.constituency}`}>
                        <div>
                          <strong>{seat.constituency}</strong>
                          <small>{seat.district}</small>
                        </div>
                        <div className="right-inline">
                          <PartyBadge party={seat.predicted} />
                          <span>{asPercentSmart(seat.confidence)} Score</span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </article>
              </aside>
            </section>

            <motion.section
              className="panel table-panel explorer-section"
              initial={prefersReducedMotion ? false : { opacity: 1, y: 20 }}
              whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.15 }}
              transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            >
              <motion.div
                className="table-head"
                initial={prefersReducedMotion ? false : { opacity: 1, y: 10 }}
                whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.35, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="explorer-title-block">
                  <h2 className="explorer-title">Constituency-level Historical Projection</h2>
                  <p className="explorer-subtitle">{rows.length} constituencies analysed</p>
                </div>
              </motion.div>
              <motion.div
                className="table-wrap"
                initial={prefersReducedMotion ? false : { opacity: 1, y: 14 }}
                whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.1 }}
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
                    {filteredRows.map((row) => (
                      <tr key={`${row.ac_no}-${row.constituency}`}>
                        <td>{row.ac_no}</td>
                        <td>{row.constituency}</td>
                        <td>{row.district}</td>
                        <td>
                          <PartyBadge party={row.predicted} />
                        </td>
                        <td>{asPercentSmart(row.confidence)}</td>
                        <td>{asPercentPrecise(row.DMK_ALLIANCE)}</td>
                        <td>{asPercentPrecise(row.AIADMK_NDA)}</td>
                        <td>{asPercentPrecise(row.TVK)}</td>
                        <td>{asPercentPrecise(row.NTK)}</td>
                        <td>{asPercentPrecise(row.OTHERS)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </motion.div>
            </motion.section>
          </>
        )}
      </main>
    </div>
  );
}
