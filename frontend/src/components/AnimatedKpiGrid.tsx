import { ReactNode } from "react";

export type KpiCard = {
  heading: string;
  value: ReactNode;
};

type AnimatedKpiGridProps = {
  cards: KpiCard[];
  /** Changing this value forces every card's heading + value to re-fire the
   *  roll-in animation. Pass the active analysis tab id (e.g. "default",
   *  "long_term_trend") so switching tabs replays the entrance. */
  animateToken?: string | number;
};

/** Per-card stagger so the four cards roll in sequence rather than in
 *  unison. Heading rolls in first; value follows ~120ms later. */
const HEADING_STAGGER_MS = 70;
const VALUE_STAGGER_MS = 70;
const VALUE_HEAD_OFFSET_MS = 120;

export function AnimatedKpiGrid({ cards, animateToken = "" }: AnimatedKpiGridProps) {
  return (
    <section className="kpi-grid">
      {cards.map((card, i) => (
        <article className="panel kpi-card" key={i}>
          <h3>
            <span className="kpi-roll-box">
              <span
                className="kpi-roll-text"
                key={`h-${i}-${animateToken}`}
                style={{ ["--kpi-roll-delay" as string]: `${i * HEADING_STAGGER_MS}ms` }}
              >
                {card.heading}
              </span>
            </span>
          </h3>
          <strong aria-live="polite">
            <span className="kpi-roll-box">
              <span
                className="kpi-roll-text"
                key={`v-${i}-${animateToken}-${stringifyValue(card.value)}`}
                style={{
                  ["--kpi-roll-delay" as string]:
                    `${VALUE_HEAD_OFFSET_MS + i * VALUE_STAGGER_MS}ms`,
                }}
              >
                {card.value}
              </span>
            </span>
          </strong>
        </article>
      ))}
    </section>
  );
}

function stringifyValue(v: ReactNode): string {
  if (typeof v === "string" || typeof v === "number") return String(v);
  return "";
}
