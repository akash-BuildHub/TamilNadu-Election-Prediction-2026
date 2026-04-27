import { useEffect, useRef, useState } from "react";
import { asPercent } from "../utils/format";

interface AnimatedKpiGridProps {
  totalConstituencies: number;
  projectedWinner: string;
  averageWinMargin: number;
}

let hasAnimatedKpiInSession = false;

export function AnimatedKpiGrid({
  totalConstituencies,
  projectedWinner,
  averageWinMargin,
}: AnimatedKpiGridProps) {
  const kpiGridRef = useRef<HTMLElement | null>(null);
  const hasAnimatedRef = useRef(hasAnimatedKpiInSession);
  const previousWinnerRef = useRef(hasAnimatedRef.current ? projectedWinner : "");
  const [animatedWinMargin, setAnimatedWinMargin] = useState(
    hasAnimatedRef.current ? averageWinMargin : 0,
  );
  const [animatedTotal, setAnimatedTotal] = useState(
    hasAnimatedRef.current ? totalConstituencies : 0,
  );
  const [animatedWinner, setAnimatedWinner] = useState(
    hasAnimatedRef.current ? projectedWinner : "",
  );
  const [winnerRollToken, setWinnerRollToken] = useState(
    hasAnimatedRef.current ? 1 : 0,
  );

  useEffect(() => {
    if (!hasAnimatedRef.current) return;
    setAnimatedWinMargin(averageWinMargin);
    setAnimatedTotal(totalConstituencies);
    if (previousWinnerRef.current !== projectedWinner) {
      previousWinnerRef.current = projectedWinner;
      setAnimatedWinner(projectedWinner);
      setWinnerRollToken((prev) => prev + 1);
    }
  }, [averageWinMargin, totalConstituencies, projectedWinner]);

  useEffect(() => {
    const element = kpiGridRef.current;
    if (!element || hasAnimatedRef.current) return;

    let confidenceFrameId = 0;
    let totalFrameId = 0;

    const triggerAnimationOnce = () => {
      if (hasAnimatedRef.current) return;
      hasAnimatedRef.current = true;
      hasAnimatedKpiInSession = true;

      const confidenceDurationMs = 1200;
      const confidenceStart = performance.now();
      const confidenceTarget = averageWinMargin;
      const confidenceTick = (now: number) => {
        const t = Math.min((now - confidenceStart) / confidenceDurationMs, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        setAnimatedWinMargin(confidenceTarget * eased);
        if (t < 1) {
          confidenceFrameId = requestAnimationFrame(confidenceTick);
        }
      };
      confidenceFrameId = requestAnimationFrame(confidenceTick);

      const totalDurationMs = 900;
      const totalStart = performance.now();
      const totalTarget = totalConstituencies;
      const totalTick = (now: number) => {
        const t = Math.min((now - totalStart) / totalDurationMs, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        setAnimatedTotal(Math.round(totalTarget * eased));
        if (t < 1) {
          totalFrameId = requestAnimationFrame(totalTick);
        }
      };
      totalFrameId = requestAnimationFrame(totalTick);

      previousWinnerRef.current = projectedWinner;
      setAnimatedWinner(projectedWinner);
      setWinnerRollToken((prev) => prev + 1);
    };

    if (typeof IntersectionObserver === "undefined") {
      triggerAnimationOnce();
      return () => {
        cancelAnimationFrame(confidenceFrameId);
        cancelAnimationFrame(totalFrameId);
      };
    }

    const observer = new IntersectionObserver((entries, observerInstance) => {
      const shouldAnimate = entries.some(
        (entry) => entry.isIntersecting && entry.intersectionRatio >= 0.35,
      );
      if (!shouldAnimate) return;
      triggerAnimationOnce();
      observerInstance.disconnect();
    }, { threshold: [0, 0.35] });

    observer.observe(element);
    const fallbackTimerId = window.setTimeout(triggerAnimationOnce, 300);

    return () => {
      window.clearTimeout(fallbackTimerId);
      observer.disconnect();
      cancelAnimationFrame(confidenceFrameId);
      cancelAnimationFrame(totalFrameId);
    };
  }, [averageWinMargin, totalConstituencies, projectedWinner]);

  return (
    <section className="kpi-grid" ref={kpiGridRef}>
      <article className="panel kpi-card">
        <h3>Total Constituencies</h3>
        <strong>{animatedTotal}</strong>
      </article>
      <article className="panel kpi-card">
        <h3>Data Reference</h3>
        <strong>2011 &ndash; 2026</strong>
      </article>
      <article className="panel kpi-card">
        <h3>Projected Winner</h3>
        <strong className="winner-roll-box" aria-live="polite">
          <span className="winner-roll-text" key={winnerRollToken}>
            {animatedWinner || "-"}
          </span>
        </strong>
      </article>
      <article className="panel kpi-card">
        <h3>Average Winning Score</h3>
        <strong>{asPercent(animatedWinMargin)}</strong>
      </article>
    </section>
  );
}
