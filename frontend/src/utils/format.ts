export function asPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function asPercentSmart(value: number): string {
  const percent = value * 100;
  const absPercent = Math.abs(percent);

  if (absPercent < 1) return `${percent.toFixed(3)}%`;
  if (absPercent < 10) return `${percent.toFixed(2)}%`;
  return `${percent.toFixed(1)}%`;
}

export function asPercentPrecise(value: number, decimals = 3): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function asSeatPercent(seats: number, total: number): string {
  if (total <= 0) return "0.0%";
  return `${((seats / total) * 100).toFixed(1)}%`;
}
