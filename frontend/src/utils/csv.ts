// Lightweight CSV exporter used by the per-tab "Download Final
// Prediction Sheet" buttons. Generates the file in-browser from the
// rows the dashboard has already fetched -- no extra backend round-trip.

export type CsvColumn<TRow> = {
  header: string;
  // Function form lets us project nested / computed values per column.
  value: (row: TRow) => string | number | null | undefined;
};

function escapeCsvCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "";
  const s = typeof value === "number" ? String(value) : value;
  if (/[",\r\n]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

export function rowsToCsv<TRow>(columns: CsvColumn<TRow>[], rows: TRow[]): string {
  const headerLine = columns.map((c) => escapeCsvCell(c.header)).join(",");
  const bodyLines = rows.map((row) =>
    columns.map((c) => escapeCsvCell(c.value(row))).join(",")
  );
  // Prefix UTF-8 BOM so Excel opens Unicode constituency names correctly.
  return "﻿" + [headerLine, ...bodyLines].join("\r\n");
}

export function downloadCsv<TRow>(
  filename: string,
  columns: CsvColumn<TRow>[],
  rows: TRow[]
): void {
  const csv = rowsToCsv(columns, rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  // Revoke on next tick so Safari finishes the download first.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
