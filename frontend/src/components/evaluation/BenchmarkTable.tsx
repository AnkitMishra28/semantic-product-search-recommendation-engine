"use client";

import React from "react";
import { formatNumber, formatPercent } from "@/lib/utils";

export interface BenchmarkColumn {
  key: string;
  label: string;
  isPercent?: boolean;
  digits?: number;
}

export interface BenchmarkRow {
  name: string;
  highlight?: boolean;
  values: Record<string, number | string | null | undefined>;
}

interface BenchmarkTableProps {
  columns: BenchmarkColumn[];
  rows: BenchmarkRow[];
  caption?: string;
}

/**
 * Generic metric comparison table rendered directly from backend-sourced
 * JSON — no numbers in this component are computed or estimated client-side.
 */
export const BenchmarkTable: React.FC<BenchmarkTableProps> = ({ columns, rows, caption }) => {
  return (
    <div className="glass-panel overflow-hidden rounded-2xl border border-slate-800 shadow-xl">
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse min-w-[580px]">
          <thead>
            <tr className="bg-slate-950/80 border-b border-slate-800">
              <th scope="col" className="text-left font-semibold text-muted-foreground px-4 py-3 font-mono text-[11px] uppercase tracking-wider">
                Method / Pipeline
              </th>
              {columns.map((col) => (
                <th key={col.key} scope="col" className="text-right font-mono font-semibold text-muted-foreground px-4 py-3 text-[11px] uppercase tracking-wider">
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.name}
                className={`border-b border-slate-800/60 last:border-0 transition-colors ${
                  row.highlight ? "bg-cyan-400/10 hover:bg-cyan-400/15" : "hover:bg-secondary/30"
                }`}
              >
                <td className="px-4 py-3 font-semibold text-foreground whitespace-nowrap flex items-center gap-2">
                  {row.highlight && <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 shrink-0" />}
                  <span>{row.name}</span>
                </td>
                {columns.map((col) => {
                  const raw = row.values[col.key];
                  const display =
                    typeof raw === "number"
                      ? col.isPercent
                        ? formatPercent(raw, col.digits ?? 2)
                        : formatNumber(raw, col.digits ?? 4)
                      : raw ?? "—";
                  return (
                    <td
                      key={col.key}
                      className={`px-4 py-3 text-right font-mono text-xs ${
                        row.highlight ? "text-cyan-300 font-bold" : "text-foreground/90 font-medium"
                      }`}
                    >
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && (
        <p className="px-4 py-2.5 text-[11px] text-muted-foreground bg-slate-950/60 border-t border-slate-800/80">
          {caption}
        </p>
      )}
    </div>
  );
};
