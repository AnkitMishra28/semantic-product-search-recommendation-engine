"use client";

import React, { useState } from "react";
import { formatNumber, formatPercent } from "@/lib/utils";

export interface BarDatum {
  label: string;
  value: number;
  secondaryValue?: number;
  highlight?: boolean;
  color?: string;
  tooltipText?: string;
}

interface InteractiveBarChartProps {
  data: BarDatum[];
  isPercent?: boolean;
  digits?: number;
  height?: number;
  title?: string;
  yAxisLabel?: string;
}

export const InteractiveBarChart: React.FC<InteractiveBarChartProps> = ({
  data,
  isPercent = false,
  digits = 4,
  height = 240,
  title,
  yAxisLabel,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) return null;

  const maxValue = Math.max(...data.map((d) => d.value), 0.0001);
  const chartHeight = height - 60;
  const paddingX = 40;
  const paddingY = 20;

  return (
    <div className="glass-panel rounded-2xl border border-slate-800 p-5 space-y-3 shadow-xl">
      {title && (
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-foreground font-sans">{title}</span>
          {yAxisLabel && <span className="text-[10px] font-mono text-muted-foreground">{yAxisLabel}</span>}
        </div>
      )}

      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${Math.max(500, data.length * 100)} ${height}`}
          className="w-full h-auto min-w-[480px]"
          style={{ maxHeight: height }}
        >
          {/* Horizontal Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const y = paddingY + chartHeight * (1 - ratio);
            const val = maxValue * ratio;
            return (
              <g key={ratio}>
                <line
                  x1={paddingX}
                  y1={y}
                  x2="100%"
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={paddingX - 6}
                  y={y + 3}
                  textAnchor="end"
                  fill="#64748b"
                  fontSize="9"
                  fontFamily="monospace"
                >
                  {isPercent ? `${(val * 100).toFixed(0)}%` : val.toFixed(digits === 4 ? 3 : digits)}
                </text>
              </g>
            );
          })}

          {/* Bars */}
          {data.map((item, idx) => {
            const barWidth = 44;
            const slotWidth = (500 - paddingX * 2) / data.length;
            const x = paddingX + idx * slotWidth + (slotWidth - barWidth) / 2;
            const barHeight = Math.max(4, (item.value / maxValue) * chartHeight);
            const y = paddingY + chartHeight - barHeight;
            const isHovered = hoveredIdx === idx;

            const fillColor = item.highlight
              ? "#22d3ee"
              : item.color || "#38bdf8";

            return (
              <g
                key={item.label}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                className="cursor-pointer transition-all duration-200"
              >
                {/* Bar */}
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  rx="6"
                  fill={fillColor}
                  fillOpacity={isHovered ? 1 : item.highlight ? 0.9 : 0.65}
                  stroke={item.highlight ? "#a5f3fc" : "#0284c7"}
                  strokeWidth={isHovered ? "2" : item.highlight ? "1.5" : "0.5"}
                />

                {/* Top Value Label */}
                <text
                  x={x + barWidth / 2}
                  y={y - 6}
                  textAnchor="middle"
                  fill={isHovered || item.highlight ? "#22d3ee" : "#94a3b8"}
                  fontSize="10"
                  fontWeight={item.highlight ? "bold" : "normal"}
                  fontFamily="monospace"
                >
                  {isPercent ? formatPercent(item.value, 2) : formatNumber(item.value, digits)}
                </text>

                {/* Bottom X-axis Label */}
                <text
                  x={x + barWidth / 2}
                  y={paddingY + chartHeight + 18}
                  textAnchor="middle"
                  fill={isHovered ? "#f8fafc" : "#94a3b8"}
                  fontSize="10"
                  fontFamily="sans-serif"
                  fontWeight={isHovered ? "600" : "400"}
                >
                  {item.label.length > 14 ? `${item.label.slice(0, 13)}…` : item.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {hoveredIdx !== null && data[hoveredIdx] && (
        <div className="rounded-xl border border-cyan-400/30 bg-slate-900/90 p-2.5 text-xs text-cyan-200 flex items-center justify-between font-mono animate-in fade-in duration-150">
          <span className="font-semibold">{data[hoveredIdx].label}</span>
          <span className="font-bold text-cyan-300">
            {isPercent
              ? formatPercent(data[hoveredIdx].value, 4)
              : formatNumber(data[hoveredIdx].value, 4)}
          </span>
          {data[hoveredIdx].tooltipText && (
            <span className="text-[11px] text-muted-foreground font-sans">
              ({data[hoveredIdx].tooltipText})
            </span>
          )}
        </div>
      )}
    </div>
  );
};
