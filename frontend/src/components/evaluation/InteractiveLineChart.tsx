"use client";

import React, { useState } from "react";
import { formatNumber } from "@/lib/utils";

export interface LineSeries {
  name: string;
  color: string;
  data: { x: number | string; y: number; label?: string }[];
  isPercent?: boolean;
}

interface InteractiveLineChartProps {
  series: LineSeries[];
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  highlightX?: number | string;
  highlightNote?: string;
  height?: number;
}

export const InteractiveLineChart: React.FC<InteractiveLineChartProps> = ({
  series,
  title,
  xAxisLabel,
  yAxisLabel,
  highlightX,
  highlightNote,
  height = 280,
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<{
    seriesName: string;
    x: number | string;
    y: number;
    color: string;
  } | null>(null);

  if (!series || series.length === 0 || series[0].data.length === 0) return null;

  const pointsCount = series[0].data.length;
  const allY = series.flatMap((s) => s.data.map((d) => d.y));
  const minY = Math.min(...allY, 0);
  const maxY = Math.max(...allY, 0.0001);
  const yRange = maxY - minY || 1;

  const paddingLeft = 50;
  const paddingRight = 40;
  const paddingTop = 30;
  const paddingBottom = 40;
  const chartWidth = 560;
  const chartHeight = height - paddingTop - paddingBottom;

  const getX = (idx: number) => {
    if (pointsCount === 1) return paddingLeft + chartWidth / 2;
    return paddingLeft + (idx / (pointsCount - 1)) * (chartWidth - paddingLeft - paddingRight);
  };

  const getY = (val: number) => {
    return paddingTop + chartHeight - ((val - minY) / yRange) * chartHeight;
  };

  return (
    <div className="glass-panel rounded-2xl border border-slate-800 p-5 space-y-3 shadow-xl">
      {title && (
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-foreground font-sans">{title}</span>
          <div className="flex items-center gap-4 text-[11px] font-mono">
            {series.map((s) => (
              <div key={s.name} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: s.color }} />
                <span className="text-muted-foreground">{s.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${chartWidth} ${height}`}
          className="w-full h-auto min-w-[500px]"
          style={{ maxHeight: height }}
        >
          {/* Y Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const val = minY + ratio * yRange;
            const y = getY(val);
            return (
              <g key={ratio}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={chartWidth - paddingRight}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  fill="#64748b"
                  fontSize="9"
                  fontFamily="monospace"
                >
                  {formatNumber(val, 3)}
                </text>
              </g>
            );
          })}

          {/* Highlight Marker for Optimal/Default Parameter */}
          {highlightX !== undefined && (
            (() => {
              const hIdx = series[0].data.findIndex((d) => String(d.x) === String(highlightX));
              if (hIdx === -1) return null;
              const hX = getX(hIdx);
              return (
                <g>
                  <line
                    x1={hX}
                    y1={paddingTop}
                    x2={hX}
                    y2={paddingTop + chartHeight}
                    stroke="#22d3ee"
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                    opacity="0.8"
                  />
                  <rect
                    x={hX - 30}
                    y={paddingTop - 18}
                    width="60"
                    height="16"
                    rx="4"
                    fill="#083344"
                    stroke="#22d3ee"
                    strokeWidth="1"
                  />
                  <text
                    x={hX}
                    y={paddingTop - 6}
                    textAnchor="middle"
                    fill="#22d3ee"
                    fontSize="9"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    {highlightNote || "OPTIMAL"}
                  </text>
                </g>
              );
            })()
          )}

          {/* Series Lines */}
          {series.map((s) => {
            const pathData = s.data
              .map((d, idx) => `${idx === 0 ? "M" : "L"} ${getX(idx)} ${getY(d.y)}`)
              .join(" ");

            return (
              <g key={s.name}>
                <path
                  d={pathData}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.85"
                />
                {/* Data Points */}
                {s.data.map((d, idx) => {
                  const cx = getX(idx);
                  const cy = getY(d.y);
                  const isHighlighted = highlightX !== undefined && String(d.x) === String(highlightX);

                  return (
                    <g
                      key={idx}
                      onMouseEnter={() =>
                        setHoveredPoint({
                          seriesName: s.name,
                          x: d.x,
                          y: d.y,
                          color: s.color,
                        })
                      }
                      onMouseLeave={() => setHoveredPoint(null)}
                      className="cursor-pointer"
                    >
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isHighlighted ? 5.5 : 4}
                        fill={isHighlighted ? "#a5f3fc" : s.color}
                        stroke="#030712"
                        strokeWidth="2"
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* X Axis Labels */}
          {series[0].data.map((d, idx) => {
            const x = getX(idx);
            const isHighlighted = highlightX !== undefined && String(d.x) === String(highlightX);
            return (
              <text
                key={idx}
                x={x}
                y={paddingTop + chartHeight + 18}
                textAnchor="middle"
                fill={isHighlighted ? "#22d3ee" : "#94a3b8"}
                fontSize="10"
                fontFamily="monospace"
                fontWeight={isHighlighted ? "bold" : "normal"}
              >
                {d.x}
              </text>
            );
          })}

          {/* Axis Labels */}
          {xAxisLabel && (
            <text
              x={chartWidth / 2}
              y={height - 4}
              textAnchor="middle"
              fill="#64748b"
              fontSize="10"
              fontFamily="sans-serif"
            >
              {xAxisLabel}
            </text>
          )}
        </svg>
      </div>

      {hoveredPoint && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-2 text-xs flex items-center justify-between font-mono animate-in fade-in duration-150">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: hoveredPoint.color }} />
            <span className="text-muted-foreground">{hoveredPoint.seriesName}</span>
            <span className="text-foreground">({xAxisLabel || "x"} = {hoveredPoint.x}):</span>
          </div>
          <span className="font-bold" style={{ color: hoveredPoint.color }}>
            {formatNumber(hoveredPoint.y, 4)}
          </span>
        </div>
      )}
    </div>
  );
};
