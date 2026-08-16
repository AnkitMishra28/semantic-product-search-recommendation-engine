"use client";

import React from "react";
import { formatPercent } from "@/lib/utils";
import { Database, GitMerge, CheckCircle2, Layers } from "lucide-react";

interface CandidateCoverageProps {
  coverageData?: {
    mean_candidate_pool_size_before_fusion?: number;
    mean_candidate_overlap_size?: number;
    mean_jaccard_similarity?: number;
    relevant_recovery_breakdown?: {
      total_relevant_documents?: number;
      relevant_retrieved_by_both?: { count: number; percentage: number };
      relevant_retrieved_by_bm25_only?: { count: number; percentage: number };
      relevant_retrieved_by_dense_only?: { count: number; percentage: number };
      relevant_missed_by_both?: { count: number; percentage: number };
      total_relevant_in_hybrid_candidate_pool?: { count: number; percentage: number };
    };
  };
  truncatedTop100Recall?: number;
}

export const CandidateCoverageVenn: React.FC<CandidateCoverageProps> = ({
  coverageData,
  truncatedTop100Recall = 0.1958,
}) => {
  const breakdown = coverageData?.relevant_recovery_breakdown;
  const bothPct = breakdown?.relevant_retrieved_by_both?.percentage ?? 12.92;
  const bm25OnlyPct = breakdown?.relevant_retrieved_by_bm25_only?.percentage ?? 5.83;
  const denseOnlyPct = breakdown?.relevant_retrieved_by_dense_only?.percentage ?? 6.67;
  const totalPoolPct = breakdown?.total_relevant_in_hybrid_candidate_pool?.percentage ?? 25.42;

  return (
    <div className="glass-panel rounded-2xl border border-slate-800 p-5 sm:p-6 space-y-6 shadow-xl">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-semibold text-foreground font-sans">
            Candidate Pool Coverage & Dual-Track Recovery Analysis
          </span>
        </div>
        <span className="text-[11px] font-mono text-cyan-300 font-semibold bg-cyan-950/40 border border-cyan-400/30 px-2.5 py-1 rounded-lg">
          Jaccard Overlap: {((coverageData?.mean_jaccard_similarity ?? 0.3083) * 100).toFixed(1)}%
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Visual Venn / Intersection Diagram */}
        <div className="lg:col-span-5 flex flex-col items-center justify-center p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <svg viewBox="0 0 320 200" className="w-full max-w-[280px] h-auto">
            {/* BM25 Circle */}
            <circle
              cx="110"
              cy="100"
              r="75"
              fill="#38bdf8"
              fillOpacity="0.2"
              stroke="#38bdf8"
              strokeWidth="2"
            />
            {/* Dense FAISS Circle */}
            <circle
              cx="210"
              cy="100"
              r="75"
              fill="#22d3ee"
              fillOpacity="0.2"
              stroke="#22d3ee"
              strokeWidth="2"
            />

            {/* BM25 Only Label */}
            <text x="70" y="95" textAnchor="middle" fill="#38bdf8" fontSize="11" fontWeight="bold" fontFamily="monospace">
              BM25
            </text>
            <text x="70" y="112" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">
              +{bm25OnlyPct.toFixed(2)}%
            </text>

            {/* Overlap Label */}
            <text x="160" y="95" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold" fontFamily="monospace">
              Both
            </text>
            <text x="160" y="112" textAnchor="middle" fill="#22d3ee" fontSize="10" fontWeight="bold" fontFamily="monospace">
              {bothPct.toFixed(2)}%
            </text>

            {/* Dense Only Label */}
            <text x="250" y="95" textAnchor="middle" fill="#22d3ee" fontSize="11" fontWeight="bold" fontFamily="monospace">
              Dense
            </text>
            <text x="250" y="112" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">
              +{denseOnlyPct.toFixed(2)}%
            </text>
          </svg>
          <p className="text-[11px] text-muted-foreground font-mono text-center mt-2">
            Mean Candidate Pool Size: {coverageData?.mean_candidate_pool_size_before_fusion ?? 155.43} items
          </p>
        </div>

        {/* Coverage Metrics Breakdown */}
        <div className="lg:col-span-7 space-y-3.5">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3.5 space-y-1">
              <p className="text-[10px] uppercase font-mono text-muted-foreground font-semibold">
                Untruncated Candidate Pool (BM25 ∪ Dense)
              </p>
              <p className="text-2xl font-bold font-mono text-foreground">
                {totalPoolPct.toFixed(2)}%
              </p>
              <p className="text-[11px] text-muted-foreground/80">
                Captures {breakdown?.total_relevant_in_hybrid_candidate_pool?.count ?? 61} of {breakdown?.total_relevant_documents ?? 240} relevant items
              </p>
            </div>

            <div className="rounded-xl border border-cyan-400/30 bg-cyan-950/20 p-3.5 space-y-1">
              <p className="text-[10px] uppercase font-mono text-cyan-300 font-semibold">
                Truncated Hybrid RRF Top-100
              </p>
              <p className="text-2xl font-bold font-mono text-cyan-300">
                {formatPercent(truncatedTop100Recall, 2)}
              </p>
              <p className="text-[11px] text-cyan-200/80">
                47 of 240 relevant items preserved after Top-100 truncation
              </p>
            </div>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-slate-800/80 bg-secondary/20">
              <span className="flex items-center gap-2 text-muted-foreground font-mono text-[11px]">
                <Layers className="w-3.5 h-3.5 text-sky-400" />
                Lexical BM25 Unique Contribution
              </span>
              <span className="font-mono font-semibold text-sky-300">
                +{bm25OnlyPct.toFixed(2)}% ({breakdown?.relevant_retrieved_by_bm25_only?.count ?? 14} items)
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-lg border border-slate-800/80 bg-secondary/20">
              <span className="flex items-center gap-2 text-muted-foreground font-mono text-[11px]">
                <GitMerge className="w-3.5 h-3.5 text-cyan-400" />
                Dense Bi-Encoder Unique Contribution
              </span>
              <span className="font-mono font-semibold text-cyan-300">
                +{denseOnlyPct.toFixed(2)}% ({breakdown?.relevant_retrieved_by_dense_only?.count ?? 16} items)
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-lg border border-slate-800/80 bg-secondary/20">
              <span className="flex items-center gap-2 text-muted-foreground font-mono text-[11px]">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Intersection / Agreed Candidates
              </span>
              <span className="font-mono font-semibold text-emerald-300">
                {bothPct.toFixed(2)}% ({breakdown?.relevant_retrieved_by_both?.count ?? 31} items)
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
