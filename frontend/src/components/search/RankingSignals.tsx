"use client";

import React, { useState } from "react";
import { ChevronDown, Activity } from "lucide-react";
import { RerankSignal, RetrievalSignal } from "@/types";
import { formatNumber } from "@/lib/utils";

interface RankingSignalsProps {
  retrievalSignal?: RetrievalSignal | null;
  rerankSignal?: RerankSignal | null;
  finalScore: number;
  finalRank: number;
}

/**
 * Expandable "ML Signals" panel. Only renders rows for signals the backend
 * actually returned — a search request can run dense-retrieval-only, so
 * rerank_signal may legitimately be absent.
 */
export const RankingSignals: React.FC<RankingSignalsProps> = ({
  retrievalSignal,
  rerankSignal,
  finalScore,
  finalRank,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-xl border border-slate-800 bg-secondary/20 overflow-hidden transition-colors">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition"
      >
        <span className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wide text-cyan-300/90 font-semibold">
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          Multi-Stage ML Signals
        </span>
        <ChevronDown className={`w-3.5 h-3.5 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 px-3 pb-3 pt-1 text-xs font-mono border-t border-slate-800/80 bg-slate-950/40">
          {retrievalSignal && (
            <>
              <dt className="text-muted-foreground">Dense Bi-Encoder Rank</dt>
              <dd className="text-foreground text-right font-semibold">#{retrievalSignal.initial_rank}</dd>
              <dt className="text-muted-foreground">Dense Similarity Score</dt>
              <dd className="text-foreground text-right">{formatNumber(retrievalSignal.initial_score, 4)}</dd>
            </>
          )}
          {rerankSignal && (
            <>
              <dt className="text-muted-foreground">Cross-Encoder Rank</dt>
              <dd className="text-cyan-300 text-right font-semibold">#{rerankSignal.rerank_rank}</dd>
              <dt className="text-muted-foreground">Cross-Encoder Logit</dt>
              <dd className="text-cyan-300 text-right">{formatNumber(rerankSignal.rerank_score, 3)}</dd>
            </>
          )}
          <dt className="text-foreground font-semibold pt-1 border-t border-slate-800">Final Blended Rank</dt>
          <dd className="text-cyan-400 text-right font-bold pt-1 border-t border-slate-800">#{finalRank}</dd>
          <dt className="text-foreground font-semibold">Final Multi-Signal Score</dt>
          <dd className="text-cyan-400 text-right font-bold">{formatNumber(finalScore, 4)}</dd>
          {!retrievalSignal && !rerankSignal && (
            <p className="col-span-2 text-muted-foreground italic text-[11px] pt-1">
              No first-stage retrieval/rerank signal returned for this candidate.
            </p>
          )}
        </dl>
      )}
    </div>
  );
};
