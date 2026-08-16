"use client";

import React from "react";
import { Clock, Activity, Zap, CheckCircle } from "lucide-react";
import { PipelineStageTiming } from "@/types";

interface PipelineInspectorProps {
  timings: PipelineStageTiming;
  totalRetrieved: number;
  totalReturned: number;
}

export const PipelineInspector: React.FC<PipelineInspectorProps> = ({
  timings,
  totalRetrieved,
  totalReturned,
}) => {
  const stages = [
    {
      name: "Stage 1: Query Understanding",
      ms: timings.query_understanding_ms,
      desc: "Entity normalization & intent parsing",
      count: "1 query",
    },
    {
      name: "Stage 2: FAISS Dense Retrieval",
      ms: timings.dense_retrieval_ms,
      desc: "all-MiniLM-L6 vector ANN search",
      count: `${totalRetrieved} retrieved`,
    },
    {
      name: "Stage 3: Cross-Encoder Rerank",
      ms: timings.cross_encoder_rerank_ms,
      desc: "ms-marco cross-attention rescore",
      count: `${totalReturned} reranked`,
    },
    {
      name: "Stage 4: Personalization & Explanations",
      ms: timings.business_ranking_ms + timings.explanation_generation_ms,
      desc: "Business rules & grounded rationale",
      count: "Final rank",
    },
  ];

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 shadow-xl space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-3 gap-2">
        <div className="flex items-center gap-2.5 text-foreground font-semibold">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
            <Activity className="w-4 h-4" />
          </div>
          <span className="text-sm font-sans tracking-tight">Multi-Stage Latency & Execution Profile</span>
        </div>
        <div className="flex items-center gap-2 font-mono text-cyan-300 font-semibold bg-cyan-400/10 border border-cyan-400/30 px-3 py-1 rounded-xl text-xs shadow-sm">
          <Clock className="w-3.5 h-3.5" />
          <span>Total Latency: {timings.total_latency_ms.toFixed(1)} ms</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {stages.map((stage) => (
          <div
            key={stage.name}
            className="rounded-xl border border-slate-800/90 bg-secondary/25 p-3.5 space-y-2 hover:border-slate-700 transition-colors"
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold text-xs text-foreground/90">{stage.name}</span>
              <span className="font-mono text-cyan-300 font-bold text-xs">{stage.ms.toFixed(1)} ms</span>
            </div>
            <p className="text-muted-foreground text-[11px] leading-snug">{stage.desc}</p>
            <div className="pt-1.5 flex items-center justify-between text-[11px] text-muted-foreground/90 font-mono border-t border-slate-800/60">
              <span>{stage.count}</span>
              <div className="flex items-center gap-1 text-emerald-400 text-[10px]">
                <CheckCircle className="w-3.5 h-3.5" />
                <span>Passed</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
