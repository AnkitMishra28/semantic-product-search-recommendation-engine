"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  ArrowDown,
  Search,
  Cpu,
  Layers,
  GitMerge,
  Sparkles,
  UserCheck,
  MessageSquareQuote,
  PackageCheck,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

interface Stage {
  icon: React.ReactNode;
  title: string;
  detail: string;
  stageNum: string;
}

const STAGES: Stage[] = [
  { stageNum: "01", icon: <Search className="w-4 h-4" />, title: "User Query", detail: "Natural language query with intent and soft constraints" },
  { stageNum: "02", icon: <Cpu className="w-4 h-4" />, title: "Query Understanding", detail: "Intent classification, brand/price constraint parsing, entity extraction" },
  { stageNum: "03", icon: <Layers className="w-4 h-4" />, title: "Dense Bi-Encoder Retrieval", detail: "FAISS HNSW vector search over all-MiniLM-L6 embeddings (384d)" },
  { stageNum: "04", icon: <GitMerge className="w-4 h-4" />, title: "BM25 & RRF Fusion (Offline Track)", detail: "Lexical BM25 candidate generation fused with Dense via Reciprocal Rank Fusion" },
  { stageNum: "05", icon: <Sparkles className="w-4 h-4" />, title: "Cross-Encoder Reranking", detail: "ms-marco-MiniLM-L6 cross-attention rescoring over retrieved candidate pool" },
  { stageNum: "06", icon: <UserCheck className="w-4 h-4" />, title: "Multi-Signal Personalization", detail: "Weighted blend of semantic score, star rating, review volume, and historical affinity" },
  { stageNum: "07", icon: <MessageSquareQuote className="w-4 h-4" />, title: "Grounded Explanation", detail: "Evidence-constrained rationale generation with explicit catalog guardrails" },
  { stageNum: "08", icon: <PackageCheck className="w-4 h-4" />, title: "Final Ranked Result", detail: "Explainable, multi-stage scored product recommendations" },
];

interface PipelineVisualizationProps {
  compact?: boolean;
}

export const PipelineVisualization: React.FC<PipelineVisualizationProps> = ({ compact = false }) => {
  return (
    <div className="space-y-6">
      <div className="glass mx-auto max-w-4xl rounded-3xl p-6 sm:p-8 border border-slate-800/80 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
            <p className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold">
              End-to-End Architecture Flow
            </p>
          </div>
          <span className="text-[11px] font-mono text-muted-foreground">
            8 Sequential & Evaluated Stages
          </span>
        </div>

        <div className="flex flex-col items-stretch max-w-2xl mx-auto space-y-2">
          {STAGES.map((stage, idx) => {
            const isOfflineTrack = stage.title.includes("Offline Track");
            return (
              <React.Fragment key={stage.title}>
                <motion.div
                  initial={false}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex items-center gap-4 rounded-2xl border p-4 transition-all ${
                    isOfflineTrack
                      ? "border-amber-500/30 bg-amber-500/5 text-amber-200/90"
                      : "border-slate-800/90 bg-secondary/30 hover:border-cyan-400/40 hover:bg-secondary/50 text-foreground"
                  }`}
                >
                  <span className="font-mono text-xs font-bold text-muted-foreground/60 shrink-0">
                    {stage.stageNum}
                  </span>
                  <div
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border ${
                      isOfflineTrack
                        ? "border-amber-400/40 bg-amber-400/10 text-amber-300"
                        : "border-cyan-400/30 bg-cyan-400/10 text-cyan-300 shadow-[0_0_16px_rgba(34,211,238,0.15)]"
                    }`}
                  >
                    {stage.icon}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-semibold text-foreground truncate">{stage.title}</p>
                      {isOfflineTrack ? (
                        <span className="inline-flex items-center gap-1 rounded-full border border-amber-400/40 bg-amber-400/10 px-2 py-0.5 text-[10px] font-mono text-amber-300 shrink-0">
                          <AlertCircle className="w-2.5 h-2.5" />
                          Offline Track
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-0.5 text-[10px] font-mono text-cyan-300 shrink-0">
                          <CheckCircle2 className="w-2.5 h-2.5" />
                          Live Path
                        </span>
                      )}
                    </div>
                    {!compact && (
                      <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">
                        {stage.detail}
                      </p>
                    )}
                  </div>
                </motion.div>
                {idx < STAGES.length - 1 && (
                  <div className="flex justify-center py-0.5 text-cyan-400/50">
                    <ArrowDown className="w-4 h-4 animate-bounce" style={{ animationDuration: "2s" }} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Distinction Callout */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 max-w-4xl mx-auto">
        <div className="rounded-2xl border border-cyan-400/30 bg-cyan-950/20 p-4 text-xs text-cyan-100 shadow-sm space-y-1.5">
          <div className="flex items-center gap-2 text-cyan-300 font-semibold font-mono uppercase text-[11px]">
            <CheckCircle2 className="w-4 h-4" />
            <span>Live Endpoint Path</span>
          </div>
          <p className="text-muted-foreground leading-relaxed">
            Query Understanding → Dense FAISS (HNSW) Retrieval → Cross-Encoder Reranking → Personalization → Grounded Explanation.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-secondary/30 p-4 text-xs text-muted-foreground shadow-sm space-y-1.5">
          <div className="flex items-center gap-2 text-foreground/90 font-semibold font-mono uppercase text-[11px]">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Offline Benchmark Track</span>
          </div>
          <p className="text-muted-foreground leading-relaxed">
            BM25 + Dense FAISS candidate generation and Reciprocal Rank Fusion (RRF) are benchmarked offline on evaluation datasets.
          </p>
        </div>
      </div>
    </div>
  );
};
