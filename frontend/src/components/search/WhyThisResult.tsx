"use client";

import React from "react";
import { CheckCircle2, AlertTriangle, Info, ShieldCheck, Bot, Cpu } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { GroundedExplanation } from "@/types";
import { formatPercent } from "@/lib/utils";

interface WhyThisResultProps {
  explanation: GroundedExplanation;
}

/**
 * Renders the structured, evidence-grounded explanation produced by the
 * Phase 10 GroundedExplainer. Every claim shown here is either a verified
 * `is_matched` reason (backed by `evidence`) or an explicit warning about
 * missing catalog evidence — never a fabricated attribute.
 */
export const WhyThisResult: React.FC<WhyThisResultProps> = ({ explanation }) => {
  const { summary, reasons, warnings, semantic_match_score, generation_method, grounded } = explanation;

  return (
    <div className="space-y-4 text-xs">
      {/* Primary Summary Box */}
      <div className="rounded-2xl border border-cyan-400/30 bg-cyan-950/25 p-4 space-y-2.5 shadow-sm">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 font-semibold text-cyan-300 font-sans">
            <Info className="w-4 h-4 text-cyan-400" />
            <span>Grounded ML Explanation</span>
          </div>
          <div className="flex items-center gap-1.5">
            {grounded && (
              <Badge className="border-cyan-400/30 bg-cyan-400/15 text-cyan-200 text-[10px] gap-1 px-2 py-0.5">
                <ShieldCheck className="w-3.5 h-3.5" /> Grounded
              </Badge>
            )}
            <Badge variant="outline" className="border-slate-800 bg-secondary/50 text-[10px] gap-1 font-mono">
              {generation_method === "llm" ? <Bot className="w-3 h-3 text-cyan-300" /> : <Cpu className="w-3 h-3 text-cyan-300" />}
              {generation_method === "llm" ? "LLM Layer" : "Deterministic"}
            </Badge>
          </div>
        </div>

        <p className="text-foreground/95 leading-relaxed font-sans">{summary}</p>

        {semantic_match_score !== undefined && semantic_match_score !== null && (
          <div className="flex items-center gap-3 pt-1 border-t border-cyan-400/20">
            <span className="text-[10px] text-muted-foreground uppercase font-mono">Semantic Alignment</span>
            <div className="h-2 flex-1 rounded-full bg-slate-950 overflow-hidden max-w-[160px] border border-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-blue-500"
                style={{ width: `${Math.min(100, Math.max(0, semantic_match_score * 100))}%` }}
              />
            </div>
            <span className="text-[11px] font-mono text-cyan-300 font-bold">
              {formatPercent(semantic_match_score, 1)}
            </span>
          </div>
        )}
      </div>

      {/* Verified Feature Reasons */}
      {reasons.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] font-mono uppercase tracking-wider text-muted-foreground font-semibold">
            Verified Feature Alignments
          </p>
          <ul className="space-y-2">
            {reasons.map((reason, idx) => (
              <li
                key={idx}
                className="flex items-start gap-2.5 rounded-xl border border-slate-800/90 bg-secondary/30 p-3"
              >
                {reason.is_matched ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                )}
                <div className="min-w-0 space-y-0.5 flex-1">
                  <p className="font-semibold text-foreground/95">
                    {reason.label}:
                    <span className="ml-1.5 text-muted-foreground font-normal">{reason.text}</span>
                  </p>
                  {reason.evidence && (
                    <p className="text-muted-foreground/85 italic text-[11px] pt-0.5">
                      Evidence: &ldquo;{reason.evidence}&rdquo;
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Hallucination Guardrail Warnings */}
      {warnings.length > 0 && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-3.5 space-y-2">
          <div className="flex items-center gap-2 text-amber-400 font-semibold font-mono text-[11px] uppercase">
            <AlertTriangle className="w-4 h-4" />
            <span>Metadata Guardrail Warnings</span>
          </div>
          <ul className="space-y-1 text-muted-foreground/90 pl-1">
            {warnings.map((w, idx) => (
              <li key={idx} className="flex items-start gap-1.5">
                <span className="text-amber-400 font-bold">•</span>
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
