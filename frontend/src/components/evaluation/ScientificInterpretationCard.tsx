"use client";

import React from "react";
import { Info, CheckCircle2, ShieldCheck, AlertCircle } from "lucide-react";

interface ScientificInterpretationProps {
  title?: string;
  measured: string;
  finding: string;
  conclusion: string;
  scopeNote?: string;
}

export const ScientificInterpretationCard: React.FC<ScientificInterpretationProps> = ({
  title = "Scientific Interpretation",
  measured,
  finding,
  conclusion,
  scopeNote,
}) => {
  return (
    <div className="rounded-2xl border border-cyan-400/25 bg-gradient-to-br from-cyan-950/20 via-slate-950 to-slate-950 p-5 space-y-3 shadow-xl">
      <div className="flex items-center justify-between border-b border-cyan-400/20 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-cyan-400/15 border border-cyan-400/30 text-cyan-300">
            <Info className="w-3.5 h-3.5" />
          </div>
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-300">
            {title}
          </span>
        </div>
        <span className="text-[10px] font-mono text-muted-foreground uppercase bg-secondary/40 border border-slate-800 px-2 py-0.5 rounded">
          Empirical Finding
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
        <div className="space-y-1 bg-secondary/20 rounded-xl p-3 border border-slate-800/70">
          <p className="font-mono text-[10px] uppercase text-muted-foreground font-semibold flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
            1. What Was Measured
          </p>
          <p className="text-foreground/90 leading-relaxed font-sans">{measured}</p>
        </div>

        <div className="space-y-1 bg-secondary/20 rounded-xl p-3 border border-slate-800/70">
          <p className="font-mono text-[10px] uppercase text-cyan-300 font-semibold flex items-center gap-1.5">
            <CheckCircle2 className="w-3 h-3 text-cyan-400" />
            2. Empirical Finding
          </p>
          <p className="text-foreground/90 leading-relaxed font-sans">{finding}</p>
        </div>

        <div className="space-y-1 bg-cyan-950/30 rounded-xl p-3 border border-cyan-400/20">
          <p className="font-mono text-[10px] uppercase text-cyan-200 font-semibold flex items-center gap-1.5">
            <ShieldCheck className="w-3 h-3 text-cyan-300" />
            3. Applied Conclusion
          </p>
          <p className="text-cyan-100/90 leading-relaxed font-sans">{conclusion}</p>
        </div>
      </div>

      {scopeNote && (
        <div className="flex items-start gap-2 pt-1 text-[11px] text-muted-foreground border-t border-slate-800/60">
          <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-400/80" />
          <p className="leading-snug">
            <span className="font-semibold text-foreground/80">Benchmark Scope & Guardrail: </span>
            {scopeNote}
          </p>
        </div>
      )}
    </div>
  );
};
