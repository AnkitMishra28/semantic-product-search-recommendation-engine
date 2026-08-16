"use client";

import React from "react";
import { X, Sparkles, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { WhyThisResult } from "@/components/search/WhyThisResult";
import { GroundedExplanation, Product } from "@/types";

interface ExplanationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  product: Product | null;
  query?: string;
  groundedExplanation?: GroundedExplanation | null;
  isLoading?: boolean;
  onRequestExplanation?: () => void;
}

export const ExplanationDrawer: React.FC<ExplanationDrawerProps> = ({
  isOpen,
  onClose,
  product,
  query,
  groundedExplanation,
  isLoading = false,
  onRequestExplanation,
}) => {
  if (!isOpen || !product) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Why this result explanation"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4 animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="glass-elevated relative w-full max-w-lg rounded-3xl p-6 sm:p-7 shadow-2xl space-y-5 max-h-[85vh] overflow-y-auto border border-slate-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-800 pb-3.5">
          <div className="flex items-center gap-2.5 text-foreground font-semibold">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="text-sm font-sans tracking-tight">Explainable Retrieval & Ranking Rationale</span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close explanation"
            className="rounded-xl p-1.5 text-muted-foreground hover:bg-secondary/60 hover:text-foreground transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4 text-xs">
          {query && (
            <div className="rounded-xl bg-secondary/30 p-3 border border-slate-800/80">
              <span className="text-[11px] font-mono text-cyan-300 uppercase font-semibold">Search Query:</span>
              <p className="font-mono text-foreground mt-1">&ldquo;{query}&rdquo;</p>
            </div>
          )}

          <div className="rounded-xl bg-secondary/20 p-3.5 border border-slate-800/80 space-y-1">
            <span className="text-[11px] font-mono text-cyan-300 uppercase font-semibold">Target Product:</span>
            <p className="font-semibold text-foreground text-sm leading-snug">{product.title}</p>
            <p className="text-muted-foreground font-mono text-xs">ASIN: {product.asin}</p>
          </div>

          {isLoading && (
            <div className="rounded-2xl border border-slate-800 bg-secondary/30 p-6 text-center text-muted-foreground space-y-2">
              <span className="animate-spin rounded-full inline-block h-5 w-5 border-2 border-cyan-400 border-t-transparent" />
              <p className="text-xs">Generating grounded explanation from verified product metadata...</p>
            </div>
          )}

          {!isLoading && groundedExplanation && <WhyThisResult explanation={groundedExplanation} />}

          {!isLoading && !groundedExplanation && (
            <div className="rounded-2xl border border-dashed border-slate-800 p-6 text-center space-y-3">
              <HelpCircle className="w-6 h-6 mx-auto text-muted-foreground" />
              <p className="text-muted-foreground text-xs">No grounded explanation is currently cached for this result.</p>
              {onRequestExplanation && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRequestExplanation}
                  className="rounded-xl border-slate-800 hover:border-cyan-400/40 text-xs"
                >
                  Generate Explanation
                </Button>
              )}
            </div>
          )}
        </div>

        <div className="pt-3 border-t border-slate-800 flex justify-end">
          <Button
            variant="secondary"
            size="sm"
            onClick={onClose}
            className="rounded-xl px-5 h-9 bg-secondary/60 hover:bg-secondary text-xs"
          >
            Close
          </Button>
        </div>
      </div>
    </div>
  );
};
