"use client";

import React from "react";
import { Cpu, Tag, Layers, ArrowRight, Gauge } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { QueryUnderstandingResult } from "@/types";
import { formatCurrencyCode, humanizeLabel } from "@/lib/utils";

interface QueryUnderstandingBadgeProps {
  data: QueryUnderstandingResult;
}

/**
 * Visualizes the structured output of the Query Understanding pipeline.
 * Every chip below maps directly to a field the backend actually returns
 * (category, brand, price_min/max, attributes, detected_entities) — no
 * attribute is ever invented client-side.
 */
export const QueryUnderstandingBadge: React.FC<QueryUnderstandingBadgeProps> = ({ data }) => {
  const modifiers = data.detected_entities?.modifiers || [];
  const attributeEntries = Object.entries(data.attributes || {});
  const hasPriceConstraint = data.price_min != null || data.price_max != null;
  const currency = data.currency || "USD";

  const priceLabel = hasPriceConstraint
    ? data.price_min != null && data.price_max != null
      ? `${formatCurrencyCode(data.price_min, currency)} – ${formatCurrencyCode(data.price_max, currency)}`
      : data.price_max != null
        ? `≤ ${formatCurrencyCode(data.price_max, currency)}`
        : `≥ ${formatCurrencyCode(data.price_min as number, currency)}`
    : null;

  const hasAnyConstraint = data.category || data.brand || priceLabel || attributeEntries.length > 0;

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 shadow-xl space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-3 gap-2">
        <div className="flex items-center gap-2.5 text-foreground font-semibold">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
            <Cpu className="w-4 h-4" />
          </div>
          <span className="text-sm font-sans tracking-tight">Stage 1 · Query Understanding Analysis</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {data.confidence !== undefined && (
            <Badge variant="outline" className="font-mono text-[11px] gap-1.5 border-slate-800 bg-secondary/50 text-cyan-300">
              <Gauge className="w-3.5 h-3.5 text-cyan-400" />
              Confidence: {(data.confidence * 100).toFixed(0)}%
            </Badge>
          )}
          <Badge variant="outline" className="font-mono text-[11px] border-slate-800 bg-secondary/50 text-foreground/90">
            Intent: <span className="text-cyan-300 font-semibold uppercase ml-1">{data.intent}</span>
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        {/* Extracted Constraints */}
        <div className="space-y-2 rounded-xl bg-secondary/20 p-3.5 border border-slate-800/80">
          <span className="text-muted-foreground font-medium flex items-center gap-1.5 text-xs font-mono uppercase text-cyan-300">
            <Tag className="w-3.5 h-3.5" /> Structured Entities
          </span>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {data.category && (
              <Badge variant="secondary" className="bg-secondary/60 text-foreground border-slate-700">
                Category: {humanizeLabel(data.category)}
              </Badge>
            )}
            {data.brand && (
              <Badge className="border-cyan-400/30 bg-cyan-400/15 text-cyan-200">
                Brand: {data.brand}
              </Badge>
            )}
            {priceLabel && (
              <Badge variant="outline" className="border-amber-400/30 bg-amber-400/10 text-amber-200">
                Budget: {priceLabel}
              </Badge>
            )}
            {attributeEntries.map(([attrKey, values]) =>
              values.map((v) => (
                <Badge key={`${attrKey}-${v}`} variant="outline" className="border-slate-800 text-foreground/80">
                  {humanizeLabel(attrKey)}: {v.toUpperCase()}
                </Badge>
              ))
            )}
            {!hasAnyConstraint && (
              <span className="text-muted-foreground italic text-xs">No hard metadata constraints extracted</span>
            )}
          </div>
        </div>

        {/* Normalized Query */}
        <div className="space-y-2 rounded-xl bg-secondary/20 p-3.5 border border-slate-800/80">
          <span className="text-muted-foreground font-medium flex items-center gap-1.5 text-xs font-mono uppercase text-cyan-300">
            <Layers className="w-3.5 h-3.5" /> Canonical Normalized Query
          </span>
          <p className="font-mono text-cyan-100 bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-2 text-xs">
            {data.normalized_query}
          </p>
          {modifiers.length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {modifiers.map((m) => (
                <span key={m} className="font-mono text-[10px] rounded border border-slate-800 bg-secondary/40 px-2 py-0.5 text-muted-foreground">
                  +{m}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Expansions */}
        <div className="space-y-2 rounded-xl bg-secondary/20 p-3.5 border border-slate-800/80">
          <span className="text-muted-foreground font-medium flex items-center gap-1.5 text-xs font-mono uppercase text-cyan-300">
            <ArrowRight className="w-3.5 h-3.5" /> Semantic Synonyms / Expansions
          </span>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {data.expanded_queries && data.expanded_queries.length > 0 ? (
              data.expanded_queries.map((eq) => (
                <span
                  key={eq}
                  className="bg-slate-950/60 border border-slate-800 rounded-lg px-2.5 py-1 text-foreground/90 font-mono text-[11px]"
                >
                  &ldquo;{eq}&rdquo;
                </span>
              ))
            ) : (
              <span className="text-muted-foreground italic text-xs">Dense vector lookup (all-MiniLM-L6)</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
