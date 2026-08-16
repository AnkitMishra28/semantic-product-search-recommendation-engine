"use client";

import React from "react";
import { Sparkles, HelpCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency, formatNumber, humanizeLabel } from "@/lib/utils";
import { RecommendationItem } from "@/types";

interface RecommendationCardProps {
  rec: RecommendationItem;
  onSelectProduct?: (asin: string) => void;
  onExplain?: (asin: string) => void;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({ rec, onSelectProduct, onExplain }) => {
  const summary = rec.grounded_explanation?.summary || rec.explanation?.summary;
  const signalEntries = Object.entries(rec.signals || {}).filter(([, v]) => v !== undefined && v !== null);

  return (
    <Card className="surface-ring glass-panel cursor-default rounded-2xl border border-slate-800 transition-all duration-300 hover:border-cyan-400/40 hover:-translate-y-1 hover:shadow-2xl flex flex-col justify-between overflow-hidden">
      <CardHeader className="p-4 pb-2 space-y-2">
        <div className="flex items-center justify-between">
          <Badge className="text-[10px] font-mono border border-cyan-400/30 bg-cyan-400/10 text-cyan-300 uppercase px-2 py-0.5">
            {rec.recommendation_type.replace(/_/g, " ")}
          </Badge>
          <span className="text-xs font-mono text-muted-foreground bg-secondary/50 border border-slate-800 px-2 py-0.5 rounded-md">
            Score: {formatNumber(rec.score, 3)}
          </span>
        </div>
        <CardTitle
          className="text-xs sm:text-sm font-semibold line-clamp-2 cursor-pointer hover:text-cyan-300 transition-colors text-foreground/95 leading-snug"
          onClick={() => onSelectProduct?.(rec.product.asin)}
        >
          {rec.product.title}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-4 pt-1 space-y-3 flex-1">
        <div className="flex items-center justify-between text-xs border-t border-slate-800/80 pt-2">
          <span className="text-muted-foreground truncate max-w-[150px]">
            {rec.product.brand || rec.product.categories[0] || "—"}
          </span>
          <span className="font-bold text-foreground font-mono">{formatCurrency(rec.product.price)}</span>
        </div>
        <p className="text-[11px] font-mono text-muted-foreground/80">ASIN: {rec.product.asin}</p>

        {summary && (
          <p className="text-xs text-cyan-100 line-clamp-2 rounded-xl border border-cyan-400/25 bg-cyan-950/20 p-2.5 leading-relaxed">
            {summary}
          </p>
        )}

        {signalEntries.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {signalEntries.map(([key, value]) => (
              <span
                key={key}
                className="text-[10px] font-mono rounded-md border border-slate-800 bg-secondary/40 px-2 py-0.5 text-muted-foreground"
              >
                {humanizeLabel(key)}: {formatNumber(value, 2)}
              </span>
            ))}
          </div>
        )}
      </CardContent>

      <CardFooter className="p-4 pt-2 border-t border-slate-800/80 flex items-center justify-between bg-secondary/20">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onExplain?.(rec.product.asin)}
          className="text-xs text-muted-foreground hover:text-cyan-200 hover:bg-cyan-400/10 gap-1.5 px-2.5 h-7 rounded-lg"
        >
          <HelpCircle className="w-3.5 h-3.5 text-cyan-300" />
          <span>Why Recommended</span>
        </Button>
        {rec.grounded_explanation?.grounded && <Sparkles className="w-3.5 h-3.5 text-cyan-300" />}
      </CardFooter>
    </Card>
  );
};
