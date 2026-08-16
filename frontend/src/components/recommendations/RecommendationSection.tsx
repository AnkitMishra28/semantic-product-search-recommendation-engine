"use client";

import React from "react";
import { Sparkles } from "lucide-react";
import { RecommendationItem } from "@/types";
import { RecommendationCard } from "./RecommendationCard";

interface RecommendationSectionProps {
  recommendations: RecommendationItem[];
  anchorTitle?: string;
  title?: string;
  onSelectProduct?: (asin: string) => void;
  onExplain?: (asin: string) => void;
}

export const RecommendationSection: React.FC<RecommendationSectionProps> = ({
  recommendations,
  anchorTitle,
  title = "Frequently Co-Purchased & Complementary Items",
  onSelectProduct,
  onExplain,
}) => {
  if (recommendations.length === 0) return null;

  return (
    <div className="space-y-4 pt-6">
      <div className="flex items-center justify-between border-b border-border/80 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-cyan-200" />
          <h2 className="text-base font-semibold text-foreground">{title}</h2>
        </div>
        {anchorTitle && (
          <span className="text-xs text-muted-foreground line-clamp-1 max-w-sm font-mono">
            Anchor: {anchorTitle}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {recommendations.map((rec) => (
          <RecommendationCard
            key={rec.product.asin}
            rec={rec}
            onSelectProduct={onSelectProduct}
            onExplain={onExplain}
          />
        ))}
      </div>
    </div>
  );
};
