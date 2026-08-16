"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Star, HelpCircle, ArrowUpRight, Sparkles, ImageOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { RankingSignals } from "@/components/search/RankingSignals";
import { formatCount, formatCurrency } from "@/lib/utils";
import { SearchResultItem } from "@/types";

interface ProductCardProps {
  item: SearchResultItem;
  rankIndex: number;
  onSelectProduct?: (asin: string) => void;
  onExplain?: (asin: string) => void;
}

export const ProductCard: React.FC<ProductCardProps> = ({
  item,
  rankIndex,
  onSelectProduct,
  onExplain,
}) => {
  const { product, final_score, retrieval_signal, rerank_signal, explanation } = item;
  const [imageError, setImageError] = useState(false);

  return (
    <Card className="surface-ring glass-panel flex flex-col justify-between overflow-hidden rounded-2xl border border-slate-800/90 transition-all duration-300 hover:border-cyan-400/40 hover:-translate-y-1 hover:shadow-2xl group">
      {/* Product Image Section */}
      {product.image_url && !imageError ? (
        <div className="relative h-44 w-full overflow-hidden rounded-t-2xl bg-gradient-to-b from-slate-900/60 to-slate-950/80 p-3">
          <Image
            src={product.image_url}
            alt={product.title}
            fill
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            className="object-contain p-2 transition-transform duration-500 group-hover:scale-105"
            onError={() => setImageError(true)}
            unoptimized
          />
        </div>
      ) : (
        <div className="flex h-44 w-full items-center justify-center rounded-t-2xl bg-secondary/30 text-muted-foreground/40">
          <ImageOff className="w-8 h-8" />
        </div>
      )}

      {/* Card Header & Title */}
      <CardHeader className="p-5 pb-2 space-y-2.5">
        <div className="flex items-center justify-between gap-2">
          <Badge className="font-mono text-[10px] border border-cyan-400/30 bg-cyan-400/10 text-cyan-300 font-semibold px-2 py-0.5">
            Rank #{rankIndex + 1}
          </Badge>
          <Badge variant="outline" className="text-[10px] font-mono border-slate-800 bg-secondary/50 text-foreground/80">
            Score: {final_score.toFixed(4)}
          </Badge>
        </div>

        <CardTitle
          className="text-sm font-semibold line-clamp-2 text-foreground/95 hover:text-cyan-300 transition-colors cursor-pointer leading-snug"
          onClick={() => onSelectProduct?.(product.asin)}
        >
          {product.title}
        </CardTitle>

        <div className="flex items-center justify-between text-xs text-muted-foreground gap-2 pt-0.5">
          {product.brand ? (
            <span className="font-medium text-foreground/80 truncate">
              Brand: <span className="text-cyan-300/90">{product.brand}</span>
            </span>
          ) : (
            <span />
          )}
          <span className="font-mono text-[11px] text-muted-foreground/80 shrink-0">
            ASIN: {product.asin}
          </span>
        </div>
      </CardHeader>

      {/* Card Content & Features */}
      <CardContent className="p-5 pt-1 space-y-3.5 flex-1">
        <div className="flex items-center justify-between border-t border-slate-800/80 pt-2.5">
          <div className="flex items-center gap-1.5 text-amber-400 font-medium text-xs">
            <Star className="w-3.5 h-3.5 fill-current" />
            <span>{product.average_rating ? product.average_rating.toFixed(1) : "N/A"}</span>
            <span className="text-muted-foreground font-normal">
              ({formatCount(product.rating_count)})
            </span>
          </div>
          <span className="text-base font-bold text-foreground font-mono">
            {formatCurrency(product.price)}
          </span>
        </div>

        {product.features && product.features.length > 0 && (
          <ul className="space-y-1.5 text-xs text-muted-foreground">
            {product.features.slice(0, 2).map((feat, idx) => (
              <li key={idx} className="line-clamp-1 flex items-start gap-1.5">
                <span className="text-cyan-400 font-bold shrink-0">•</span>
                <span className="truncate">{feat}</span>
              </li>
            ))}
          </ul>
        )}

        {explanation && (
          <div className="rounded-xl border border-cyan-400/25 bg-cyan-950/20 p-2.5 text-xs text-cyan-100 flex items-start gap-2">
            <Sparkles className="w-4 h-4 text-cyan-300 shrink-0 mt-0.5" />
            <p className="line-clamp-2 leading-relaxed">{explanation}</p>
          </div>
        )}

        <RankingSignals
          retrievalSignal={retrieval_signal}
          rerankSignal={rerank_signal}
          finalScore={final_score}
          finalRank={rankIndex + 1}
        />
      </CardContent>

      {/* Card Footer Actions */}
      <CardFooter className="p-5 pt-2 border-t border-slate-800/80 flex items-center justify-between gap-2 bg-secondary/15">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onExplain?.(product.asin)}
          className="text-xs text-muted-foreground hover:text-cyan-200 hover:bg-cyan-400/10 gap-1.5 px-2.5 h-8 rounded-lg"
        >
          <HelpCircle className="w-3.5 h-3.5 text-cyan-300" />
          <span>Why This Result</span>
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => onSelectProduct?.(product.asin)}
          className="text-xs gap-1.5 px-3 h-8 rounded-lg border-slate-800 hover:border-cyan-400/40 hover:bg-secondary/70"
        >
          <span>Find Similar</span>
          <ArrowUpRight className="w-3.5 h-3.5 text-cyan-300" />
        </Button>
      </CardFooter>
    </Card>
  );
};
