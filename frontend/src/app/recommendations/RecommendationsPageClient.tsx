"use client";

import React, { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Sparkles, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { RecommendationCard } from "@/components/recommendations/RecommendationCard";
import { ExplanationDrawer } from "@/components/recommendations/ExplanationDrawer";
import { ErrorState } from "@/components/shared/ErrorState";
import { MetricGridSkeleton } from "@/components/shared/LoadingSkeleton";
import { apiClient } from "@/lib/api";
import { GroundedExplanation, Product, RecommendationStrategy, RecommendResponse } from "@/types";

type Mode = "item" | "user";

const STRATEGIES: { value: RecommendationStrategy; label: string }[] = [
  { value: "hybrid", label: "Hybrid" },
  { value: "hybrid_mmr", label: "Hybrid + MMR Diversity" },
  { value: "popularity", label: "Popularity" },
  { value: "content", label: "Content-Based" },
  { value: "collaborative", label: "Collaborative" },
];

const SAMPLE_ASINS = [
  { asin: "B0BW4PFM58", name: "Anker USB-C Hub" },
  { asin: "B075X8471B", name: "Bose QuietComfort 35" },
  { asin: "B0BJVTD6VK", name: "Magnetic Car Mount" },
  { asin: "B0C1LBWM4Q", name: "Scosche MagicMount" },
];

export default function RecommendationsPageClient() {
  const searchParams = useSearchParams();
  const prefillAsin = searchParams.get("asin") || "";

  const [mode, setMode] = useState<Mode>("item");
  const [asin, setAsin] = useState(prefillAsin);
  const [historyInput, setHistoryInput] = useState("");
  const [topK, setTopK] = useState(8);
  const [strategy, setStrategy] = useState<RecommendationStrategy>("hybrid");
  const [lambdaDiversity, setLambdaDiversity] = useState(0.7);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [response, setResponse] = useState<RecommendResponse | null>(null);

  const [explanationModal, setExplanationModal] = useState<{
    isOpen: boolean;
    product: Product | null;
    groundedExplanation: GroundedExplanation | null;
  }>({ isOpen: false, product: null, groundedExplanation: null });

  const runRecommend = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const req =
        mode === "item"
          ? {
              asin: asin.trim(),
              top_k: topK,
              strategy,
              lambda_diversity: strategy === "hybrid_mmr" ? lambdaDiversity : undefined,
              generate_explanations: true,
            }
          : {
              user_history_asins: historyInput
                .split(",")
                .map((a) => a.trim())
                .filter(Boolean),
              top_k: topK,
              strategy,
              lambda_diversity: strategy === "hybrid_mmr" ? lambdaDiversity : undefined,
              generate_explanations: true,
            };
      const resp = await apiClient.recommend(req);
      setResponse(resp);
    } catch (err) {
      setResponse(null);
      setError(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (prefillAsin) {
      setAsin(prefillAsin);
      setMode("item");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillAsin]);

  useEffect(() => {
    if (prefillAsin) {
      runRecommend();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillAsin]);

  const handleExplain = (productAsin: string) => {
    const rec = response?.recommendations.find((r) => r.product.asin === productAsin);
    if (!rec) return;
    setExplanationModal({
      isOpen: true,
      product: rec.product,
      groundedExplanation: rec.grounded_explanation || null,
    });
  };

  const canSubmit = mode === "item" ? asin.trim().length > 0 : historyInput.trim().length > 0;

  return (
    <div className="space-y-10">
      {/* Header Banner */}
      <section className="glass-panel rounded-3xl p-6 sm:p-8 border border-slate-800 shadow-2xl space-y-2">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold">
            Recommendation Engine Lab
          </span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          Recommendation Console
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Interactive evaluation interface for Item-to-Item graphs, User History aggregation, Collaborative Filtering, and Maximal Marginal Relevance (MMR) diversity reranking on real Amazon catalog data.
        </p>
      </section>

      {/* Control Console */}
      <Card className="glass-panel rounded-3xl border border-slate-800 shadow-2xl overflow-hidden">
        <CardContent className="p-6 sm:p-8 space-y-6">
          {/* Mode Switcher */}
          <div className="inline-flex rounded-2xl border border-slate-800 bg-secondary/40 p-1.5 shadow-inner">
            <button
              type="button"
              onClick={() => setMode("item")}
              className={`rounded-xl px-5 py-2 text-xs font-semibold transition-all ${
                mode === "item"
                  ? "border border-cyan-400/40 bg-gradient-to-r from-cyan-400/20 to-blue-500/20 text-cyan-200 shadow-[0_0_20px_rgba(34,211,238,0.2)]"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              By Product (ASIN)
            </button>
            <button
              type="button"
              onClick={() => setMode("user")}
              className={`rounded-xl px-5 py-2 text-xs font-semibold transition-all ${
                mode === "user"
                  ? "border border-cyan-400/40 bg-gradient-to-r from-cyan-400/20 to-blue-500/20 text-cyan-200 shadow-[0_0_20px_rgba(34,211,238,0.2)]"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              By User History
            </button>
          </div>

          {mode === "item" ? (
            <div className="space-y-2">
              <label htmlFor="asin-input" className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold">
                Anchor Product ASIN
              </label>
              <Input
                id="asin-input"
                value={asin}
                onChange={(e) => setAsin(e.target.value)}
                placeholder="e.g. B0BW4PFM58"
                className="h-12 bg-secondary/30 border-slate-800 text-sm font-mono focus-visible:ring-cyan-400 rounded-xl"
              />
              <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-muted-foreground">
                <span className="font-mono text-[10px] text-muted-foreground uppercase">Sample ASINs:</span>
                {SAMPLE_ASINS.map((sample) => (
                  <button
                    key={sample.asin}
                    type="button"
                    onClick={() => {
                      setAsin(sample.asin);
                    }}
                    className="rounded-lg border border-slate-800 bg-secondary/30 px-2 py-0.5 font-mono text-[11px] text-muted-foreground hover:border-cyan-400/40 hover:text-cyan-200"
                  >
                    {sample.asin} ({sample.name})
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <label htmlFor="history-input" className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold">
                Recent User History ASINs (comma-separated)
              </label>
              <Input
                id="history-input"
                value={historyInput}
                onChange={(e) => setHistoryInput(e.target.value)}
                placeholder="e.g. B0BW4PFM58, B075X8471B, B0BJVTD6VK"
                className="h-12 bg-secondary/30 border-slate-800 text-sm font-mono focus-visible:ring-cyan-400 rounded-xl"
              />
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
            <div className="space-y-1.5">
              <label htmlFor="topk-input" className="text-xs font-mono uppercase text-muted-foreground font-medium">
                Top Candidates (K)
              </label>
              <Input
                id="topk-input"
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 1)}
                className="h-11 bg-secondary/30 border-slate-800 font-mono text-sm rounded-xl"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="strategy-select" className="text-xs font-mono uppercase text-muted-foreground font-medium">
                Recommendation Strategy
              </label>
              <select
                id="strategy-select"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as RecommendationStrategy)}
                className="flex h-11 w-full rounded-xl border border-slate-800 bg-secondary/40 px-3 py-2 text-xs font-sans text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-cyan-400"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.value} value={s.value} className="bg-slate-900 text-foreground">
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            {strategy === "hybrid_mmr" && (
              <div className="space-y-1.5">
                <label htmlFor="lambda-input" className="text-xs font-mono uppercase text-muted-foreground font-medium">
                  MMR λ Diversity Weight ({lambdaDiversity})
                </label>
                <Input
                  id="lambda-input"
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={lambdaDiversity}
                  onChange={(e) => setLambdaDiversity(Number(e.target.value))}
                  className="h-11 bg-secondary/30 border-slate-800 font-mono text-sm rounded-xl"
                />
              </div>
            )}
          </div>

          <div className="pt-2">
            <Button
              onClick={runRecommend}
              disabled={!canSubmit || isLoading}
              className="gap-2 px-6 h-11 rounded-xl bg-gradient-to-r from-cyan-400 to-blue-500 hover:from-cyan-300 hover:to-blue-400 text-slate-950 font-semibold shadow-[0_0_24px_rgba(34,211,238,0.25)] transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <span className="animate-spin rounded-full h-4 w-4 border-2 border-slate-950 border-t-transparent" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              <span>Get Recommendations</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="glass rounded-2xl p-4 border border-slate-800 flex items-start gap-3 text-xs text-muted-foreground">
        <Info className="w-4 h-4 shrink-0 mt-0.5 text-cyan-400" />
        <p className="leading-relaxed">
          <span className="font-semibold text-foreground">Multi-Signal Recommendation Logic:</span> blends co-occurrence graphs with user-item interactions and MMR diversity reranking. Check the{" "}
          <a href="/evaluation" className="text-cyan-300 underline underline-offset-2 hover:text-cyan-200">
            Evaluation
          </a>{" "}
          page for offline benchmark comparisons across Popularity, Content-Based, Collaborative Filtering, and Hybrid strategies.
        </p>
      </div>

      {isLoading && <MetricGridSkeleton count={8} />}

      {!isLoading && error !== null && <ErrorState error={error} onRetry={runRecommend} />}

      {!isLoading && !error && response && response.recommendations.length === 0 && (
        <div className="glass rounded-3xl border border-dashed border-slate-800 p-16 text-center text-muted-foreground text-sm space-y-1">
          <p className="font-semibold text-foreground">No Recommendations Found</p>
          <p className="text-xs max-w-md mx-auto">
            The catalog has limited co-purchase interactions for this ASIN. Try one of the suggested sample ASINs above.
          </p>
        </div>
      )}

      {!isLoading && !error && response && response.recommendations.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <span>{response.total_returned} Recommendation{response.total_returned === 1 ? "" : "s"}</span>
            </h2>
            <span className="text-xs text-muted-foreground font-mono bg-secondary/40 border border-slate-800 px-2.5 py-1 rounded-lg">
              Strategy: {response.strategy} · {response.execution_time_ms.toFixed(1)}ms
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {response.recommendations.map((rec) => (
              <RecommendationCard key={rec.product.asin} rec={rec} onExplain={handleExplain} />
            ))}
          </div>
        </section>
      )}

      <ExplanationDrawer
        isOpen={explanationModal.isOpen}
        onClose={() => setExplanationModal((prev) => ({ ...prev, isOpen: false }))}
        product={explanationModal.product}
        groundedExplanation={explanationModal.groundedExplanation}
      />
    </div>
  );
}
