"use client";

import React, { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { SearchBar } from "@/components/search/SearchBar";
import { QueryUnderstandingBadge } from "@/components/search/QueryUnderstandingBadge";
import { PipelineInspector } from "@/components/search/PipelineInspector";
import { ProductGrid } from "@/components/search/ProductGrid";
import { RecommendationSection } from "@/components/recommendations/RecommendationSection";
import { ExplanationDrawer } from "@/components/recommendations/ExplanationDrawer";
import { ErrorState } from "@/components/shared/ErrorState";
import { ProductGridSkeleton } from "@/components/shared/LoadingSkeleton";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api";
import { GroundedExplanation, Product, RecommendationItem, SearchResponse } from "@/types";

export default function SearchPageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [lastQuery, setLastQuery] = useState<string>("");

  const [explanationModal, setExplanationModal] = useState<{
    isOpen: boolean;
    product: Product | null;
    query: string;
    groundedExplanation: GroundedExplanation | null;
    isLoadingExplanation: boolean;
  }>({
    isOpen: false,
    product: null,
    query: "",
    groundedExplanation: null,
    isLoadingExplanation: false,
  });

  const runSearch = useCallback(async (query: string) => {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    setRecommendations([]);
    setLastQuery(query);
    try {
      const resp = await apiClient.search({
        query,
        top_k_retrieval: 100,
        top_k_reranking: 20,
        enable_reranking: true,
        enable_explanation: true,
        ranking_strategy: "hybrid",
      });
      setSearchResponse(resp);

      if (resp.results.length > 0) {
        const topAsin = resp.results[0].product.asin;
        try {
          const recResp = await apiClient.recommend({ asin: topAsin, top_k: 4, strategy: "hybrid" });
          setRecommendations(recResp.recommendations);
        } catch {
          // Complementary recommendations are an optional bonus; failures don't block search results
        }
      }
    } catch (err) {
      setSearchResponse(null);
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (initialQuery && initialQuery !== lastQuery) {
      runSearch(initialQuery);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery]);

  const handleSearch = (query: string) => {
    router.push(`/search?q=${encodeURIComponent(query)}`);
    runSearch(query);
  };

  const handleExplain = async (asin: string) => {
    const item = searchResponse?.results.find((r) => r.product.asin === asin);
    if (!item) return;

    setExplanationModal({
      isOpen: true,
      product: item.product,
      query: searchResponse?.query || "",
      groundedExplanation: item.grounded_explanation || null,
      isLoadingExplanation: false,
    });

    if (!item.grounded_explanation) {
      setExplanationModal((prev) => ({ ...prev, isLoadingExplanation: true }));
      try {
        const explanation = await apiClient.explain({ query: searchResponse?.query, product_id: asin });
        setExplanationModal((prev) => ({ ...prev, groundedExplanation: explanation, isLoadingExplanation: false }));
      } catch {
        setExplanationModal((prev) => ({ ...prev, isLoadingExplanation: false }));
      }
    }
  };

  return (
    <div className="space-y-10">
      {/* Search Header Banner */}
      <section className="glass-panel rounded-3xl p-6 sm:p-8 border border-slate-800 shadow-2xl space-y-2">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold">
            Real-Time Inference Console
          </span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          Semantic Product Search
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Executes the complete multi-stage pipeline: Query Understanding → Dense FAISS (all-MiniLM-L6) → Cross-Encoder Neural Reranking (ms-marco) → Personalization → Grounded Explanations with strict metadata guardrails.
        </p>
      </section>

      {/* Search Input Bar */}
      <SearchBar
        onSearch={handleSearch}
        isLoading={isLoading}
        initialQuery={initialQuery}
        queryChips={[
          "wireless noise cancelling over-ear headphones",
          "4k webcam for streaming and work meetings",
          "lightweight bluetooth earbuds with long battery",
          "usb c hub with hdmi ethernet and pd charging",
        ]}
      />

      {/* Loading State */}
      {isLoading && (
        <div className="space-y-6">
          <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4" aria-live="polite">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-cyan-300 font-semibold flex items-center gap-2">
                <span className="animate-spin rounded-full h-3.5 w-3.5 border-2 border-cyan-400 border-t-transparent" />
                Pipeline Execution In Progress
              </span>
              <span className="text-[11px] font-mono text-muted-foreground">Dense vector + Cross-Encoder</span>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3 text-xs space-y-1">
                <p className="font-semibold text-foreground/90">1. Query Understanding</p>
                <p className="text-[11px] text-muted-foreground">Parsing entities, intent & price constraints...</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3 text-xs space-y-1">
                <p className="font-semibold text-foreground/90">2. Dense Bi-Encoder ANN</p>
                <p className="text-[11px] text-muted-foreground">Searching 60,000 FAISS HNSW embeddings...</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3 text-xs space-y-1">
                <p className="font-semibold text-foreground/90">3. Neural Reranking</p>
                <p className="text-[11px] text-muted-foreground">ms-marco cross-attention rescoring...</p>
              </div>
            </div>
            <Skeleton className="h-2 w-full rounded-full bg-slate-800" />
          </div>

          <Skeleton className="h-28 w-full rounded-2xl bg-secondary/30" />
          <Skeleton className="h-24 w-full rounded-2xl bg-secondary/30" />
          <ProductGridSkeleton count={6} />
        </div>
      )}

      {/* Error State */}
      {!isLoading && error !== null && (
        <ErrorState error={error} onRetry={() => runSearch(lastQuery || initialQuery)} />
      )}

      {/* Search Results */}
      {!isLoading && error === null && searchResponse && (
        <>
          <section className="space-y-5 animate-in fade-in duration-300">
            <QueryUnderstandingBadge data={searchResponse.query_understanding} />
            <PipelineInspector
              timings={searchResponse.timings}
              totalRetrieved={searchResponse.total_retrieved}
              totalReturned={searchResponse.total_returned}
            />
          </section>

          <section className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                <span>Ranked Products</span>
                <span className="text-xs font-mono text-cyan-300 font-normal">
                  ({searchResponse.results.length} candidates)
                </span>
              </h2>
              <span className="text-xs text-muted-foreground font-mono">
                Strategy: {searchResponse.results.length > 0 ? "Hybrid Multi-Signal" : "—"}
              </span>
            </div>

            <ProductGrid
              items={searchResponse.results}
              onExplain={handleExplain}
              onSelectProduct={(asin) => router.push(`/recommendations?asin=${encodeURIComponent(asin)}`)}
            />
          </section>

          <RecommendationSection
            recommendations={recommendations}
            anchorTitle={searchResponse.results[0]?.product.title}
            title="Customers Also Considered"
            onExplain={handleExplain}
            onSelectProduct={(asin) => router.push(`/recommendations?asin=${encodeURIComponent(asin)}`)}
          />
        </>
      )}

      {/* Initial Empty / Idle State */}
      {!isLoading && !error && !searchResponse && (
        <div className="glass rounded-3xl border border-dashed border-slate-800 p-16 text-center text-muted-foreground text-sm space-y-2">
          <p className="text-base font-semibold text-foreground">Awaiting Search Query</p>
          <p className="text-xs text-muted-foreground max-w-md mx-auto">
            Submit a natural language search query above to execute real-time dense retrieval, cross-encoder neural reranking, and grounded explanation generation.
          </p>
        </div>
      )}

      {/* Grounded Explanation Modal Drawer */}
      <ExplanationDrawer
        isOpen={explanationModal.isOpen}
        onClose={() => setExplanationModal((prev) => ({ ...prev, isOpen: false }))}
        product={explanationModal.product}
        query={explanationModal.query}
        groundedExplanation={explanationModal.groundedExplanation}
        isLoading={explanationModal.isLoadingExplanation}
        onRequestExplanation={
          explanationModal.product ? () => handleExplain(explanationModal.product!.asin) : undefined
        }
      />
    </div>
  );
}
