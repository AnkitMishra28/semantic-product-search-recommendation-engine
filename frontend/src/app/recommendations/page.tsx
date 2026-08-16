import type { Metadata } from "next";
import { Suspense } from "react";
import RecommendationsPageClient from "./RecommendationsPageClient";
import { MetricGridSkeleton } from "@/components/shared/LoadingSkeleton";

export const metadata: Metadata = {
  title: "Recommendations — Semantic Product Search & Recommendation Engine",
};

export default function RecommendationsPage() {
  return (
    <Suspense fallback={<MetricGridSkeleton count={8} />}>
      <RecommendationsPageClient />
    </Suspense>
  );
}
