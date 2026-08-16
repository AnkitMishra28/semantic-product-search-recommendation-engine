import type { Metadata } from "next";
import { Suspense } from "react";
import SearchPageClient from "./SearchPageClient";
import { ProductGridSkeleton } from "@/components/shared/LoadingSkeleton";

export const metadata: Metadata = {
  title: "Search — Semantic Product Search & Recommendation Engine",
};

export default function SearchPage() {
  return (
    <Suspense fallback={<ProductGridSkeleton />}>
      <SearchPageClient />
    </Suspense>
  );
}
