"use client";

import React from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export const ProductCardSkeleton: React.FC = () => (
  <Card className="glass-panel rounded-2xl border border-slate-800/80 overflow-hidden">
    <div className="h-44 w-full bg-slate-900/50 p-4">
      <Skeleton className="h-full w-full rounded-xl bg-slate-800/50" />
    </div>
    <CardHeader className="p-5 pb-2 space-y-2.5">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-16 rounded-md bg-slate-800/70" />
        <Skeleton className="h-4 w-14 rounded-md bg-slate-800/70" />
      </div>
      <Skeleton className="h-4 w-full rounded-md bg-slate-800/70" />
      <Skeleton className="h-3.5 w-2/3 rounded-md bg-slate-800/60" />
    </CardHeader>
    <CardContent className="p-5 pt-1 space-y-3">
      <div className="flex items-center justify-between">
        <Skeleton className="h-3.5 w-24 rounded-md bg-slate-800/60" />
        <Skeleton className="h-5 w-16 rounded-md bg-slate-800/80" />
      </div>
      <Skeleton className="h-3 w-full rounded-md bg-slate-800/50" />
      <Skeleton className="h-3 w-4/5 rounded-md bg-slate-800/50" />
    </CardContent>
  </Card>
);

export const ProductGridSkeleton: React.FC<{ count?: number }> = ({ count = 6 }) => (
  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4" aria-busy="true" aria-live="polite">
    {Array.from({ length: count }).map((_, i) => (
      <ProductCardSkeleton key={i} />
    ))}
  </div>
);

export const MetricCardSkeleton: React.FC = () => (
  <Card className="glass rounded-2xl border border-slate-800 p-4 space-y-2">
    <CardContent className="p-0 space-y-2">
      <Skeleton className="h-3 w-28 rounded bg-slate-800/70" />
      <Skeleton className="h-7 w-20 rounded bg-slate-800/90" />
    </CardContent>
  </Card>
);

export const MetricGridSkeleton: React.FC<{ count?: number }> = ({ count = 4 }) => (
  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4" aria-busy="true" aria-live="polite">
    {Array.from({ length: count }).map((_, i) => (
      <MetricCardSkeleton key={i} />
    ))}
  </div>
);
