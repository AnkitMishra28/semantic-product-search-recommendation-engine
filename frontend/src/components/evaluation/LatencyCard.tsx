"use client";

import React from "react";
import { Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatMs } from "@/lib/utils";

interface LatencyStats {
  p50_ms?: number;
  p90_ms?: number;
  p95_ms?: number;
  p99_ms?: number;
  mean_ms?: number;
}

interface LatencyCardProps {
  title: string;
  stats: LatencyStats;
}

export const LatencyCard: React.FC<LatencyCardProps> = ({ title, stats }) => (
  <Card className="glass-panel rounded-2xl border border-slate-800 p-4 space-y-3">
    <CardHeader className="p-0 pb-1">
      <CardTitle className="text-xs font-semibold flex items-center justify-between">
        <span className="flex items-center gap-2 text-foreground font-sans">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          {title}
        </span>
        <span className="font-mono text-[10px] text-cyan-300/80 uppercase">Offline Latency</span>
      </CardTitle>
    </CardHeader>
    <CardContent className="p-0 grid grid-cols-5 gap-2 text-center pt-1 border-t border-slate-800/80">
      {(["p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms"] as const).map((k) => (
        <div key={k} className="rounded-lg bg-secondary/30 p-2 border border-slate-800/60">
          <p className="text-[9px] uppercase text-muted-foreground font-mono font-semibold">
            {k.replace("_ms", "")}
          </p>
          <p className="text-xs font-mono font-bold text-foreground mt-0.5">{formatMs(stats[k])}</p>
        </div>
      ))}
    </CardContent>
  </Card>
);
