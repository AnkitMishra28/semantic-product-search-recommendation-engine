"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface EvaluationMetricCardProps {
  label: string;
  value: string;
  sublabel?: string;
  accent?: boolean;
}

export const EvaluationMetricCard: React.FC<EvaluationMetricCardProps> = ({
  label,
  value,
  sublabel,
  accent = false,
}) => (
  <Card
    className={cn(
      "glass rounded-2xl border transition-all duration-200",
      accent
        ? "border-cyan-400/40 bg-cyan-950/20 shadow-[0_0_20px_rgba(34,211,238,0.15)]"
        : "border-slate-800 bg-secondary/30"
    )}
  >
    <CardContent className="p-4 text-center space-y-1">
      <p className="text-[11px] uppercase tracking-wider text-muted-foreground font-mono font-semibold">
        {label}
      </p>
      <p className={cn("text-xl sm:text-2xl font-bold font-mono", accent ? "text-cyan-300" : "text-foreground")}>
        {value}
      </p>
      {sublabel && <p className="text-[10px] text-muted-foreground leading-snug">{sublabel}</p>}
    </CardContent>
  </Card>
);
