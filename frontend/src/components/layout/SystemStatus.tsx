"use client";

import React, { useEffect, useMemo, useState } from "react";

type Status = "checking" | "ready" | "degraded" | "offline";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function SystemStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let active = true;

    const check = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/ready`, { cache: "no-store" });
        if (!active) return;
        if (!response.ok) {
          setStatus("offline");
          return;
        }
        const payload = await response.json();
        setStatus(payload?.status === "ready" ? "ready" : "degraded");
      } catch {
        if (active) setStatus("offline");
      }
    };

    check();
    const id = window.setInterval(check, 30000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (status === "ready") return "Backend ready";
    if (status === "degraded") return "Backend initializing";
    if (status === "offline") return "Backend offline";
    return "Checking backend";
  }, [status]);

  const dotClass =
    status === "ready"
      ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
      : status === "degraded"
        ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]"
        : status === "offline"
          ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]"
          : "bg-cyan-400 animate-pulse";

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-border/80 bg-secondary/40 px-3 py-1 text-[11px] text-muted-foreground shadow-sm"
      aria-live="polite"
    >
      <span className={`h-2 w-2 rounded-full ${dotClass}`} />
      <span className="font-mono text-foreground/80">{statusLabel}</span>
    </div>
  );
}
