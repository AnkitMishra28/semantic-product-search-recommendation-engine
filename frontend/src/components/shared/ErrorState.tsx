"use client";

import React from "react";
import { AlertTriangle, PlugZap, TimerOff, RefreshCw, ServerCrash } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api";

interface ErrorStateProps {
  error: unknown;
  onRetry?: () => void;
  className?: string;
}

function describeError(error: unknown): { icon: React.ReactNode; title: string; detail: string } {
  if (error instanceof ApiError) {
    if (error.kind === "network") {
      return {
        icon: <PlugZap className="w-6 h-6 text-destructive" />,
        title: "Backend unavailable",
        detail: error.message,
      };
    }
    if (error.kind === "timeout") {
      return {
        icon: <TimerOff className="w-6 h-6 text-destructive" />,
        title: "Request timed out",
        detail: error.message,
      };
    }
    if (error.status && error.status >= 500) {
      return {
        icon: <ServerCrash className="w-6 h-6 text-destructive" />,
        title: `Server error (${error.status})`,
        detail: error.message,
      };
    }
    return {
      icon: <AlertTriangle className="w-6 h-6 text-destructive" />,
      title: error.status ? `Request failed (${error.status})` : "Request failed",
      detail: error.message,
    };
  }

  return {
    icon: <AlertTriangle className="w-6 h-6 text-destructive" />,
    title: "Something went wrong",
    detail: error instanceof Error ? error.message : "An unexpected error occurred.",
  };
}

export const ErrorState: React.FC<ErrorStateProps> = ({ error, onRetry, className }) => {
  const { icon, title, detail } = describeError(error);

  return (
    <div
      role="alert"
      className={`glass-panel rounded-3xl border border-destructive/40 bg-gradient-to-b from-destructive/10 via-slate-950 to-slate-950 p-8 text-center space-y-4 shadow-2xl ${className ?? ""}`}
    >
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-destructive/40 bg-destructive/15 text-destructive shadow-[0_0_24px_rgba(239,68,68,0.2)]">
        {icon}
      </div>
      <div className="space-y-1.5">
        <h3 className="text-base font-bold text-foreground font-sans">{title}</h3>
        <p className="text-xs text-muted-foreground max-w-md mx-auto leading-relaxed">{detail}</p>
      </div>
      {onRetry && (
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          className="gap-2 px-5 h-9 rounded-xl border-slate-800 hover:border-destructive/50 hover:bg-destructive/10 text-foreground text-xs"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Request</span>
        </Button>
      )}
    </div>
  );
};
