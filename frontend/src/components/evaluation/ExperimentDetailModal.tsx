"use client";

import React, { useState, useEffect } from "react";
import { X, Copy, Check, Terminal, FileJson, Layers, Clock, Database } from "lucide-react";
import { apiClient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface ExperimentDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  experimentId: string | null;
  filename?: string;
  track?: string;
}

export const ExperimentDetailModal: React.FC<ExperimentDetailModalProps> = ({
  isOpen,
  onClose,
  experimentId,
  filename,
  track,
}) => {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"summary" | "json">("summary");

  useEffect(() => {
    if (isOpen && (experimentId || filename)) {
      setIsLoading(true);
      apiClient
        .getExperimentDetail(experimentId || filename || "")
        .then((res) => {
          setData(res);
        })
        .catch(() => {
          setData({ error: "Failed to load experiment artifact details." });
        })
        .finally(() => {
          setIsLoading(false);
        });
    } else {
      setData(null);
    }
  }, [isOpen, experimentId, filename]);

  if (!isOpen) return null;

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Experiment Artifact Inspector"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 backdrop-blur-md p-4 animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="glass-elevated relative w-full max-w-3xl rounded-3xl p-6 sm:p-7 shadow-2xl space-y-5 max-h-[88vh] overflow-hidden flex flex-col border border-slate-800"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3.5 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
              <FileJson className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-foreground font-mono">
                {data?.experiment_id || experimentId || filename}
              </h3>
              <p className="text-[11px] text-muted-foreground font-mono">{track || filename}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close modal"
            className="rounded-xl p-1.5 text-muted-foreground hover:bg-secondary/60 hover:text-foreground transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 shrink-0">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab("summary")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-colors ${
                activeTab === "summary"
                  ? "bg-cyan-400/15 border border-cyan-400/30 text-cyan-300"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Structured Overview
            </button>
            <button
              onClick={() => setActiveTab("json")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-colors ${
                activeTab === "json"
                  ? "bg-cyan-400/15 border border-cyan-400/30 text-cyan-300"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Raw Artifact JSON
            </button>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleCopy}
            disabled={!data}
            className="h-8 gap-1.5 text-xs rounded-xl border-slate-800 hover:border-cyan-400/40 text-muted-foreground hover:text-cyan-200"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? "Copied" : "Copy JSON"}</span>
          </Button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto space-y-4 pr-1 text-xs">
          {isLoading && (
            <div className="py-16 text-center text-muted-foreground space-y-2">
              <span className="animate-spin rounded-full inline-block h-6 w-6 border-2 border-cyan-400 border-t-transparent" />
              <p className="text-xs">Loading experiment artifact payload...</p>
            </div>
          )}

          {!isLoading && data && activeTab === "summary" && (
            <div className="space-y-4">
              {/* Metadata Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3 space-y-1">
                  <p className="text-[10px] font-mono uppercase text-muted-foreground">Experiment Timestamp</p>
                  <p className="font-mono text-foreground font-semibold text-[11px]">
                    {data.timestamp ? new Date(data.timestamp).toLocaleString() : "—"}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3 space-y-1">
                  <p className="text-[10px] font-mono uppercase text-muted-foreground">Dataset Scope</p>
                  <p className="font-mono text-foreground font-semibold text-[11px] truncate">
                    {data.dataset?.name || data.dataset || "Amazon Reviews 2023"}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-secondary/30 p-3 space-y-1">
                  <p className="text-[10px] font-mono uppercase text-muted-foreground">Source Artifact</p>
                  <p className="font-mono text-cyan-300 font-semibold text-[11px] truncate">
                    {filename || `${data.experiment_id}.json`}
                  </p>
                </div>
              </div>

              {/* Models / Parameters */}
              {(data.models || data.parameters || data.optimal_hyperparameters) && (
                <div className="rounded-2xl border border-slate-800 bg-secondary/20 p-4 space-y-2">
                  <p className="text-xs font-mono font-bold uppercase text-cyan-300">
                    Configuration & Parameters
                  </p>
                  <pre className="text-[11px] font-mono text-foreground/90 bg-slate-950/70 rounded-xl p-3 border border-slate-800 overflow-x-auto">
                    {JSON.stringify(data.models || data.parameters || data.optimal_hyperparameters, null, 2)}
                  </pre>
                </div>
              )}

              {/* Metrics Breakdown */}
              {(data.metrics || data.methods_comparison || data.master_test_benchmark || data.master_comparison_pipelines) && (
                <div className="rounded-2xl border border-slate-800 bg-secondary/20 p-4 space-y-2">
                  <p className="text-xs font-mono font-bold uppercase text-cyan-300">
                    Measured Benchmark Results
                  </p>
                  <pre className="text-[11px] font-mono text-cyan-100 bg-slate-950/70 rounded-xl p-3 border border-slate-800 overflow-x-auto max-h-60">
                    {JSON.stringify(
                      data.metrics ||
                        data.methods_comparison ||
                        data.master_test_benchmark ||
                        data.master_comparison_pipelines,
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}

              {/* Latency Stats */}
              {(data.latency || data.latency_ms || data.latency_benchmarks || data.latency_breakdown) && (
                <div className="rounded-2xl border border-slate-800 bg-secondary/20 p-4 space-y-2">
                  <p className="text-xs font-mono font-bold uppercase text-cyan-300 flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    Offline Latency Breakdown
                  </p>
                  <pre className="text-[11px] font-mono text-foreground/90 bg-slate-950/70 rounded-xl p-3 border border-slate-800 overflow-x-auto">
                    {JSON.stringify(
                      data.latency ||
                        data.latency_ms ||
                        data.latency_benchmarks ||
                        data.latency_breakdown,
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}
            </div>
          )}

          {!isLoading && data && activeTab === "json" && (
            <div className="relative">
              <pre className="text-[11px] font-mono text-cyan-200/90 bg-slate-950 p-4 rounded-2xl border border-slate-800 overflow-x-auto max-h-[50vh]">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 pt-3 flex items-center justify-between shrink-0">
          <span className="text-[10px] font-mono text-muted-foreground">
            Immutable artifact served from <code className="text-slate-400">experiments/results/</code>
          </span>
          <Button variant="secondary" size="sm" onClick={onClose} className="rounded-xl px-5 h-8 text-xs">
            Close
          </Button>
        </div>
      </div>
    </div>
  );
};
