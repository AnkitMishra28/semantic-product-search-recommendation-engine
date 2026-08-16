"use client";

import React from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Layers,
  GitMerge,
  Sparkles,
  LineChart,
  ShieldCheck,
  ArrowRight,
  Cpu,
  Database,
  Search,
  CheckCircle2,
} from "lucide-react";
import { SearchBar } from "@/components/search/SearchBar";
import { PipelineVisualization } from "@/components/shared/PipelineVisualization";
import { Card, CardContent } from "@/components/ui/card";

const HIGHLIGHTS = [
  {
    icon: <Layers className="w-5 h-5 text-cyan-400" />,
    title: "Multi-Stage Retrieval",
    subtitle: "High-Recall Candidate Generation",
    detail:
      "Dense FAISS (HNSW) bi-encoder vector search retrieves high-recall candidates in real-time, while BM25 lexical search + RRF are benchmarked offline.",
  },
  {
    icon: <Sparkles className="w-5 h-5 text-cyan-400" />,
    title: "Neural Reranking",
    subtitle: "Cross-Encoder Precision",
    detail:
      "Cross-encoder with full self-attention (ms-marco-MiniLM) rescores top candidates to capture fine-grained semantic nuances and query-product fit.",
  },
  {
    icon: <GitMerge className="w-5 h-5 text-cyan-400" />,
    title: "Personalized Ranking",
    subtitle: "Multi-Signal Optimization",
    detail:
      "Combines semantic similarity scores with business signals, verified star ratings, review volume, and historical interaction affinities.",
  },
  {
    icon: <ShieldCheck className="w-5 h-5 text-cyan-400" />,
    title: "Grounded Explanations",
    subtitle: "Evidence Guardrails",
    detail:
      "Generates transparent, evidence-grounded rationales backed by verified product metadata with explicit warnings for unverified claims.",
  },
  {
    icon: <LineChart className="w-5 h-5 text-cyan-400" />,
    title: "Measurable Evaluation",
    subtitle: "Track-Level Benchmarks",
    detail:
      "Authoritative offline benchmark artifacts quantify Recall@K, MRR@10, and NDCG@10 across candidate generation and reranking stages.",
  },
];

const RESEARCH_CHIPS = [
  "wireless noise cancelling over-ear headphones",
  "usb c 65w charger for ultrabook travel",
  "1080p webcam low-light conference setup",
  "portable 2tb ssd thunderbolt compatible",
];

export default function LandingPage() {
  const router = useRouter();

  const handleSearch = (query: string) => {
    router.push(`/search?q=${encodeURIComponent(query)}`);
  };

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-800/80 bg-gradient-to-b from-slate-900/90 via-slate-950/80 to-slate-950/95 px-5 py-10 sm:px-10 sm:py-16 shadow-2xl">
        <div className="pointer-events-none absolute -right-20 top-[-80px] h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -left-20 bottom-[-100px] h-96 w-96 rounded-full bg-blue-600/10 blur-3xl" />

        <div className="relative mx-auto max-w-5xl space-y-8 text-center">
          <motion.div
            initial={false}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3.5 py-1.5 text-xs font-mono text-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.15)]"
          >
            <Cpu className="h-3.5 w-3.5 animate-pulse" />
            <span>ML SEARCH & RECOMMENDATION RESEARCH</span>
          </motion.div>

          <div className="space-y-4">
            <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl leading-[1.12]">
              Semantic Product Search,
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-sky-200 to-blue-400">
                Built as a Multi-Stage ML System.
              </span>
            </h1>

            <p className="mx-auto max-w-3xl text-sm sm:text-base leading-relaxed text-muted-foreground">
              Applied-scientist-level neural discovery engine featuring live query understanding, dense FAISS
              bi-encoder retrieval, cross-encoder reranking, multi-signal personalization, and grounded LLM
              explanations with hallucination guardrails across 60,000 real catalog items.
            </p>
          </div>

          <div className="pt-2">
            <SearchBar onSearch={handleSearch} queryChips={RESEARCH_CHIPS} />
          </div>

          {/* Architecture Spec Strip */}
          <div className="mx-auto grid max-w-4xl grid-cols-2 gap-3 text-left sm:grid-cols-4 pt-4">
            <div className="glass rounded-xl p-3.5 border border-slate-800/80 space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase text-cyan-300 font-semibold">
                <Database className="h-3.5 w-3.5" />
                <span>Catalog</span>
              </div>
              <p className="text-xs font-medium text-foreground">60,000 Real Products</p>
              <p className="text-[10px] text-muted-foreground">Amazon Reviews 2023</p>
            </div>

            <div className="glass rounded-xl p-3.5 border border-slate-800/80 space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase text-cyan-300 font-semibold">
                <Search className="h-3.5 w-3.5" />
                <span>Retrieval</span>
              </div>
              <p className="text-xs font-medium text-foreground">FAISS HNSW Index</p>
              <p className="text-[10px] text-muted-foreground">all-MiniLM-L6-v2 (384d)</p>
            </div>

            <div className="glass rounded-xl p-3.5 border border-slate-800/80 space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase text-cyan-300 font-semibold">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Reranking</span>
              </div>
              <p className="text-xs font-medium text-foreground">Cross-Encoder</p>
              <p className="text-[10px] text-muted-foreground">ms-marco-MiniLM-L6</p>
            </div>

            <div className="glass rounded-xl p-3.5 border border-slate-800/80 space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] font-mono uppercase text-cyan-300 font-semibold">
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>Explainability</span>
              </div>
              <p className="text-xs font-medium text-foreground">Grounded Rationale</p>
              <p className="text-[10px] text-muted-foreground">With Evidence & Warnings</p>
            </div>
          </div>
        </div>
      </section>

      {/* Feature / Component Deep-Dives */}
      <section className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-border/80 pb-3">
          <div>
            <h2 className="text-xl font-semibold text-foreground">Core System Components</h2>
            <p className="text-xs text-muted-foreground">
              Engineered for low latency, high recall, and fine-grained ranking precision.
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-300/90 self-start sm:self-auto">
            Production-Grade ML Architecture
          </span>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {HIGHLIGHTS.map((h, idx) => (
            <motion.div
              key={h.title}
              initial={false}
              className="h-full"
            >
              <Card className="surface-ring glass h-full rounded-2xl border border-slate-800/80 hover:border-cyan-400/40 transition-all duration-300">
                <CardContent className="p-5 space-y-3 flex flex-col justify-between h-full">
                  <div className="space-y-2.5">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300 shadow-[0_0_16px_rgba(34,211,238,0.15)]">
                      {h.icon}
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-foreground">{h.title}</p>
                      <p className="text-[10px] font-mono text-cyan-300/80 uppercase">{h.subtitle}</p>
                    </div>
                    <p className="text-xs leading-relaxed text-muted-foreground">{h.detail}</p>
                  </div>
                  <div className="pt-2 flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground/80">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                    <span>Evaluated component</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Pipeline Visualization */}
      <section className="space-y-6">
        <div className="text-center space-y-1.5">
          <h2 className="text-xl font-semibold text-foreground">Multi-Stage Pipeline Execution Architecture</h2>
          <p className="text-xs text-muted-foreground max-w-2xl mx-auto">
            The system clearly decouples the real-time live execution path from the offline benchmarking track.
          </p>
        </div>
        <PipelineVisualization />
      </section>

      {/* Navigation Quick Links */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Link
          href="/recommendations"
          className="surface-ring glass group rounded-2xl p-5 space-y-2 border border-slate-800/80 hover:border-cyan-400/40 transition-all"
        >
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground group-hover:text-cyan-300 transition-colors">
              Recommendation Lab
            </span>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-1 group-hover:text-cyan-300 transition-all" />
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Test Item-to-Item (ASIN) and User-History multi-signal recommendations with MMR diversity sweeps.
          </p>
        </Link>

        <Link
          href="/evaluation"
          className="surface-ring glass group rounded-2xl p-5 space-y-2 border border-slate-800/80 hover:border-cyan-400/40 transition-all"
        >
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground group-hover:text-cyan-300 transition-colors">
              Evaluation & Benchmarks
            </span>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-1 group-hover:text-cyan-300 transition-all" />
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Inspect authoritative Recall@K, MRR@10, and NDCG@10 artifacts derived from real experiments.
          </p>
        </Link>

        <Link
          href="/about"
          className="surface-ring glass group rounded-2xl p-5 space-y-2 border border-slate-800/80 hover:border-cyan-400/40 transition-all"
        >
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground group-hover:text-cyan-300 transition-colors">
              About & Methodology
            </span>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-1 group-hover:text-cyan-300 transition-all" />
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Read project thesis, architecture deep-dive, latency analysis, and Amazon disclaimer.
          </p>
        </Link>
      </section>
    </div>
  );
}
