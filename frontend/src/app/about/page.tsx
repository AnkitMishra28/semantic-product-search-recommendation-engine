import type { Metadata } from "next";
import { ShieldAlert } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { PipelineVisualization } from "@/components/shared/PipelineVisualization";

export const metadata: Metadata = {
  title: "About — Semantic Product Search & Recommendation Engine",
};

const STACK_GROUPS: { title: string; subtitle: string; items: string[] }[] = [
  {
    title: "Frontend Engineering",
    subtitle: "Real-Time User Experience",
    items: [
      "Next.js 14 (App Router & Server/Client Components)",
      "TypeScript (Strict typing & schemas)",
      "Tailwind CSS (HSL design tokens & glassmorphism)",
      "Framer Motion (Hardware-accelerated animations)",
      "Lucide Icons & Web Vitals optimizations",
    ],
  },
  {
    title: "Backend & API Architecture",
    subtitle: "High-Throughput ML Serving",
    items: [
      "FastAPI (Asynchronous micro-framework)",
      "Pydantic v2 (Request/response validation & serialization)",
      "Uvicorn (ASGI server with lifespan lifecycle)",
      "Dynamic multi-origin CORS support for local/remote clients",
    ],
  },
  {
    title: "Retrieval & Neural Ranking",
    subtitle: "Multi-Stage Candidate Processing",
    items: [
      "FAISS (faiss-cpu with HNSW Index, M=32, efConstruction=200)",
      "sentence-transformers (all-MiniLM-L6-v2, 384-dimensional embeddings)",
      "Cross-Encoder (ms-marco-MiniLM-L-6-v2, cross-attention scoring)",
      "BM25 lexical retrieval (rank-bm25) & Reciprocal Rank Fusion (RRF)",
    ],
  },
  {
    title: "Personalization, Grounding & Evaluation",
    subtitle: "Explainability & Rigorous Metrics",
    items: [
      "Multi-signal personalization (business signals, ratings, review volume)",
      "Grounded Explanation layer with hallucination guardrails",
      "Maximal Marginal Relevance (MMR) diversity reranking",
      "scikit-learn, pandas, numpy for offline MRR/NDCG/Recall benchmarking",
    ],
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-12">
      {/* Thesis Header */}
      <section className="glass-panel rounded-3xl p-6 sm:p-10 border border-slate-800 shadow-2xl space-y-4 max-w-5xl">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold">
            Applied Scientist Research Thesis
          </span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-bold tracking-tight text-foreground">
          Amazon-Scale Semantic Product Search & Recommendation Engine
        </h1>
        <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
          Modern e-commerce product search operates in high-dimensional semantic spaces where lexical exact-match
          frequently fails on descriptive, multi-attribute, or use-case queries. This project implements and benchmarks
          a multi-stage neural search and recommendation architecture inspired by large-scale industry systems:
          coupling high-recall dense bi-encoder vector retrieval (FAISS HNSW), high-precision neural cross-encoder
          reranking (ms-marco), multi-signal personalization, and evidence-grounded explainability with strict catalog guardrails.
        </p>
      </section>

      {/* Mandatory Amazon Disclaimer */}
      <div className="rounded-3xl border border-amber-500/40 bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent p-5 text-xs text-amber-200/90 flex items-start gap-3 shadow-lg max-w-5xl">
        <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1 leading-relaxed">
          <p className="font-semibold text-amber-300 uppercase font-mono text-[11px]">
            Academic & Applied Scientist Portfolio Prototype
          </p>
          <p>
            This project is inspired by Amazon-scale e-commerce search architecture for educational and
            research purposes. It is not affiliated with, endorsed by, or built on proprietary Amazon
            systems, and does not claim production parity with any commercial Amazon service.
          </p>
        </div>
      </div>

      {/* Architecture Section */}
      <section className="space-y-6">
        <div className="border-b border-slate-800 pb-3">
          <h2 className="text-xl font-semibold text-foreground">System Architecture</h2>
          <p className="text-xs text-muted-foreground mt-1">
            Visual overview of the live inference pipeline and offline evaluation tracks.
          </p>
        </div>
        <PipelineVisualization />
      </section>

      {/* Architecture Deep Dive Cards */}
      <section className="grid grid-cols-1 gap-4 md:grid-cols-2 max-w-5xl">
        <Card className="glass-panel rounded-2xl border border-slate-800">
          <CardContent className="p-6 space-y-2.5">
            <p className="text-sm font-semibold text-foreground flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              Live vs Offline Architecture Distinction
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              In live runtime serving, search requests execute query understanding, dense FAISS (HNSW) retrieval,
              cross-encoder neural reranking, business-signal personalization, and grounded explanation generation.
              Lexical BM25 candidate generation and Reciprocal Rank Fusion (RRF) are benchmarked offline and surfaced
              through experiment artifacts on the Evaluation page.
            </p>
          </CardContent>
        </Card>

        <Card className="glass-panel rounded-2xl border border-slate-800">
          <CardContent className="p-6 space-y-2.5">
            <p className="text-sm font-semibold text-foreground flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              Engineering Scope & Real Data Integrity
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Every metric, latency distribution, product attribute, ranking score, and recommendation on this
              platform is backed by real artifacts: 60,000 processed Amazon Electronics items, genuine FAISS indexes,
              pre-computed interaction matrices, and reproducible Python evaluation benchmarks.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* Technology Stack Grid */}
      <section className="space-y-6 max-w-5xl">
        <div className="border-b border-slate-800 pb-3">
          <h2 className="text-xl font-semibold text-foreground">Technology Stack & Engineering Decisions</h2>
          <p className="text-xs text-muted-foreground mt-1">
            Production-grade open-source stack selected for latency, correctness, and developer velocity.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {STACK_GROUPS.map((group) => (
            <Card key={group.title} className="glass-panel rounded-2xl border border-slate-800 hover:border-cyan-400/30 transition-colors">
              <CardContent className="p-6 space-y-3">
                <div>
                  <p className="text-sm font-semibold text-foreground">{group.title}</p>
                  <p className="text-[11px] font-mono text-cyan-300 uppercase">{group.subtitle}</p>
                </div>
                <ul className="space-y-2 text-xs text-muted-foreground">
                  {group.items.map((item) => (
                    <li key={item} className="flex items-start gap-2">
                      <span className="text-cyan-400 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Dataset Attribution */}
      <section className="glass rounded-2xl p-6 border border-slate-800 space-y-2 max-w-5xl">
        <h2 className="text-sm font-semibold text-foreground uppercase font-mono tracking-wider text-cyan-300">
          Dataset & Evaluation Corpus
        </h2>
        <p className="text-xs text-muted-foreground leading-relaxed">
          Built on the <strong>Amazon Reviews 2023 (Electronics Category)</strong> corpus, comprising 60,000 real products,
          complete with titles, brand metadata, categories, bullet-point feature descriptions, pricing, verified star ratings,
          and co-purchasing behavioral graphs.
        </p>
      </section>
    </div>
  );
}
