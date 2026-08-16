import type { Metadata } from "next";
import Link from "next/link";
import "@/app/globals.css";
import { Orbit, Sparkles } from "lucide-react";
import { JetBrains_Mono, Sora, Space_Grotesk } from "next/font/google";
import { Nav } from "@/components/layout/Nav";
import { SystemStatus } from "@/components/layout/SystemStatus";

const sora = Sora({
  subsets: ["latin"],
  variable: "--font-sora",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "Semantic Product Search & Recommendation Engine",
  description:
    "An Amazon-inspired multi-stage research prototype implementing hybrid retrieval, cross-encoder reranking, personalization, and grounded explanations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${sora.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable} min-h-screen text-foreground flex flex-col selection:bg-primary selection:text-primary-foreground`}
      >
        <header className="sticky top-0 z-50 w-full px-3 pt-3 sm:px-6 sm:pt-4">
          <div className="glass surface-ring mx-auto flex min-h-16 w-full max-w-7xl items-center justify-between gap-3 rounded-2xl px-4 sm:px-6 py-2.5">
            <Link href="/" className="flex items-center gap-3 shrink-0 group" aria-label="Go to home page">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/30 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 text-cyan-300 shadow-[0_0_24px_rgba(34,211,238,0.25)] group-hover:border-cyan-300/50 group-hover:shadow-[0_0_28px_rgba(34,211,238,0.35)] transition-all">
                <Orbit className="h-5 w-5 group-hover:rotate-45 transition-transform duration-500" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold tracking-tight text-foreground font-sans">
                    Semantic Search & Rec Engine
                  </span>
                  <span className="hidden sm:inline-flex items-center gap-1 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-0.5 text-[10px] font-mono text-cyan-300">
                    <Sparkles className="h-2.5 w-2.5" />
                    Research Platform
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground hidden sm:block">
                  Amazon-Scale Multi-Stage Neural Search Prototype
                </p>
              </div>
            </Link>

            <Nav />

            <div className="hidden lg:flex shrink-0">
              <SystemStatus />
            </div>
          </div>
        </header>

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 pb-12 pt-6 sm:px-6 sm:pt-8 space-y-10">
          {children}
        </main>

        <footer className="px-4 pb-6 sm:px-6 sm:pb-8 mt-auto">
          <div className="glass mx-auto grid max-w-7xl grid-cols-1 gap-4 rounded-2xl p-5 text-xs text-muted-foreground sm:grid-cols-3 sm:items-center">
            <div className="space-y-1">
              <p className="font-semibold text-foreground flex items-center gap-2">
                <Orbit className="h-4 w-4 text-cyan-400" />
                Semantic Search & Rec Engine
              </p>
              <p className="font-mono text-[11px]">Applied Scientist Portfolio Prototype · 60k Catalog</p>
            </div>
            <div className="text-[11px] sm:text-center space-y-0.5">
              <p>Dataset: Amazon Reviews 2023 (Electronics Category)</p>
              <p className="text-muted-foreground/75">Dense Bi-Encoder (all-MiniLM-L6) · Cross-Encoder (ms-marco)</p>
            </div>
            <div className="text-[11px] sm:text-right space-y-0.5">
              <p className="text-foreground/90 font-medium">Educational Research Prototype</p>
              <p className="text-muted-foreground/75">Not affiliated with or endorsed by Amazon.</p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
