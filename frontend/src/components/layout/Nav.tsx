"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/search", label: "Search" },
  { href: "/recommendations", label: "Recommendations" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/about", label: "About" },
];

export const Nav: React.FC = () => {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = React.useState(false);

  React.useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <div className="relative">
      <nav aria-label="Primary" className="hidden items-center gap-1.5 md:flex">
        {LINKS.map((link) => {
          const isActive = link.href === "/" ? pathname === "/" : pathname?.startsWith(link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "relative rounded-xl px-3.5 py-2 text-xs font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                isActive
                  ? "border border-cyan-400/35 bg-cyan-400/15 text-cyan-200 shadow-[0_0_20px_rgba(34,211,238,0.18)] font-semibold"
                  : "border border-transparent text-muted-foreground hover:border-slate-800 hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      <button
        type="button"
        className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-border/80 bg-secondary/50 text-muted-foreground hover:text-foreground md:hidden"
        aria-label={mobileOpen ? "Close navigation menu" : "Open navigation menu"}
        aria-expanded={mobileOpen}
        onClick={() => setMobileOpen((v) => !v)}
      >
        {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
      </button>

      {mobileOpen && (
        <div className="glass-elevated absolute right-0 top-12 z-50 w-60 rounded-2xl p-2.5 shadow-2xl md:hidden border border-border/90 animate-in fade-in slide-in-from-top-2 duration-200">
          <nav aria-label="Mobile primary" className="flex flex-col gap-1">
            {LINKS.map((link) => {
              const isActive = link.href === "/" ? pathname === "/" : pathname?.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    "rounded-xl border px-3 py-2.5 text-xs font-medium transition-colors",
                    isActive
                      ? "border-cyan-400/40 bg-cyan-400/15 text-cyan-200 font-semibold"
                      : "border-transparent text-muted-foreground hover:border-slate-800 hover:bg-secondary/60 hover:text-foreground"
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
      )}
    </div>
  );
};
