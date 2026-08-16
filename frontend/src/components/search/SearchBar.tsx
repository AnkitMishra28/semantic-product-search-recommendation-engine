"use client";

import React, { useEffect, useState } from "react";
import { Search, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading?: boolean;
  initialQuery?: string;
  queryChips?: string[];
}

const SAMPLE_QUERIES = [
  "wireless noise cancelling over-ear headphones",
  "fast charging usb-c power bank for travel",
  "ergonomic mechanical keyboard for mac",
  "gaming laptop with rtx gpu and 16gb ram under 80000",
];

export const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  isLoading = false,
  initialQuery = "",
  queryChips,
}) => {
  const [query, setQuery] = useState(initialQuery);
  const chips = queryChips && queryChips.length > 0 ? queryChips : SAMPLE_QUERIES;

  useEffect(() => {
    setQuery(initialQuery);
  }, [initialQuery]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleChipClick = (sample: string) => {
    setQuery(sample);
    onSearch(sample);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-3.5">
      <form
        onSubmit={handleSubmit}
        className="glass-panel surface-ring relative flex items-center overflow-hidden rounded-2xl border border-slate-800 focus-within:border-cyan-400/50 shadow-2xl transition-all"
      >
        <div className="pl-5 text-cyan-300">
          <Search className="w-5 h-5" />
        </div>
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search products by natural description, specs, or use-case..."
          aria-label="Search products"
          className="h-14 sm:h-16 border-0 bg-transparent py-4 text-sm sm:text-base focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/50 text-foreground"
        />
        <div className="hidden pr-3 text-[11px] text-muted-foreground/70 sm:flex items-center">
          <kbd className="rounded-md border border-slate-800 bg-secondary/60 px-2 py-1 font-mono text-[10px]">
            Enter
          </kbd>
        </div>
        <div className="pr-3">
          <Button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="gap-2 px-5 sm:px-6 h-10 sm:h-11 rounded-xl bg-gradient-to-r from-cyan-400 to-blue-500 hover:from-cyan-300 hover:to-blue-400 text-slate-950 font-semibold shadow-[0_0_24px_rgba(34,211,238,0.3)] transition-all disabled:opacity-50"
          >
            {isLoading ? (
              <span className="animate-spin rounded-full h-4 w-4 border-2 border-slate-950 border-t-transparent" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            <span>Search</span>
          </Button>
        </div>
      </form>

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground pt-1">
        <span className="font-mono text-[11px] text-cyan-300/80 uppercase font-semibold">Try Queries:</span>
        {chips.map((sample) => (
          <button
            key={sample}
            type="button"
            onClick={() => handleChipClick(sample)}
            className="rounded-full border border-slate-800/90 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground transition-all hover:border-cyan-400/40 hover:bg-cyan-400/10 hover:text-cyan-200"
          >
            &ldquo;{sample}&rdquo;
          </button>
        ))}
      </div>
    </div>
  );
};
