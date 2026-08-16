"use client";

import React from "react";
import { SearchResultItem } from "@/types";
import { ProductCard } from "./ProductCard";

interface ProductGridProps {
  items: SearchResultItem[];
  onSelectProduct?: (asin: string) => void;
  onExplain?: (asin: string) => void;
}

export const ProductGrid: React.FC<ProductGridProps> = ({
  items,
  onSelectProduct,
  onExplain,
}) => {
  if (items.length === 0) {
    return (
      <div className="glass rounded-2xl border border-dashed border-border p-12 text-center text-muted-foreground">
        <p className="text-sm">No products found for this query.</p>
        <p className="mt-1 text-xs">Try broader terms, alternate brands, or fewer hard constraints.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {items.map((item, index) => (
        <ProductCard
          key={item.product.asin}
          item={item}
          rankIndex={index}
          onSelectProduct={onSelectProduct}
          onExplain={onExplain}
        />
      ))}
    </div>
  );
};
