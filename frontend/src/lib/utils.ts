import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge CSS classes cleanly with Tailwind support.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format currency numbers in USD.
 */
export function formatCurrency(amount?: number | null): string {
  if (amount === undefined || amount === null) return "N/A";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(amount);
}

/**
 * Format review counts cleanly (e.g. 1.2k, 45k).
 */
export function formatCount(count: number): string {
  if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return count.toString();
}

/**
 * Format a currency amount using an explicit ISO currency code — used for
 * query-understanding price constraints, which may be extracted in a
 * currency other than the catalog's native USD (e.g. "under ₹80,000").
 */
export function formatCurrencyCode(amount: number, currencyCode: string = "USD"): string {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currencyCode,
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `${amount} ${currencyCode}`;
  }
}

/**
 * Format a 0-1 fraction as a percentage string, e.g. 0.1958 -> "19.58%".
 */
export function formatPercent(value?: number | null, digits: number = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * Format a millisecond duration for display, switching to seconds above 1000ms.
 */
export function formatMs(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${value.toFixed(1)}ms`;
}

/**
 * Format a plain metric/score number with fixed precision, guarding against
 * missing values so the UI never silently renders "undefined".
 */
export function formatNumber(value?: number | null, digits: number = 4): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

/**
 * Convert a snake_case identifier into a readable label, e.g.
 * "cross_encoder_rerank_ms" -> "Cross Encoder Rerank Ms".
 */
export function humanizeLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
