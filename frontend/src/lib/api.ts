/**
 * Typed API Client for the FastAPI Search & Recommendation Backend.
 *
 * All requests go through /api/v1/* on the real backend — no mock or
 * duplicated ML logic lives here. Every method surfaces a normalized
 * ApiError so callers can render loading / error / empty states
 * consistently across pages.
 */

import {
  ExperimentSummary,
  ExplainRequest,
  GroundedExplanation,
  HealthStatus,
  MetricsPayload,
  Product,
  RecommendRequest,
  RecommendResponse,
  SearchRequest,
  SearchResponse,
  SimpleExplanationResponse,
  SystemReadiness,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 30000;

export type ApiErrorKind = "network" | "timeout" | "http";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number;
  details?: unknown;

  constructor(kind: ApiErrorKind, message: string, status?: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.details = details;
  }
}

async function parseBackendError(res: Response): Promise<ApiError> {
  let message = res.statusText || `Request failed with status ${res.status}`;
  let details: unknown;
  try {
    const body = await res.json();
    details = body;
    if (typeof body?.message === "string") {
      message = body.message;
    } else if (typeof body?.detail === "string") {
      message = body.detail;
    }
  } catch {
    // Response body was not JSON — keep the default status text message.
  }
  return new ApiError("http", message, res.status, details);
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers || {}) },
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("timeout", `Request to ${path} timed out after ${timeoutMs}ms.`);
    }
    throw new ApiError(
      "network",
      `Unable to reach the backend at ${API_BASE_URL}. Is the FastAPI server running?`
    );
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    throw await parseBackendError(res);
  }

  return res.json() as Promise<T>;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /** GET /api/v1/health — basic liveness check. */
  async getHealth(): Promise<HealthStatus> {
    return request<HealthStatus>("/api/v1/health", { method: "GET" });
  }

  /** GET /api/v1/ready — model/index/catalog readiness. */
  async getReadiness(): Promise<SystemReadiness> {
    return request<SystemReadiness>("/api/v1/ready", { method: "GET" });
  }

  /** POST /api/v1/search — execute multi-stage semantic product search. */
  async search(req: SearchRequest): Promise<SearchResponse> {
    return request<SearchResponse>("/api/v1/search", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  /** POST /api/v1/recommend — item-to-item or personalized recommendations. */
  async recommend(req: RecommendRequest): Promise<RecommendResponse> {
    return request<RecommendResponse>("/api/v1/recommend", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  /** POST /api/v1/explain — full structured grounded explanation. */
  async explain(req: ExplainRequest): Promise<GroundedExplanation> {
    return request<GroundedExplanation>("/api/v1/explain", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  /** POST /api/v1/explain/search — legacy simple string explanation. */
  async explainSearch(query: string, asin: string): Promise<SimpleExplanationResponse> {
    return request<SimpleExplanationResponse>("/api/v1/explain/search", {
      method: "POST",
      body: JSON.stringify({ query, asin }),
    });
  }

  /** GET /api/v1/products/{id} — verified catalog product lookup. */
  async getProduct(productId: string): Promise<Product> {
    return request<Product>(`/api/v1/products/${encodeURIComponent(productId)}`, { method: "GET" });
  }

  /** GET /api/v1/metrics — authoritative offline benchmark artifacts. */
  async getMetrics(): Promise<MetricsPayload> {
    return request<MetricsPayload>("/api/v1/metrics", { method: "GET" });
  }

  /** GET /api/v1/evaluate/experiments — tracked experiment registry. */
  async getExperiments(): Promise<ExperimentSummary[]> {
    return request<ExperimentSummary[]>("/api/v1/evaluate/experiments", { method: "GET" });
  }

  /** GET /api/v1/evaluate/experiments/{id} — full raw experiment JSON artifact. */
  async getExperimentDetail(idOrFilename: string): Promise<Record<string, any>> {
    return request<Record<string, any>>(`/api/v1/evaluate/experiments/${encodeURIComponent(idOrFilename)}`, {
      method: "GET",
    });
  }
}

export const apiClient = new ApiClient();

