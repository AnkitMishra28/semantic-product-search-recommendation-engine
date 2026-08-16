/**
 * Frontend TypeScript Domain & API Interfaces matching FastAPI backend schemas.
 *
 * Kept in sync with:
 *   backend/app/models/product.py
 *   backend/app/models/search.py
 *   backend/app/models/recommendation.py
 *   backend/app/models/explanation.py
 *   backend/app/api/v1/*.py
 */

export interface Product {
  asin: string;
  parent_asin?: string | null;
  title: string;
  description?: string;
  features: string[];
  price?: number | null;
  brand?: string | null;
  categories: string[];
  average_rating?: number | null;
  rating_number?: number;
  rating_count: number;
  image_url?: string | null;
  images?: string[];
  bought_together: string[];
  metadata?: Record<string, any>;
}

export interface QueryUnderstandingResult {
  raw_query: string;
  normalized_query: string;
  intent: string;
  category?: string | null;
  brand?: string | null;
  price_min?: number | null;
  price_max?: number | null;
  currency?: string;
  attributes?: Record<string, string[]>;
  detected_entities: {
    brands?: string[];
    categories?: string[];
    modifiers?: string[];
    [key: string]: string[] | undefined;
  };
  expanded_queries: string[];
  hard_filters?: Record<string, any>;
  soft_signals?: Record<string, any>;
  confidence?: number;
}

export interface RetrievalSignal {
  stage: string;
  initial_score: number;
  initial_rank: number;
}

export interface RerankSignal {
  stage: string;
  rerank_score: number;
  rerank_rank: number;
}

export interface PipelineStageTiming {
  query_understanding_ms: number;
  dense_retrieval_ms: number;
  cross_encoder_rerank_ms: number;
  business_ranking_ms: number;
  explanation_generation_ms: number;
  total_latency_ms: number;
}

/** A single structured, evidence-grounded reason supporting a match. */
export interface ExplanationReason {
  type: string;
  label: string;
  text: string;
  evidence: string;
  is_matched: boolean;
}

/** Full structured grounded explanation (Phase 10 GroundedExplainer output). */
export interface GroundedExplanation {
  product_id: string;
  summary: string;
  reasons: ExplanationReason[];
  semantic_match_score?: number | null;
  grounded: boolean;
  warnings: string[];
  generation_method: string;
}

export interface SearchResultItem {
  product: Product;
  final_score: number;
  retrieval_signal?: RetrievalSignal | null;
  rerank_signal?: RerankSignal | null;
  explanation?: string | null;
  grounded_explanation?: GroundedExplanation | null;
}

export interface ProductFilter {
  min_price?: number | null;
  max_price?: number | null;
  brand?: string | null;
  categories?: string[] | null;
  min_rating?: number | null;
}

export type RankingStrategy = "cross_encoder" | "hybrid" | "dense_only";

export interface SearchRequest {
  query: string;
  top_k_retrieval?: number;
  top_k_reranking?: number;
  filters?: ProductFilter | null;
  enable_reranking?: boolean;
  enable_explanation?: boolean;
  ranking_strategy?: RankingStrategy;
}

export interface SearchResponse {
  query: string;
  query_understanding: QueryUnderstandingResult;
  total_retrieved: number;
  total_returned: number;
  results: SearchResultItem[];
  timings: PipelineStageTiming;
}

/** Legacy structured rationale attached inline to a RecommendationItem. */
export interface ExplanationItem {
  summary: string;
  key_features_matched: string[];
  shared_categories: string[];
  confidence?: number | null;
}

export type RecommendationType =
  | "popularity"
  | "content_based"
  | "collaborative"
  | "collaborative_bought_together"
  | "hybrid"
  | "hybrid_mmr"
  | string;

export interface RecommendationItem {
  product: Product;
  score: number;
  recommendation_type: RecommendationType;
  signals?: Record<string, number>;
  reasons?: string[];
  explanation?: ExplanationItem | null;
  grounded_explanation?: GroundedExplanation | null;
}

export type RecommendationStrategy = "popularity" | "content" | "collaborative" | "hybrid" | "hybrid_mmr" | string;

export interface RecommendRequest {
  user_id?: string;
  asin?: string;
  user_history_asins?: string[];
  top_k?: number;
  strategy?: RecommendationStrategy;
  filters?: Record<string, any> | null;
  lambda_diversity?: number | null;
  generate_explanations?: boolean;
}

export interface RecommendResponse {
  user_id?: string | null;
  anchor_asin?: string | null;
  strategy: string;
  total_returned: number;
  recommendations: RecommendationItem[];
  execution_time_ms: number;
}

export interface ExplainRequest {
  query?: string;
  product_id?: string;
  asin?: string;
  anchor_product_id?: string;
  user_id?: string;
  strategy?: string;
}

export interface SimpleExplanationResponse {
  asin: string;
  explanation: string;
}

/** GET /api/v1/ready response shape (backend/app/api/v1/health.py). */
export interface SystemReadiness {
  status: "ready" | "initializing" | string;
  app_name: string;
  version: string;
  components: {
    model_registry: string;
    catalog: {
      status: string;
      product_count: number;
    };
    vector_index: {
      status: string;
      backend: string;
      indexed_documents: number;
    };
    cross_encoder: {
      status: string;
      model: string;
      device: string;
    };
    llm_explainer: {
      status: string;
      mode: string;
      model: string;
    };
  };
}

export interface HealthStatus {
  status: string;
  app_name: string;
  version: string;
  environment: string;
}

/**
 * GET /api/v1/metrics response — a passthrough of authoritative offline
 * benchmark artifacts (experiments results.json files). Shape is
 * intentionally loose: the frontend must render whatever keys are actually
 * present and never assume a field exists.
 */
export interface MetricsPayload {
  status?: string;
  benchmark_provenance?: string;
  scientific_notes?: Record<string, string>;
  hybrid_retrieval?: Record<string, any>;
  cross_encoder_reranking?: Record<string, any>;
  recommendation_engine?: Record<string, any>;
  [key: string]: any;
}

/** GET /api/v1/evaluate/experiments list item. */
export interface ExperimentSummary {
  filename: string;
  experiment_id?: string | null;
  timestamp?: string | null;
  track?: string | null;
  dataset?: string | null;
  metrics_summary?: Record<string, any> | null;
  latency?: Record<string, any> | null;
  file_size_bytes?: number;
}

export type ExperimentDetail = Record<string, any>;

/** Normalized shape used by the API client / UI error states. */
export interface ApiErrorInfo {
  kind: "network" | "timeout" | "http";
  status?: number;
  message: string;
}

