# Frontend Application — Next.js & React

A modern, research-grade user interface for semantic product discovery, multi-stage retrieval inspection, and explainable recommendations.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom HSL design system tokens
- **Animations**: Framer Motion
- **Icons**: Lucide React

## Routes

| Route | Purpose |
| --- | --- |
| `/` | Landing page — hero, search entry point, architecture highlights |
| `/dashboard` | Legacy route redirecting to `/` (backward compatibility) |
| `/search` | Live semantic search (`POST /api/search`), query understanding, ML signals, grounded explanations. Supports `?q=` |
| `/recommendations` | Live recommendations (`POST /api/recommend`) by anchor ASIN or user history |
| `/evaluation` | Live offline benchmark results (`GET /api/metrics`, `GET /api/evaluate/experiments`) |
| `/about` | Architecture overview and technology stack |

## Design Principles

- **Original Visual Identity**: Inspired by modern AI research tools, crisp typography, balanced whitespace, and deep slate/charcoal tones.
- **Deep Observability**: Expandable "ML Signals" panels on each result showing dense retrieval rank/score and cross-encoder rerank rank/score, sourced only from what the backend actually returns.
- **Explainability**: "Why this result?" / "Why recommended?" drawers rendering the Phase 10 grounded explanation (verified reasons + explicit warnings for unsupported attributes) — never a fabricated claim.
- **No fabricated data**: benchmark numbers on `/evaluation` are read live from the backend, never hardcoded in the frontend.

## Development

```bash
cd frontend
npm install
cp .env.example .env.local   # adjust NEXT_PUBLIC_API_URL if needed
npm run dev
```

The frontend will run on `http://localhost:3000` (or `http://localhost:3001` when 3000 is occupied) and communicate with the FastAPI backend on `http://localhost:8000` (start it separately: `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` from the `backend` directory).

## Validation

```bash
npm run lint
npx tsc --noEmit
npm run build
```
