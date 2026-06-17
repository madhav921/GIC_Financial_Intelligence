# GIC Financial Intelligence — React Frontend

React 18 dashboard for the GIC Plan-to-Perform Engine.

## Quick Start

```bash
cd frontend
npm install
npm start          # Opens http://localhost:3000
```

## Backend Connection

The dashboard works standalone with mock data. To connect real data:

```bash
# Start FastAPI backend (from repo root)
uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# The React proxy (package.json "proxy") forwards /api calls to :8000
```

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/executive` | Executive Summary | CFO KPIs, alerts, segment revenue |
| `/commodity` | Commodity Intelligence | Price charts, BOM weights, FFN metrics |
| `/pnl` | Financial P&L | Waterfall chart, segment analysis, sensitivity |
| `/simulation` | Scenario Simulation | Monte Carlo what-if builder, 7 presets |
| `/market` | Market Monitor | Live indices, FX, crypto, FRED macro |
| `/governance` | Governance & LLM | Audit trail, LLM narratives, bias tracking |
| `/data` | Data Explorer | Dataset catalog, quality report |

## Tech Stack

- **React 18** + React Router v6
- **Recharts** — charts (BarChart, AreaChart, LineChart)
- **Tailwind CSS** — via CDN in `public/index.html`
- **Axios** — API client with mock fallback

## LLM Backend (Governance page)

The Governance page calls the backend which uses an open-source LLM:

| Backend | Model | Requirement |
|---------|-------|-------------|
| Ollama | llama3.2:1b | `ollama pull llama3.2:1b` |
| HuggingFace | google/flan-t5-base | `pip install transformers torch` |
| Template | Built-in | Always available |

## Build for Production

```bash
npm run build      # Creates frontend/build/
```

Serve with: `npx serve -s build -l 3000`
