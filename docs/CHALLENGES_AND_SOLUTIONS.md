# GIC Platform — Challenges & Solutions

A concise record of the real technical and architectural decisions made during the build. Drawn from git history, commit messages, and source diffs.

---

## Data Sourcing

| # | Challenge | Root Cause | Solution |
|---|-----------|-----------|---------|
| D1 | No direct exchange API for Steel, Aluminum, Cobalt, Nickel, Lithium | These commodities have no free liquid futures API | Used ETF/equity proxies via yfinance (SLX×7.5 for Steel, AA×60 for Aluminum, GLNCY×1600 for Cobalt, VALE×750 for Nickel, LIT×0.25 for Lithium); scale factors calibrated to LME equivalents |
| D2 | No exchange instrument at all for Rhodium, Polypropylene, ABS Resin | OTC-only or no public pricing | Modelled via Ornstein-Uhlenbeck mean-reverting process calibrated to LPPM (Rhodium) and ICIS (PP, ABS) indices |
| D3 | SSL/TLS certificate errors when calling yfinance | Corporate MITM proxy or stale OS cert store | Built 4-tier session strategy: `truststore` (OS cert store) → `certifi` bundle → Python default → `SSL_VERIFY=false` env var (last resort); session reused globally |
| D4 | Realtime WebSocket ticker showing fictional prices (Steel 490, Lithium 4.0) | `_seed_prices()` was reading synthetic CSV defaults instead of real data | Changed seeder to load from `MarketDataProvider` (data/raw/ → yfinance live → synthetic fallback); mean-revert anchor now tracks real market levels |
| D5 | Commodity Intelligence chart history and forecast on different scales | History seeded from synthetic CSV; forecast used real yfinance prices (e.g., Copper history ended at $8,500, forecast started at $13,900) | Unified both through `MarketDataProvider`; added `/forecast/commodity-history` endpoint returning real 36-month history |
| D6 | Lithium unit inconsistency | Synthetic data stored Lithium in USD/t (~18,000); real LIT ETF proxy scaled to USD/kg (~21); BOM calculations mixed units | Added explicit `unit` field per commodity in both settings.yaml and frontend; scale factor (×0.25) now converts LIT ETF price to USD/kg consistently |

---

## Financial Model Accuracy

| # | Challenge | Root Cause | Solution |
|---|-----------|-----------|---------|
| F1 | P&L showing negative gross margins (COGS > revenue) | `CostDrivers` used since-2019 cumulative commodity index change as the COGS driver — 6 years of compounded inflation made COGS mathematically exceed revenue | Switched to YoY (12-month lag) change: `commodity_index / commodity_index.shift(12) - 1`; this captures the real annual headwind a procurement team faces |
| F2 | `/pnl/annual` returning stale or wrong year data | Route used `tail(12)` rows on `annual_df`, but `annual_df` aggregates by (year, segment) — `tail(12)` was returning 12 year-segment rows, not 12 months | Filtered to `most_recent_year = annual_df['year'].max()` instead |
| F3 | Monte Carlo variance decomposition inflating VaR by ~48× | `annual_mean_oi` was multiplied by `len(base_pnl)` (which was 48 = 4 segments × 12 months) instead of used directly | Fixed the formula; also added a pre-aggregation step to collapse month-segment rows to one-row-per-month before passing to Monte Carlo |
| F4 | `/simulation/monthly-fan` returning nonsensical percentile fans | `pnl_df` contained one row per (month, segment) combination; Monte Carlo treated each row as an independent time point | Added `pnl_df.groupby('date').sum()` before calling `run_monthly_fan` |
| F5 | Scenario simulations inflated by 7-year cumulative revenue | `/simulation/variance-decomposition`, `/simulation/scenario`, and `compare-presets` were using the full 7-year `sales_df` | Added `sales_df[sales_df['year'] == most_recent_year]` filter to all three endpoints |
| F6 | Pipeline CV failing for Natural Gas, Polypropylene, ABS Resin (3/12 commodities) | Date column dtype mismatch: CSV loader produced `str`, `reset_index()` produced `datetime64`; inner merge on mismatched types produced empty DataFrames, causing XGBoost CV to error | Added `pd.to_datetime()` coercion on both frames before merge in `prepare_commodity_features` |

---

## Deployment (Vercel)

| # | Challenge | Root Cause | Solution |
|---|-----------|-----------|---------|
| V1 | Vercel build failing: bundle exceeded 1 GB free limit | All heavy deps (polars, pyarrow, scipy, statsmodels, ccxt, yfinance, streamlit, matplotlib) were in the main install | Split `pyproject.toml`: slim base (~300 MB, core FastAPI stack only) + `[full]` optional group for local/Docker; added `.vercelignore` to exclude `data/`, `models/`, `docs/`, `tests/`, `frontend/` from the serverless bundle |
| V2 | API crashed at cold start on Vercel: filesystem is read-only | `AuditTrail`, `setup_logging`, and `auth/store.py` all attempted to create directories at import time | Each now wraps the write attempt in `try/except OSError` and falls back to `/tmp/gic_audit`, `/tmp/gic_users.json`, etc. No behaviour change in local/Docker |
| V3 | Single bad route import crashed the entire API | FastAPI raises at startup if any `include_router` import fails; a missing optional dep (e.g., `polars`) took down all 27 routes | Wrapped all 8 route imports in `try/except`; failures stored in `_IMPORT_ERRORS` dict; exposed via `/_debug/imports` endpoint for diagnosis |
| V4 | `ModelRegistry` failing on Vercel: `models/saved/` not in bundle and not writable | Bundle size constraints forced exclusion of `models/saved/`; the registry tried to write there at init | Added `/tmp/gic_models` fallback path when `models/saved/` is not writable |
| V5 | `.vercelignore` patterns not anchoring correctly | Patterns without a leading `/` were treated as relative and matched nothing | Prefixed all ignore patterns with `/` (e.g., `/data/`, `/models/`) to anchor them to the project root |

---

## Architecture Decisions

| # | Decision | Why | Trade-off |
|---|----------|-----|-----------|
| A1 | Dropped Streamlit; rebuilt frontend as React 18 SPA | Streamlit couldn't support RBAC, WebSocket real-time feed, custom layout, or Vercel deployment cleanly | React adds build complexity, but gives full control over UI, auth, and live data patterns |
| A2 | Introduced `MarketDataProvider` as a single source of truth with 1-hour in-memory cache | Forecast routes, realtime feed, and P&L model were each independently fetching yfinance — causing rate limits and price inconsistency between pages | Cache adds a 1-hour staleness window; mitigated by `/realtime/refresh` endpoint that force-expires the cache |
| A3 | 5-layer architecture (`layers/`) wired by `GICOrchestrator` | Needed clear separation between data (L1), ML (L2), financial model (L3), simulation (L4), and governance/LLM (L5) for auditability and independent testing | Added controller indirection layer; initial alignment between controllers and actual module APIs required a fix pass after restructure |
| A4 | Open-source LLM for governance layer (no commercial API) | No API budget; needed narrative generation to work offline and in serverless | 3-tier fallback: Ollama (llama3.2:1b, best quality) → HuggingFace `flan-t5-base` (CPU, ~300 MB) → deterministic template (always available); quality degrades gracefully |
| A5 | ETF/equity proxies for commodity prices rather than Bloomberg/Refinitiv | No data vendor budget; yfinance is free but covers equities/ETFs, not LME physical spot | Prices carry equity risk premium and diverge from LME spot in stress events; scale factors require monthly recalibration; risk disclosed in data transparency section |
| A6 | SARIMAX + XGBoost ensemble (preferred per commodity) rather than a single model | Each commodity has different dominant dynamics (seasonal, macro-driven, trend); no single model type wins across all 12 | Adds model selection complexity; preferred model chosen by CV MAPE per commodity (see forecast accuracy table) |
| A7 | Synthetic data fallback (`data/synthetic/`) for offline/demo mode | `data/raw/` is gitignored; new machines have no data until `fetch_data.py` runs; Vercel serverless has no persistent storage | Risk of demo mode silently showing stale synthetic numbers; mitigated by "data source" indicator in UI and backend connect banner |

---

## Integration & Wiring

| # | Challenge | Root Cause | Solution |
|---|-----------|-----------|---------|
| I1 | Layer controllers broke orchestrator after restructure | Controllers were scaffolded with class/method names that didn't exist in the real `src/` modules | Aligned all 5 layer controllers with actual module APIs: L1 uses module functions (not a class), L4 `HedgeOptimizer` takes `futures_price/exposure_units`, L5 `BiasTracker.compute_bias` takes `pd.Series` |
| I2 | `comparePresets` returning 405 Method Not Allowed | `client.js` was calling `POST /simulation/compare-presets`; backend route is `GET` | Changed client method to `GET`; audited all client methods against backend route definitions |
| I3 | ESLint unknown rule comment breaking production build | A `// eslint-disable-next-line` comment referenced a rule that doesn't exist in the project's ESLint config | Removed the comment; build passed |
| I4 | `top_drivers` from `/insights/early-warning` causing runtime crash in UI | Backend returns `[{component, contribution, score}]` objects; frontend was treating them as plain strings | Added type guard: `typeof dr === 'string' ? dr : dr?.component` before rendering |
| I5 | Frontend P&L values displaying in USD instead of GBP | Backend returns all financial values in USD (config uses `avg_price_usd`); no conversion in original frontend code | Added `/ 1.27` divisor on all currency displays; documented as a single source of truth in CLAUDE.md |
| I6 | Commodity name mismatch between frontend and backend API calls | Frontend used "Natural Gas", "ABS Resin"; backend CSV columns are "Natural_Gas", "ABS_Resin" | Added `.replace(/ /g, '_')` transform in API client before commodity forecast calls |
| I7 | `AuditTrail` catching only `OSError` but Vercel raised `Exception` | Narrow exception catch missed permission errors on Vercel that weren't `OSError` subclasses | Broadened catch to `except Exception` before the `/tmp` fallback |

---

*Last updated: 17 June 2026 · Reconstructed from git history (`git log --oneline`, `git show` diffs)*
