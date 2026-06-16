"""Real-time market data-capture routes (REST snapshot + WebSocket live tape).

Prices are seeded from the same data source as the forecast models:
  yfinance live pull  (market_data_source=live, always tried first)
  data/raw/commodity_prices.csv  (Yahoo Finance, cached from previous pull)
  data/synthetic/commodity_prices.csv                                         → last resort

FX rates are seeded from Yahoo Finance at startup (GBPUSD=X, EURUSD=X, USDCNY=X).
The random walk uses very small per-tick volatility:
  FX:         vol=0.00008  (~0.008% per 2s tick — realistic micro-movement)
  Commodities: vol=0.0008  (~0.08% per 2s tick — subtle intraday drift)
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.config import get_settings

realtime_router = APIRouter(tags=["realtime"])


# ── Commodity metadata ─────────────────────────────────────────────────────────

_TOP_COMMODITY_NAMES: list[str] = [
    "Steel", "Lithium", "Aluminum", "Copper", "Cobalt", "Nickel",
]

_COMMODITY_UNITS: dict[str, str] = {
    "Steel":         "USD/t",
    "Lithium":       "USD/kg",
    "Aluminum":      "USD/t",
    "Copper":        "USD/t",
    "Cobalt":        "USD/t",
    "Nickel":        "USD/t",
    "Platinum":      "USD/oz",
    "Palladium":     "USD/oz",
    "Natural_Gas":   "USD/MMBtu",
    "Polypropylene": "USD/t",
    "ABS_Resin":     "USD/t",
}

# Used ONLY when both yfinance and data/raw/ fail.
_FALLBACK_PRICES: dict[str, float] = {
    "Steel":         790.0,
    "Lithium":        21.0,
    "Aluminum":     3735.0,
    "Copper":      13900.0,
    "Cobalt":      24700.0,
    "Nickel":      12250.0,
    "Platinum":     1970.0,
    "Palladium":    1420.0,
    "Natural_Gas":    30.0,
    "Polypropylene": 940.0,
    "ABS_Resin":    1440.0,
}

_DEFAULT_FX: dict[str, float] = {
    "GBP/USD": 1.2750,
    "EUR/USD": 1.1050,
    "USD/CNY": 7.1500,
}

# ── Module-level caches ────────────────────────────────────────────────────────

_fx_cache: dict[str, float] | None = None
_fx_cache_ts: float = 0.0
_FX_CACHE_TTL: float = 3600.0  # 1 hour

_market_indices_cache: dict | None = None
_market_indices_ts: float = 0.0
_MARKET_INDICES_TTL: float = 14400.0  # 4 hours

_fx_history_cache: dict | None = None
_fx_history_ts: float = 0.0
_FX_HISTORY_TTL: float = 3600.0  # 1 hour

_HEADLINES: list[str] = [
    "Lithium softening — window to extend battery-material hedge at favourable rates.",
    "Natural-gas volatility elevated; energy-intensive smelting costs at risk this quarter.",
    "EV warranty accrual trending above plan — review provisioning on new battery packs.",
    "GBP/USD drift widening unhedged FX exposure on USD-denominated raw materials.",
    "Steel prices dipping below trend — opportunity for a strategic pre-buy.",
    "Gross-margin watch: commodity index above 105 pressuring driver-based EBIT nowcast.",
]

_RISK_BANDS = (
    (0.0, 25.0, "low"),
    (25.0, 50.0, "elevated"),
    (50.0, 75.0, "high"),
    (75.0, 100.1, "critical"),
)


def _risk_band(score: float) -> str:
    for lo, hi, name in _RISK_BANDS:
        if lo <= score < hi:
            return name
    return "critical"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data seeding ───────────────────────────────────────────────────────────────


def _seed_prices_from_real_data() -> dict[str, float]:
    """Load latest commodity prices (MarketDataProvider → _FALLBACK_PRICES)."""
    try:
        from src.data.market_data_provider import get_latest_prices
        latest = get_latest_prices()
        if latest:
            prices = {name: latest[name] for name in _FALLBACK_PRICES if name in latest}
            for m in _FALLBACK_PRICES:
                if m not in prices:
                    prices[m] = _FALLBACK_PRICES[m]
            logger.info(
                "realtime: seeded from MarketDataProvider — "
                f"Steel={prices.get('Steel'):.0f}, Copper={prices.get('Copper'):.0f}, "
                f"Lithium={prices.get('Lithium'):.2f}"
            )
            return prices
    except Exception as exc:
        logger.warning(f"realtime: MarketDataProvider seed failed ({exc}); using fallback")
    return dict(_FALLBACK_PRICES)


def _seed_fx_from_real_data() -> dict[str, float]:
    """Fetch current FX rates from Yahoo Finance (1-hour cache).

    Returns dict of {"GBP/USD": 1.27, "EUR/USD": 1.10, "USD/CNY": 7.15}.
    Falls back to _DEFAULT_FX if yfinance is unavailable.
    """
    global _fx_cache, _fx_cache_ts

    if _fx_cache and (time.time() - _fx_cache_ts) < _FX_CACHE_TTL:
        return dict(_fx_cache)

    try:
        import yfinance as yf
        tickers = {"GBP/USD": "GBPUSD=X", "EUR/USD": "EURUSD=X", "USD/CNY": "USDCNY=X"}
        fx: dict[str, float] = {}
        for pair, ticker in tickers.items():
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                fx[pair] = round(float(hist["Close"].iloc[-1]), 4)
        if len(fx) == len(tickers):
            _fx_cache = dict(fx)
            _fx_cache_ts = time.time()
            logger.info(
                f"realtime: FX from Yahoo Finance — "
                f"GBP/USD={fx['GBP/USD']}, EUR/USD={fx['EUR/USD']}, USD/CNY={fx['USD/CNY']}"
            )
            return fx
    except Exception as exc:
        logger.warning(f"realtime: FX seed from Yahoo Finance failed ({exc}); using defaults")

    return dict(_DEFAULT_FX)


# ── Market feed ────────────────────────────────────────────────────────────────


class MarketFeed:
    """Stateful market feed with realistic mean-reverting micro-movements.

    Commodity vol=0.0008 per 2s tick (~0.08%/tick, ~3%/day annualised),
    FX vol=0.00008 per 2s tick (~0.008%/tick — 4th decimal fluctuates subtly).
    Both mean-revert to real Yahoo Finance anchor prices.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._tick_count = 0

        try:
            self._settings = get_settings()
        except Exception:
            self._settings = {}

        # Commodity prices seeded from real data.
        self._anchor: dict[str, float] = _seed_prices_from_real_data()
        self._prices: dict[str, float] = dict(self._anchor)
        self._prev_prices: dict[str, float] = dict(self._prices)

        # FX rates seeded from Yahoo Finance.
        self._anchor_fx: dict[str, float] = _seed_fx_from_real_data()
        self._fx: dict[str, float] = dict(self._anchor_fx)
        self._prev_fx: dict[str, float] = dict(self._fx)

        self._index: float = 100.0
        self._prev_index: float = 100.0
        self._risk: float = 42.0
        self._headline_idx: int = 0

    def _walk(self, value: float, anchor: float, *, vol: float, reversion: float) -> float:
        """One mean-reverting Ornstein-Uhlenbeck step. Returns strictly positive value."""
        shock = self._rng.gauss(0.0, vol)
        pull = reversion * (anchor - value) / (anchor or 1.0)
        new_value = value * (1.0 + shock) + pull * value
        return max(new_value, anchor * 0.05, 1e-9)

    @staticmethod
    def _pct(curr: float, prev: float) -> float:
        if prev == 0:
            return 0.0
        return round((curr - prev) / prev * 100.0, 3)

    def _ebit_nowcast(self) -> float:
        base = 3.75e9
        adjustment = (100.0 - self._index) / 20.0 * 0.15e9
        jitter = self._rng.gauss(0.0, 0.02e9)
        return round(_clamp(base + adjustment + jitter, 3.3e9, 4.2e9), 0)

    def next_tick(self) -> dict[str, Any]:
        self._tick_count += 1
        self._prev_prices = dict(self._prices)
        self._prev_fx = dict(self._fx)
        self._prev_index = self._index

        # Commodity walk — ~0.08% per tick, subtle intraday drift.
        for name in self._prices:
            anchor = self._anchor.get(name, self._prices[name])
            self._prices[name] = self._walk(
                self._prices[name], anchor, vol=0.0008, reversion=0.008
            )

        # FX walk — ~0.008% per tick, 4th decimal changes slowly.
        for pair in self._fx:
            anchor = self._anchor_fx.get(pair, _DEFAULT_FX.get(pair, self._fx[pair]))
            self._fx[pair] = self._walk(self._fx[pair], anchor, vol=0.00008, reversion=0.01)

        self._index = _clamp(
            self._walk(self._index, 100.0, vol=0.003, reversion=0.04), 92.0, 118.0
        )

        index_pressure = (self._index - 100.0) * 0.3
        self._risk = _clamp(
            self._risk + self._rng.gauss(index_pressure * 0.04, 0.6), 0.0, 100.0
        )

        if self._tick_count % 10 == 0:
            self._headline_idx = (self._headline_idx + 1) % len(_HEADLINES)

        return self._build_snapshot()

    def snapshot(self) -> dict[str, Any]:
        return self._build_snapshot()

    def _build_snapshot(self) -> dict[str, Any]:
        top_commodities = [
            {
                "name": name,
                "price": round(self._prices[name], 2),
                "change_pct": self._pct(
                    self._prices[name],
                    self._prev_prices.get(name, self._prices[name]),
                ),
                "unit": _COMMODITY_UNITS.get(name, "USD"),
            }
            for name in _TOP_COMMODITY_NAMES
            if name in self._prices
        ]

        fx = [
            {
                "pair": pair,
                "rate": round(self._fx[pair], 4),
                "change_pct": self._pct(self._fx[pair], self._prev_fx.get(pair, self._fx[pair])),
            }
            for pair in self._anchor_fx
        ]

        risk_score = round(self._risk, 1)
        active_alerts = int(risk_score // 20) + (1 if self._index > 107 else 0)

        return {
            "timestamp": _now_iso(),
            "commodity_index": round(self._index, 2),
            "commodity_index_change_pct": self._pct(self._index, self._prev_index),
            "risk_score": risk_score,
            "risk_band": _risk_band(risk_score),
            "top_commodities": top_commodities,
            "fx": fx,
            "ebit_nowcast_gbp": self._ebit_nowcast(),
            "headline_insight": _HEADLINES[self._headline_idx],
            "active_alerts": active_alerts,
            "data_source": "Yahoo Finance" if _is_real_data() else "Synthetic",
        }


def _is_real_data() -> bool:
    """True when real Yahoo Finance data is active (live pull or cached CSV)."""
    try:
        from src.data.market_data_provider import get_data_source_label
        label = get_data_source_label()
        return "Yahoo Finance" in label or "Live" in label
    except Exception:
        try:
            from src.config import get_project_root
            return (get_project_root() / "data" / "raw" / "commodity_prices.csv").exists()
        except Exception:
            return False


# ── REST endpoints ─────────────────────────────────────────────────────────────


@realtime_router.get("/realtime/snapshot")
async def get_snapshot() -> dict[str, Any]:
    """Current market snapshot (initial state for the UI)."""
    feed = MarketFeed()
    return feed.snapshot()


@realtime_router.post("/realtime/refresh")
async def refresh_market_data() -> dict[str, Any]:
    """Force-refresh the commodity price cache."""
    global _fx_cache, _fx_cache_ts, _market_indices_cache, _market_indices_ts
    try:
        from src.data.market_data_provider import invalidate_cache, get_latest_prices
        invalidate_cache()
        # Also bust FX and market-indices caches
        _fx_cache = None
        _fx_cache_ts = 0.0
        _market_indices_cache = None
        _market_indices_ts = 0.0
        latest = get_latest_prices()
        return {
            "status": "refreshed",
            "source": _get_data_source_label(),
            "latest_prices": latest,
            "commodities": len(latest),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _get_data_source_label() -> str:
    try:
        from src.data.market_data_provider import get_data_source_label as _label
        return _label()
    except Exception:
        return "Unknown"


@realtime_router.get("/realtime/market-indices")
async def get_market_indices() -> dict[str, Any]:
    """Fetch current market indices from Yahoo Finance (Gold, S&P 500, VIX, Oil, 10Y Yield).

    Cached 4 hours. Returns previous-close prices.
    Falls back to approximate 2026 reference values if Yahoo Finance is unavailable.
    """
    global _market_indices_cache, _market_indices_ts

    now = time.time()
    if _market_indices_cache and (now - _market_indices_ts) < _MARKET_INDICES_TTL:
        return _market_indices_cache

    import pandas as _pd

    _INDICES_META: dict[str, dict] = {
        "GC=F":  {"name": "Gold",      "currency": "$", "decimals": 0},
        "^GSPC": {"name": "S&P 500",   "currency": "",  "decimals": 2},
        "^VIX":  {"name": "VIX",       "currency": "",  "decimals": 2},
        "CL=F":  {"name": "Oil (WTI)", "currency": "$", "decimals": 2},
        "^TNX":  {"name": "10Y Yield", "currency": "%", "decimals": 2},
    }

    try:
        import yfinance as yf
        ticker_list = list(_INDICES_META.keys())
        raw = yf.download(ticker_list, period="5d", interval="1d", progress=False)

        if not raw.empty:
            close = raw["Close"] if isinstance(raw.columns, _pd.MultiIndex) else raw
            indices = []
            for ticker, meta in _INDICES_META.items():
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if len(series) >= 1:
                        latest = float(series.iloc[-1])
                        prev = float(series.iloc[-2]) if len(series) >= 2 else latest
                        change = (latest - prev) / prev * 100.0 if prev else 0.0
                        indices.append({
                            "name": meta["name"],
                            "ticker": ticker,
                            "value": round(latest, meta["decimals"]),
                            "change_pct": round(change, 2),
                            "currency": meta["currency"],
                            "as_of": str(series.index[-1])[:10],
                        })

            if indices:
                result: dict[str, Any] = {
                    "indices": indices,
                    "data_source": "Yahoo Finance",
                    "note": "Previous close · Gold=GC=F futures · Oil=CL=F WTI futures",
                    "as_of": _now_iso(),
                }
                _market_indices_cache = result
                _market_indices_ts = now
                logger.info(f"market-indices: fetched {len(indices)} from Yahoo Finance")
                return result
    except Exception as exc:
        logger.warning(f"market-indices: Yahoo Finance fetch failed ({exc}); returning fallback")

    fallback: dict[str, Any] = {
        "indices": [
            {"name": "Gold",      "ticker": "GC=F",  "value": 3250.0, "change_pct": 0.0, "currency": "$", "as_of": "reference"},
            {"name": "S&P 500",   "ticker": "^GSPC", "value": 5800.0, "change_pct": 0.0, "currency": "",  "as_of": "reference"},
            {"name": "VIX",       "ticker": "^VIX",  "value": 16.5,   "change_pct": 0.0, "currency": "",  "as_of": "reference"},
            {"name": "Oil (WTI)", "ticker": "CL=F",  "value": 72.0,   "change_pct": 0.0, "currency": "$", "as_of": "reference"},
            {"name": "10Y Yield", "ticker": "^TNX",  "value": 4.35,   "change_pct": 0.0, "currency": "%", "as_of": "reference"},
        ],
        "data_source": "Reference (Yahoo Finance unavailable)",
        "note": "Approximate values — start backend for live data",
        "as_of": _now_iso(),
    }
    return fallback


@realtime_router.get("/realtime/fx-history")
async def get_fx_history() -> dict[str, Any]:
    """Fetch 30-day daily FX rate history from Yahoo Finance.

    Returns GBP/USD, EUR/USD, USD/CNY with:
      history:        list of {date, rate} — last 30 daily closes
      change_1d_pct:  today vs yesterday (real day-over-day change)
      change_label:   "1d" — interval for the displayed change %

    Cached 1 hour. Used by Market Monitor FX sparklines and change-% labels.
    """
    global _fx_history_cache, _fx_history_ts

    now = time.time()
    if _fx_history_cache and (now - _fx_history_ts) < _FX_HISTORY_TTL:
        return _fx_history_cache

    import pandas as _pd

    _FX_TICKERS = {
        "GBP/USD": "GBPUSD=X",
        "EUR/USD": "EURUSD=X",
        "USD/CNY": "USDCNY=X",
    }

    try:
        import yfinance as yf
        ticker_list = list(_FX_TICKERS.values())
        raw = yf.download(ticker_list, period="35d", interval="1d", progress=False)

        if not raw.empty:
            close = raw["Close"] if isinstance(raw.columns, _pd.MultiIndex) else raw
            ticker_to_pair = {v: k for k, v in _FX_TICKERS.items()}
            pairs: dict[str, Any] = {}

            for ticker, pair in ticker_to_pair.items():
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if len(series) >= 2:
                        history = [
                            {"date": str(idx)[:10], "rate": round(float(val), 4)}
                            for idx, val in zip(series.index[-30:], series.values[-30:])
                        ]
                        latest = float(series.iloc[-1])
                        prev = float(series.iloc[-2])
                        change_1d = (latest - prev) / prev * 100.0 if prev else 0.0
                        pairs[pair] = {
                            "history": history,
                            "change_1d_pct": round(change_1d, 3),
                            "change_label": "1d",
                        }

            if pairs:
                result: dict[str, Any] = {
                    "pairs": pairs,
                    "data_source": "Yahoo Finance",
                    "note": "Daily closes · change = today vs yesterday",
                    "as_of": _now_iso(),
                }
                _fx_history_cache = result
                _fx_history_ts = now
                logger.info(f"fx-history: fetched {len(pairs)} pairs from Yahoo Finance")
                return result
    except Exception as exc:
        logger.warning(f"fx-history: Yahoo Finance fetch failed ({exc})")

    return {
        "pairs": {},
        "data_source": "Unavailable",
        "note": "Yahoo Finance unavailable",
        "as_of": _now_iso(),
    }


# ── WebSocket ──────────────────────────────────────────────────────────────────


@realtime_router.websocket("/ws/market")
async def ws_market(websocket: WebSocket) -> None:
    """Live market tape — initial snapshot then one tick every `interval` seconds."""
    await websocket.accept()

    interval = 2.0
    raw_interval = websocket.query_params.get("interval")
    if raw_interval is not None:
        try:
            interval = _clamp(float(raw_interval), 0.5, 10.0)
        except (TypeError, ValueError):
            interval = 2.0

    feed = MarketFeed()
    logger.info(f"realtime: client connected (interval={interval:.1f}s, real_data={_is_real_data()})")

    try:
        await websocket.send_json(feed.snapshot())
        while True:
            await asyncio.sleep(interval)
            try:
                tick = feed.next_tick()
            except Exception as exc:
                logger.warning(f"realtime: tick error ({exc}); skipping")
                continue
            await websocket.send_json(tick)
    except WebSocketDisconnect:
        logger.info("realtime: client disconnected")
    except Exception as exc:
        logger.warning(f"realtime: websocket loop error ({exc})")
        try:
            await websocket.close()
        except Exception:
            pass
