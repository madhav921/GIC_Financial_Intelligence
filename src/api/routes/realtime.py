"""Real-time market data-capture routes (REST snapshot + WebSocket live tape).

Prices are seeded from the same data source as the forecast models:
  data/raw/commodity_prices.csv  (Yahoo Finance, via scripts/fetch_data.py)  → primary
  data/synthetic/commodity_prices.csv                                         → fallback

The random walk mean-reverts to the real latest prices, so the live ticker
stays anchored to real market levels rather than drifting to fictional defaults.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.config import get_settings

realtime_router = APIRouter(tags=["realtime"])


# ── Commodity metadata ─────────────────────────────────────────────────────────

# The 6 commodities surfaced on the live dashboard ticker.
# Names must match column headers in commodity_prices.csv exactly.
_TOP_COMMODITY_NAMES: list[str] = [
    "Steel", "Lithium", "Aluminum", "Copper", "Cobalt", "Nickel",
]

_COMMODITY_UNITS: dict[str, str] = {
    "Steel":       "USD/t",
    "Lithium":     "USD/kg",
    "Aluminum":    "USD/t",
    "Copper":      "USD/t",
    "Cobalt":      "USD/t",
    "Nickel":      "USD/t",
    "Platinum":    "USD/oz",
    "Palladium":   "USD/oz",
    "Natural_Gas": "USD/MMBtu",
    "Polypropylene": "USD/t",
    "ABS_Resin":   "USD/t",
}

# Realistic fallback prices aligned with Yahoo Finance proxy values.
# These are used ONLY when both data/raw/ and yfinance calls fail.
# Units match COMMODITY_UNITS above (data from last known good fetch).
_FALLBACK_PRICES: dict[str, float] = {
    "Steel":         790.0,    # USD/t  (SLX ETF ×7.5 proxy)
    "Lithium":        21.0,    # USD/kg (LIT ETF ×0.25 proxy)
    "Aluminum":     3735.0,    # USD/t  (AA stock ×60 proxy)
    "Copper":      13900.0,    # USD/t  (HG=F futures ×2204.62)
    "Cobalt":      24700.0,    # USD/t  (GLNCY stock ×1600 proxy)
    "Nickel":      12250.0,    # USD/t  (VALE stock ×750 proxy)
    "Platinum":     1970.0,    # USD/oz (PL=F futures)
    "Palladium":    1420.0,    # USD/oz (PA=F futures)
    "Natural_Gas":    30.0,    # USD/MMBtu
    "Polypropylene": 940.0,    # USD/t  (synthetic)
    "ABS_Resin":    1440.0,    # USD/t  (synthetic)
}

_DEFAULT_FX: dict[str, float] = {
    "GBP/USD": 1.27,
    "EUR/USD": 1.08,
    "USD/CNY": 7.15,
}

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


def _seed_prices_from_real_data() -> dict[str, float]:
    """Load the latest commodity prices from the same source as the forecast models.

    Priority: data/raw/commodity_prices.csv → yfinance live → _FALLBACK_PRICES.
    Returns a dict keyed by commodity name with the most recent price value.
    """
    try:
        from src.data.market_data_provider import get_latest_prices
        latest = get_latest_prices()
        if latest:
            prices = {name: latest[name] for name in _FALLBACK_PRICES if name in latest}
            missing = [n for n in _FALLBACK_PRICES if n not in prices]
            if missing:
                for m in missing:
                    prices[m] = _FALLBACK_PRICES[m]
            logger.info(
                "realtime: seeded from MarketDataProvider — "
                f"Steel={prices.get('Steel'):.0f}, Copper={prices.get('Copper'):.0f}, "
                f"Lithium={prices.get('Lithium'):.2f}"
            )
            return prices
    except Exception as exc:
        logger.warning(f"realtime: MarketDataProvider seed failed ({exc}); using fallback prices")
    return dict(_FALLBACK_PRICES)


class MarketFeed:
    """Stateful market feed producing believable mean-reverting ticks.

    Prices are seeded from real market data and the random walk stays anchored
    to those real levels, so the live tape is consistent with forecast model inputs.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._tick_count = 0

        try:
            self._settings = get_settings()
        except Exception:
            self._settings = {}

        # Seed from real data (same source as forecast models).
        self._anchor: dict[str, float] = _seed_prices_from_real_data()

        # Current prices start at the real anchors.
        self._prices: dict[str, float] = dict(self._anchor)
        self._prev_prices: dict[str, float] = dict(self._prices)

        self._fx: dict[str, float] = dict(_DEFAULT_FX)
        self._prev_fx: dict[str, float] = dict(self._fx)

        # Commodity index starts at 100 (relative to our seeded baseline).
        self._index: float = 100.0
        self._prev_index: float = 100.0

        self._risk: float = 42.0
        self._headline_idx: int = 0

    # ── random-walk helpers ────────────────────────────────────────────────────

    def _walk(self, value: float, anchor: float, *, vol: float, reversion: float) -> float:
        """One mean-reverting step toward anchor. Returns strictly positive value."""
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
        """Live EBIT estimate in the £3.5–4.0bn annual range (£290-330M/month).

        Anti-correlated with commodity index: higher input costs → lower EBIT.
        """
        base = 3.75e9          # £3.75bn annual (consistent with /pnl/annual)
        # ±£150m swing across the plausible index range (95..115 → ~100 mid).
        adjustment = (100.0 - self._index) / 20.0 * 0.15e9
        jitter = self._rng.gauss(0.0, 0.02e9)
        return round(_clamp(base + adjustment + jitter, 3.3e9, 4.2e9), 0)

    # ── public API ─────────────────────────────────────────────────────────────

    def next_tick(self) -> dict[str, Any]:
        """Advance state by one tick and return a snapshot dict."""
        self._tick_count += 1

        self._prev_prices = dict(self._prices)
        self._prev_fx = dict(self._fx)
        self._prev_index = self._index

        # Walk commodity prices: mean-revert toward the real seeded anchor.
        for name in self._prices:
            anchor = self._anchor.get(name, self._prices[name])
            self._prices[name] = self._walk(
                self._prices[name], anchor, vol=0.004, reversion=0.015
            )

        # Walk FX (tight band around real rates).
        for pair in self._fx:
            anchor = _DEFAULT_FX[pair]
            self._fx[pair] = self._walk(self._fx[pair], anchor, vol=0.0010, reversion=0.03)

        # Walk commodity index, mean-reverting to 100.
        self._index = _clamp(
            self._walk(self._index, 100.0, vol=0.003, reversion=0.04), 92.0, 118.0
        )

        # Drift risk score, nudged by index.
        index_pressure = (self._index - 100.0) * 0.3
        self._risk = _clamp(
            self._risk + self._rng.gauss(index_pressure * 0.04, 0.6), 0.0, 100.0
        )

        if self._tick_count % 10 == 0:
            self._headline_idx = (self._headline_idx + 1) % len(_HEADLINES)

        return self._build_snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return the current snapshot without advancing state."""
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
            for pair in _DEFAULT_FX
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
    """True when data/raw/commodity_prices.csv is present (real yfinance data)."""
    try:
        from src.config import get_project_root
        return (get_project_root() / "data" / "raw" / "commodity_prices.csv").exists()
    except Exception:
        return False


@realtime_router.get("/realtime/snapshot")
async def get_snapshot() -> dict[str, Any]:
    """Current market snapshot (initial state for the UI)."""
    feed = MarketFeed()
    return feed.snapshot()


@realtime_router.post("/realtime/refresh")
async def refresh_market_data() -> dict[str, Any]:
    """Force-refresh the commodity price cache (call after running fetch_data.py).

    Re-seeds from the newest data/raw/commodity_prices.csv and clears the
    MarketDataProvider in-memory cache so the next request pulls fresh data.
    """
    try:
        from src.data.market_data_provider import get_latest_prices, invalidate_cache
        invalidate_cache()
        latest = get_latest_prices()
        return {
            "status": "refreshed",
            "source": get_data_source_label(),
            "latest_prices": latest,
            "commodities": len(latest),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def get_data_source_label() -> str:
    """Return human-readable label for the active market data source."""
    try:
        from src.data.market_data_provider import get_data_source_label as _label
        return _label()
    except Exception:
        return "Unknown"


@realtime_router.websocket("/ws/market")
async def ws_market(websocket: WebSocket) -> None:
    """Live market tape over WebSocket.

    Sends an initial snapshot immediately on connect, then pushes a new tick
    every ``interval`` seconds (query param, default 2.0, clamped 0.5–10).
    """
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
