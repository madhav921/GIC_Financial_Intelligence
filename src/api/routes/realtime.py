"""Real-time market data-capture routes (REST snapshot + WebSocket live tape).

Exposes a live market feed for the dashboard. The feed is self-contained: it
seeds from ``data/synthetic/commodity_prices.csv`` when present, otherwise falls
back to config-derived defaults. Each tick applies a small mean-reverting random
walk so the stream looks believable without any external data dependency.
"""

from __future__ import annotations

import asyncio
import csv
import random
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.config import get_project_root, get_settings

realtime_router = APIRouter(tags=["realtime"])


# ── Static reference data ──────────────────────────────────────────────────

# Commodities surfaced on the live tape (name -> unit). Order is preserved for
# the ``top_commodities`` payload (6 entries, as required by the UI).
_TOP_COMMODITY_UNITS: dict[str, str] = {
    "Steel": "USD/tonne",
    "Lithium": "USD/kg",
    "Aluminum": "USD/tonne",
    "Copper": "USD/tonne",
    "Cobalt": "USD/tonne",
    "Nickel": "USD/tonne",
}

# Sensible fallback prices (used when the CSV is missing/unreadable). Loosely
# aligned with the synthetic dataset's recent levels.
_DEFAULT_PRICES: dict[str, float] = {
    "Steel": 490.0,
    "Lithium": 4.0,
    "Aluminum": 1650.0,
    "Copper": 8200.0,
    "Cobalt": 13000.0,
    "Nickel": 8000.0,
}

_DEFAULT_FX: dict[str, float] = {
    "GBP/USD": 1.27,
    "EUR/USD": 1.08,
    "USD/CNY": 7.15,
}

# Curated, actionable headlines rotated through on the live feed.
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


def _seed_prices() -> dict[str, float]:
    """Seed commodity prices from the last row of the synthetic CSV, if present.

    Falls back to ``_DEFAULT_PRICES`` for any commodity not found or on any
    read error — this never raises.
    """
    prices = dict(_DEFAULT_PRICES)
    try:
        csv_path = get_project_root() / "data" / "synthetic" / "commodity_prices.csv"
        if not csv_path.exists():
            return prices
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return prices
        last = rows[-1]
        for name in _TOP_COMMODITY_UNITS:
            raw = last.get(name)
            if raw in (None, ""):
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val > 0:
                prices[name] = val
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("realtime: could not seed prices from CSV ({}); using defaults", exc)
    return prices


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarketFeed:
    """Stateful market feed producing believable mean-reverting ticks.

    Holds the current price/index/FX/risk state and advances it on each call to
    :meth:`next_tick`, which returns a fully-formed snapshot dict. This is
    deliberately decoupled from any WebSocket so the tick logic is unit-testable.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._tick_count = 0

        try:
            settings = get_settings()
        except Exception:  # pragma: no cover - defensive
            settings = {}
        self._settings = settings

        # Current absolute prices + a remembered previous tick for change_pct.
        self._prices: dict[str, float] = _seed_prices()
        self._prev_prices: dict[str, float] = dict(self._prices)

        self._fx: dict[str, float] = dict(_DEFAULT_FX)
        self._prev_fx: dict[str, float] = dict(self._fx)

        self._index: float = 100.0
        self._prev_index: float = 100.0

        self._risk: float = 42.0
        self._headline_idx: int = 0

    # ── internal random-walk helpers ───────────────────────────────────────

    def _walk(self, value: float, anchor: float, *, vol: float, reversion: float) -> float:
        """One mean-reverting step.

        ``vol`` is the per-tick volatility (fraction of value); ``reversion``
        pulls the value back toward ``anchor``. Result stays strictly positive.
        """
        shock = self._rng.gauss(0.0, vol)
        pull = reversion * (anchor - value) / value if value else 0.0
        new_value = value * (1.0 + shock + pull)
        return max(new_value, anchor * 0.05, 1e-6)

    @staticmethod
    def _pct(curr: float, prev: float) -> float:
        if prev == 0:
            return 0.0
        return round((curr - prev) / prev * 100.0, 3)

    def _ebit_nowcast(self) -> float:
        """Live EBIT estimate in the £1.3–1.5bn range, anti-correlated with the
        commodity index (higher input costs -> lower EBIT)."""
        base = 1.42e9
        # +/-0.06bn swing across the plausible index range (95..115 -> ~100 mid).
        adjustment = (100.0 - self._index) / 20.0 * 0.06e9
        jitter = self._rng.gauss(0.0, 0.004e9)
        return round(_clamp(base + adjustment + jitter, 1.30e9, 1.50e9), 0)

    # ── public API ─────────────────────────────────────────────────────────

    def next_tick(self) -> dict[str, Any]:
        """Advance state by one tick and return the current snapshot dict."""
        self._tick_count += 1

        # Snapshot previous state for change_pct computation.
        self._prev_prices = dict(self._prices)
        self._prev_fx = dict(self._fx)
        self._prev_index = self._index

        # Walk commodity prices (mean-revert toward their seed anchor).
        for name, price in self._prices.items():
            anchor = _DEFAULT_PRICES.get(name, price)
            # Use a blended anchor: seeded level matters more than the static default.
            self._prices[name] = self._walk(price, anchor, vol=0.006, reversion=0.02)

        # Walk FX (low volatility).
        for pair, rate in self._fx.items():
            anchor = _DEFAULT_FX[pair]
            self._fx[pair] = self._walk(rate, anchor, vol=0.0015, reversion=0.03)

        # Walk the commodity index, mean-reverting to ~100, clamped to ~95-115.
        self._index = _clamp(
            self._walk(self._index, 100.0, vol=0.004, reversion=0.05), 95.0, 115.0
        )

        # Drift the risk score, nudged upward when the index runs hot.
        index_pressure = (self._index - 100.0) * 0.4
        self._risk = _clamp(
            self._risk + self._rng.gauss(index_pressure * 0.05, 0.8), 0.0, 100.0
        )

        # Rotate the headline roughly every 10 ticks.
        if self._tick_count % 10 == 0:
            self._headline_idx = (self._headline_idx + 1) % len(_HEADLINES)

        return self._build_snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return the current snapshot without advancing state (initial UI state)."""
        return self._build_snapshot()

    def _build_snapshot(self) -> dict[str, Any]:
        top_commodities = [
            {
                "name": name,
                "price": round(self._prices[name], 2),
                "change_pct": self._pct(self._prices[name], self._prev_prices.get(name, self._prices[name])),
                "unit": _TOP_COMMODITY_UNITS[name],
            }
            for name in _TOP_COMMODITY_UNITS
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
        # Active alerts grow with risk and headline severity.
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
        }


@realtime_router.get("/realtime/snapshot")
async def get_snapshot() -> dict[str, Any]:
    """Return the current market snapshot (initial state for the UI)."""
    feed = MarketFeed()
    return feed.snapshot()


@realtime_router.websocket("/ws/market")
async def ws_market(websocket: WebSocket) -> None:
    """Live market tape over WebSocket.

    On connect, immediately sends the snapshot, then pushes an updated tick every
    ``interval`` seconds (query param, default 2.0, clamped to 0.5–10). Robust to
    bad ticks (per-tick try/except) and to client disconnects.
    """
    await websocket.accept()

    # Parse + clamp the interval query param defensively.
    interval = 2.0
    raw_interval = websocket.query_params.get("interval")
    if raw_interval is not None:
        try:
            interval = _clamp(float(raw_interval), 0.5, 10.0)
        except (TypeError, ValueError):
            interval = 2.0

    feed = MarketFeed()
    logger.info("realtime: client connected (interval={:.1f}s)", interval)

    try:
        # Send the initial snapshot immediately.
        await websocket.send_json(feed.snapshot())

        while True:
            await asyncio.sleep(interval)
            try:
                tick = feed.next_tick()
            except Exception as exc:  # one bad tick must not kill the socket
                logger.warning("realtime: tick generation failed ({}); skipping", exc)
                continue
            await websocket.send_json(tick)
    except WebSocketDisconnect:
        logger.info("realtime: client disconnected")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("realtime: websocket loop error ({})", exc)
        try:
            await websocket.close()
        except Exception:
            pass
