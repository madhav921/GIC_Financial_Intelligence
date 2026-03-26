"""
Synthetic data generator for local development & testing.

Generates realistic time-series data for:
- Commodity prices (Lithium, Cobalt, Nickel, Aluminum, Steel, Copper, Platinum, Rubber)
- Vehicle sales volumes by segment
- Macroeconomic indicators (interest rates, GDP growth, FX rates)
- Production & inventory data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_project_root, get_settings


# ── Realistic base prices & volatilities for automotive commodities ──
COMMODITY_PARAMS = {
    "Lithium":   {"base": 25000, "vol": 0.35, "trend": 0.02, "mean_rev": 0.05},
    "Cobalt":    {"base": 33000, "vol": 0.28, "trend": 0.01, "mean_rev": 0.06},
    "Nickel":    {"base": 18000, "vol": 0.22, "trend": 0.01, "mean_rev": 0.07},
    "Aluminum":  {"base": 2400,  "vol": 0.15, "trend": 0.005, "mean_rev": 0.10},
    "Steel":     {"base": 800,   "vol": 0.12, "trend": 0.003, "mean_rev": 0.12},
    "Copper":    {"base": 8500,  "vol": 0.18, "trend": 0.008, "mean_rev": 0.08},
    "Platinum":  {"base": 950,   "vol": 0.20, "trend": -0.005, "mean_rev": 0.09},
    "Rubber":    {"base": 1600,  "vol": 0.14, "trend": 0.002, "mean_rev": 0.11},
}

MACRO_PARAMS = {
    "gdp_growth_pct":    {"base": 2.5, "vol": 0.8, "mean_rev": 0.15},
    "interest_rate_pct":  {"base": 4.5, "vol": 0.5, "mean_rev": 0.10},
    "usd_gbp":           {"base": 0.79, "vol": 0.06, "mean_rev": 0.12},
    "usd_eur":           {"base": 0.92, "vol": 0.05, "mean_rev": 0.12},
    "cpi_index":         {"base": 110, "vol": 1.5, "mean_rev": 0.03},
    "oil_price_usd":     {"base": 80, "vol": 0.25, "mean_rev": 0.08},
}


def _ou_process(
    n_steps: int,
    base: float,
    vol: float,
    mean_rev: float,
    trend: float = 0.0,
    dt: float = 1 / 12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck process with drift — realistic mean-reverting series."""
    rng = rng or np.random.default_rng()
    prices = np.zeros(n_steps)
    prices[0] = base
    for t in range(1, n_steps):
        drift = mean_rev * (base * (1 + trend * t * dt) - prices[t - 1]) * dt
        diffusion = vol * prices[t - 1] * np.sqrt(dt) * rng.standard_normal()
        prices[t] = max(prices[t - 1] + drift + diffusion, base * 0.1)
    return prices


def generate_commodity_prices(
    start: str = "2019-01-01",
    periods: int = 84,  # 7 years monthly
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic monthly commodity price data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")

    data = {"date": dates}
    for commodity, params in COMMODITY_PARAMS.items():
        prices = _ou_process(
            periods, params["base"], params["vol"],
            params["mean_rev"], params["trend"], rng=rng,
        )
        # Add seasonal component (Q4 demand bump for some commodities)
        seasonal = 1 + 0.03 * np.sin(2 * np.pi * np.arange(periods) / 12)
        data[commodity] = np.round(prices * seasonal, 2)

    df = pd.DataFrame(data)
    logger.info(f"Generated commodity prices: {df.shape[0]} rows × {len(COMMODITY_PARAMS)} commodities")
    return df


def generate_sales_data(
    start: str = "2019-01-01",
    periods: int = 84,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic monthly vehicle sales by segment."""
    rng = np.random.default_rng(seed)
    settings = get_settings()
    dates = pd.date_range(start, periods=periods, freq="MS")

    records = []
    for seg in settings["vehicle_segments"]:
        monthly_base = seg["annual_volume"] / 12
        for i, date in enumerate(dates):
            # Trend + seasonality + noise
            trend = 1 + 0.02 * (i / 12)
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * (date.month - 3) / 12)
            noise = rng.normal(1.0, 0.08)
            volume = int(monthly_base * trend * seasonal * noise)

            # Price varies slightly
            price = seg["avg_price_usd"] * (1 + rng.normal(0, 0.03))

            records.append({
                "date": date,
                "segment": seg["segment"],
                "volume": max(volume, 0),
                "avg_price_usd": round(price, 2),
                "incentive_pct": round(rng.uniform(0.02, 0.08), 4),
                "region": rng.choice(["NA", "EU", "UK", "CN", "ROW"], p=[0.30, 0.25, 0.15, 0.20, 0.10]),
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated sales data: {df.shape[0]} rows")
    return df


def generate_macro_indicators(
    start: str = "2019-01-01",
    periods: int = 84,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic macroeconomic indicator series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    data = {"date": dates}

    for indicator, params in MACRO_PARAMS.items():
        values = _ou_process(
            periods, params["base"], params["vol"],
            params["mean_rev"], rng=rng,
        )
        data[indicator] = np.round(values, 4)

    df = pd.DataFrame(data)
    logger.info(f"Generated macro indicators: {df.shape[0]} rows")
    return df


def generate_production_inventory(
    start: str = "2019-01-01",
    periods: int = 84,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic production & inventory data."""
    rng = np.random.default_rng(seed)
    settings = get_settings()
    dates = pd.date_range(start, periods=periods, freq="MS")

    records = []
    for seg in settings["vehicle_segments"]:
        monthly_base = seg["annual_volume"] / 12
        inventory = int(monthly_base * 1.5)  # starting inventory

        for i, date in enumerate(dates):
            production = int(monthly_base * rng.normal(1.0, 0.10))
            sales = int(monthly_base * rng.normal(0.95, 0.12))
            production = max(production, 0)
            sales = max(min(sales, inventory + production), 0)
            inventory = inventory + production - sales

            records.append({
                "date": date,
                "segment": seg["segment"],
                "production_units": production,
                "sales_units": sales,
                "ending_inventory": max(inventory, 0),
                "capacity_utilization_pct": round(min(production / (monthly_base * 1.2), 1.0) * 100, 1),
                "warranty_claims": int(sales * rng.uniform(0.008, 0.025)),
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated production/inventory data: {df.shape[0]} rows")
    return df


def generate_bom_data() -> pd.DataFrame:
    """Generate Bill of Materials cost breakdown by commodity and segment."""
    settings = get_settings()
    records = []

    for seg in settings["vehicle_segments"]:
        for comm in settings["commodities"]:
            # BOM weight varies by segment (EVs use more lithium/cobalt)
            multiplier = 1.0
            if seg["segment"] == "EV" and comm["category"] == "battery_materials":
                multiplier = 2.5
            elif seg["segment"] == "Performance" and comm["category"] == "structural":
                multiplier = 1.3

            records.append({
                "segment": seg["segment"],
                "commodity": comm["name"],
                "category": comm["category"],
                "bom_weight": round(comm["bom_weight"] * multiplier, 4),
                "unit": comm["unit"],
                "qty_per_vehicle": round(comm["bom_weight"] * multiplier * seg["avg_price_usd"] / 100, 2),
            })

    return pd.DataFrame(records)


def generate_all_synthetic_data(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate all synthetic datasets and save to disk."""
    root = get_project_root()
    out_dir = root / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "commodity_prices": generate_commodity_prices(seed=seed),
        "sales_data": generate_sales_data(seed=seed),
        "macro_indicators": generate_macro_indicators(seed=seed),
        "production_inventory": generate_production_inventory(seed=seed),
        "bom_data": generate_bom_data(),
    }

    for name, df in datasets.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {name} → {path}")

    return datasets
