"""
Synthetic Warranty + Extended Sales Data Generator
==================================================

Produces a realistic monthly warranty dataset for an automotive OEM, following
the same modelling philosophy as the Ornstein-Uhlenbeck synthetic generator in
``src/data/synthetic_generator.py`` (mean-reverting series + seasonality + drift).

PUBLIC AUTOMOTIVE WARRANTY BENCHMARKS (documented sources)
----------------------------------------------------------
The base parameters below are calibrated to publicly reported figures:

* Warranty cost as a share of revenue: premium OEMs typically run **2.5–3.5%**
  of automotive revenue as warranty/recall expense (Warranty Week annual OEM
  surveys; OEM 10-K / Annual Report warranty accrual disclosures, e.g. JLR /
  Tata Motors, BMW, Daimler).
* Claims frequency: expressed as **claims per 1,000 vehicles** (industry "R/1000"
  metric used by J.D. Power IQS/Dependability studies, ~100–250 problems/100
  vehicles → re-scaled to monetised warranty claims per 1,000).
* Average claim severity (GBP per claim) varies by failure mode; EV battery and
  high-voltage systems are **lower frequency but materially higher severity**
  (NHTSA EV recall data, public battery-pack replacement cost reporting).
* Failure-mode mix references J.D. Power problem categories and NHTSA component
  recall taxonomy (Powertrain, Electrical, Infotainment, Body, Suspension) plus
  a Battery/EV category for electrified platforms.
* EV warranty "learning curve": newer EV platforms show a slow decline in
  battery claim frequency as manufacturing matures (public EV reliability
  trend reporting). Modelled here as a gentle multiplicative improvement.

Optionally enriches the synthetic series with **public NHTSA recall counts** via
a best-effort HTTP GET (5s timeout). On any failure it silently falls back to
pure synthetic data — the generator never hard-depends on network access.

Output columns
--------------
date, segment, vehicles_sold, claims_count, claims_per_1000, avg_claim_gbp,
warranty_cost_gbp, warranty_pct_revenue, dominant_failure_mode, accrual_gbp,
accrual_adequacy_pct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_project_root, get_settings

# ── Failure-mode taxonomy ────────────────────────────────────────────────────
FAILURE_MODES = [
    "Powertrain",
    "Electrical",
    "Battery/EV",
    "Infotainment",
    "Body",
    "Suspension",
]

# ── Per-segment warranty parameters (calibrated to public benchmarks) ─────────
# avg_price_gbp     : approximate transaction price (USD prices in settings × ~0.79)
# claims_per_1000   : baseline monetised warranty claims per 1,000 vehicles / month
# avg_claim_gbp     : baseline average claim severity (GBP)
# warranty_pct_base : target warranty cost as % of revenue (2.5–3.5% band)
# failure_mix       : probability weights over FAILURE_MODES
SEGMENT_PARAMS: dict[str, dict] = {
    "Luxury_SUV": {
        "avg_price_gbp": 75000,
        "claims_per_1000": 24.0,
        "avg_claim_gbp": 1350.0,
        "warranty_pct_base": 0.030,
        "failure_mix": [0.28, 0.22, 0.02, 0.20, 0.16, 0.12],
    },
    "Premium_SUV": {
        "avg_price_gbp": 49000,
        "claims_per_1000": 27.0,
        "avg_claim_gbp": 1050.0,
        "warranty_pct_base": 0.031,
        "failure_mix": [0.30, 0.20, 0.02, 0.18, 0.18, 0.12],
    },
    "Performance": {
        "avg_price_gbp": 44000,
        "claims_per_1000": 22.0,
        "avg_claim_gbp": 1200.0,
        "warranty_pct_base": 0.028,
        "failure_mix": [0.34, 0.20, 0.01, 0.15, 0.16, 0.14],
    },
    "EV": {
        # EV: lower overall frequency on mechanical modes, but battery claims are
        # rare and very high severity → drags the blended severity upward.
        "avg_price_gbp": 59000,
        "claims_per_1000": 19.0,
        "avg_claim_gbp": 2100.0,
        "warranty_pct_base": 0.034,
        "failure_mix": [0.14, 0.26, 0.18, 0.22, 0.12, 0.08],
        "battery_severity_gbp": 9500.0,  # high-voltage pack / module replacement
    },
}

NHTSA_RECALL_URL = (
    "https://api.nhtsa.gov/recalls/recallsByVehicle"
    "?make=land%20rover&model=range%20rover&modelYear=2023"
)


@dataclass
class WarrantyDataGenerator:
    """
    Generate a realistic synthetic warranty dataset (2019-01 → 2026-12) by
    segment, with seasonality, an EV-warranty learning curve, failure-mode
    attribution, and accrual adequacy.

    Mirrors the OU-style synthetic philosophy: a mean-reverting stochastic
    component on top of structural trend + seasonality.
    """

    start: str = "2019-01-01"
    end: str = "2026-12-01"
    seed: int = 42
    enrich_nhtsa: bool = True
    nhtsa_factor: float = 1.0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ── Best-effort public recall enrichment ─────────────────────────────────
    def _fetch_nhtsa_factor(self) -> float:
        """
        Best-effort GET of public NHTSA recall counts. Returns a small multiplier
        (~0.97–1.05) derived from recall volume. Any failure → 1.0 (pure synthetic).
        """
        if not self.enrich_nhtsa:
            return 1.0
        try:
            import httpx

            resp = httpx.get(NHTSA_RECALL_URL, timeout=5.0)
            resp.raise_for_status()
            payload = resp.json()
            n_recalls = int(payload.get("Count", len(payload.get("results", []))))
            # Map recall count into a gentle frequency multiplier.
            factor = 1.0 + min(n_recalls, 20) * 0.0025
            logger.info(f"NHTSA enrichment: {n_recalls} recalls → factor {factor:.3f}")
            return float(factor)
        except Exception as exc:  # noqa: BLE001 — network optional, never fail
            logger.debug(f"NHTSA enrichment skipped (offline / error): {exc}")
            return 1.0

    # ── Stochastic helper ────────────────────────────────────────────────────
    def _ou_multiplier(self, n: int, vol: float, mean_rev: float = 0.20) -> np.ndarray:
        """Mean-reverting multiplicative noise centred on 1.0 (OU-style)."""
        out = np.ones(n)
        for t in range(1, n):
            drift = mean_rev * (1.0 - out[t - 1])
            out[t] = out[t - 1] + drift + vol * self._rng.standard_normal()
            out[t] = max(out[t], 0.4)
        return out

    def _monthly_volumes(self, n: int, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        """Vehicles sold per segment per month (trend + seasonality + noise)."""
        settings = get_settings()
        vol_map = {s["segment"]: s["annual_volume"] for s in settings["vehicle_segments"]}
        volumes: dict[str, np.ndarray] = {}
        for seg in SEGMENT_PARAMS:
            monthly_base = vol_map.get(seg, 60000) / 12.0
            i = np.arange(n)
            # EV ramps faster than ICE segments
            growth = 0.05 if seg == "EV" else 0.02
            trend = 1.0 + growth * (i / 12.0)
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (dates.month - 3) / 12)
            noise = self._rng.normal(1.0, 0.07, n)
            volumes[seg] = np.clip(monthly_base * trend * seasonal * noise, 0, None)
        return volumes

    # ── Main generation ──────────────────────────────────────────────────────
    def generate(self) -> pd.DataFrame:
        """Generate the full monthly warranty time series across segments."""
        self.nhtsa_factor = self._fetch_nhtsa_factor()
        dates = pd.date_range(self.start, self.end, freq="MS")
        n = len(dates)
        volumes = self._monthly_volumes(n, dates)

        records: list[dict] = []
        for seg, p in SEGMENT_PARAMS.items():
            veh = volumes[seg]
            # Frequency: OU noise + mild seasonal (winter electrical/battery bump)
            freq_noise = self._ou_multiplier(n, vol=0.05)
            seasonal_freq = 1.0 + 0.08 * np.cos(2 * np.pi * (dates.month - 1) / 12)
            base_freq = p["claims_per_1000"] * self.nhtsa_factor

            # EV learning curve: slow decline in battery-driven frequency over time.
            ev_learn = np.ones(n)
            if seg == "EV":
                ev_learn = 1.0 - 0.22 * (np.arange(n) / max(n - 1, 1))  # ~22% improvement

            claims_per_1000 = base_freq * freq_noise * seasonal_freq * ev_learn
            claims_count = np.round(claims_per_1000 * veh / 1000.0).astype(int)

            # Severity: OU noise + slow inflation drift on parts/labour.
            sev_noise = self._ou_multiplier(n, vol=0.04)
            inflation = 1.0 + 0.025 * (np.arange(n) / 12.0)
            avg_claim = p["avg_claim_gbp"] * sev_noise * inflation

            # EV blended severity rises with battery share, but learning curve
            # reduces battery frequency over time (severity stays high, count falls).
            if seg == "EV":
                batt_share = 0.18 * ev_learn  # battery share of EV claims declines
                avg_claim = (
                    (1 - batt_share) * avg_claim
                    + batt_share * p["battery_severity_gbp"] * sev_noise
                )

            warranty_cost = claims_count * avg_claim
            revenue = veh * p["avg_price_gbp"]
            warranty_pct = np.where(revenue > 0, warranty_cost / revenue, 0.0)

            # Dominant failure mode per month (segment mix + stochastic tilt).
            mix = np.array(p["failure_mix"], dtype=float)
            dominant = []
            for t in range(n):
                tilt = mix.copy()
                if seg == "EV":
                    # Battery dominance fades as the platform matures.
                    bidx = FAILURE_MODES.index("Battery/EV")
                    tilt[bidx] *= ev_learn[t]
                    tilt = tilt / tilt.sum()
                draw = self._rng.choice(FAILURE_MODES, p=tilt / tilt.sum())
                dominant.append(draw)

            # Accrual: company books warranty_pct_base × revenue. Adequacy = accrual
            # vs incurred. We inject a deliberate EV under-accrual episode (rising
            # battery severity outran the booked reserve) for a realistic warning.
            accrual = p["warranty_pct_base"] * revenue
            if seg == "EV":
                # Mid-2024 onward, severity spike → reserve lags incurred cost.
                lag = np.where(np.arange(n) >= int(n * 0.70), 0.88, 1.0)
                accrual = accrual * lag
            accrual_adequacy = np.where(
                warranty_cost > 0, accrual / warranty_cost * 100.0, 100.0
            )

            for t in range(n):
                records.append(
                    {
                        "date": dates[t],
                        "segment": seg,
                        "vehicles_sold": int(veh[t]),
                        "claims_count": int(claims_count[t]),
                        "claims_per_1000": round(float(claims_per_1000[t]), 2),
                        "avg_claim_gbp": round(float(avg_claim[t]), 2),
                        "warranty_cost_gbp": round(float(warranty_cost[t]), 2),
                        "warranty_pct_revenue": round(float(warranty_pct[t] * 100), 3),
                        "dominant_failure_mode": dominant[t],
                        "accrual_gbp": round(float(accrual[t]), 2),
                        "accrual_adequacy_pct": round(float(accrual_adequacy[t]), 2),
                    }
                )

        df = pd.DataFrame(records).sort_values(["date", "segment"]).reset_index(drop=True)
        logger.info(
            f"Generated warranty data: {len(df)} rows "
            f"({df['segment'].nunique()} segments × {n} months)"
        )
        return df

    def save(self, path: Path | str | None = None) -> pd.DataFrame:
        """Generate and write the warranty dataset to data/synthetic/warranty_data.csv."""
        df = self.generate()
        if path is None:
            out_dir = get_project_root() / "data" / "synthetic"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "warranty_data.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved warranty data → {path}")
        return df


if __name__ == "__main__":  # pragma: no cover
    WarrantyDataGenerator().save()
