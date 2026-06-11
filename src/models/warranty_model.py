"""
Warranty Analytics Model (Actionable-Intelligence Layer).

Lightweight, dependency-light warranty analytics that consume the synthetic
warranty dataset produced by ``src/data/warranty_generator.py`` and turn it into
forward-looking, prescriptive signals:

  • ``forecast_warranty_cost``   — EWMA + seasonal-index projection (avoids heavy
                                   SARIMAX fits; SARIMAX is available in
                                   ``commodity_forecast`` if a richer fit is needed).
  • ``assess_accrual_adequacy``  — accrual vs incurred → adequate / under / over.
  • ``failure_mode_breakdown``   — % mix by failure mode + rising-mode flags.
  • ``warranty_risk_score``      — 0–100 composite (claims trend, severity trend,
                                   accrual gap).

All monetary figures are GBP. The model is deliberately robust to partial data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WarrantyForecast:
    """Container for a warranty-cost forecast."""
    dates: list[str]
    point: list[float]
    lower: list[float]
    upper: list[float]
    method: str = "EWMA+Seasonal"
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dates": self.dates,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "metrics": self.metrics,
        }


class WarrantyModel:
    """Trend + seasonal warranty analytics over the warranty dataset."""

    def __init__(self, ewma_span: int = 6) -> None:
        self.ewma_span = ewma_span

    # ── internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _monthly_total(warranty_df: pd.DataFrame, col: str) -> pd.Series:
        """Aggregate a column to a monthly total series indexed by date."""
        df = warranty_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        s = df.groupby("date")[col].sum().sort_index()
        return s

    @staticmethod
    def _seasonal_index(series: pd.Series) -> np.ndarray:
        """Multiplicative month-of-year seasonal index (length 12, mean≈1)."""
        if series.empty:
            return np.ones(12)
        df = series.to_frame("v")
        df["month"] = df.index.month
        overall = df["v"].mean()
        if overall <= 0:
            return np.ones(12)
        idx = df.groupby("month")["v"].mean() / overall
        out = np.ones(12)
        for m in range(1, 13):
            if m in idx.index and np.isfinite(idx.loc[m]):
                out[m - 1] = float(idx.loc[m])
        return out

    # ── 1. Forecast ───────────────────────────────────────────────────────────
    def forecast_warranty_cost(
        self, warranty_df: pd.DataFrame, horizon_months: int = 12
    ) -> WarrantyForecast:
        """
        Project total monthly warranty cost forward via an EWMA level estimate
        combined with a multiplicative month-of-year seasonal index.

        Uncertainty band derives from the residual std of the in-sample EWMA fit,
        widening with the square-root of the forecast step (random-walk-like).
        """
        series = self._monthly_total(warranty_df, "warranty_cost_gbp")
        if series.empty:
            return WarrantyForecast([], [], [], [], metrics={"n_obs": 0})

        # Deseasonalise, fit EWMA level on the deseasonalised series.
        seasonal = self._seasonal_index(series)
        month_of = np.array([d.month for d in series.index])
        seas_factor = seasonal[month_of - 1]
        deseason = series.values / np.where(seas_factor == 0, 1.0, seas_factor)

        ewma = pd.Series(deseason).ewm(span=self.ewma_span, adjust=False).mean()
        level = float(ewma.iloc[-1])

        # Trend: slope of last min(12, n) EWMA points.
        tail = ewma.iloc[-min(12, len(ewma)):].values
        if len(tail) >= 2:
            slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
        else:
            slope = 0.0

        resid_std = float(np.std(series.values - ewma.values * seas_factor))

        last_date = series.index[-1]
        future_dates = pd.date_range(
            last_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS"
        )

        point, lower, upper = [], [], []
        for h, d in enumerate(future_dates, start=1):
            base = max(level + slope * h, 0.0)
            seas = seasonal[d.month - 1]
            mean = base * seas
            band = 1.96 * resid_std * np.sqrt(h)
            point.append(round(mean, 2))
            lower.append(round(max(mean - band, 0.0), 2))
            upper.append(round(mean + band, 2))

        logger.info(
            f"Warranty forecast: level={level:,.0f}, slope={slope:,.0f}/mo, "
            f"h={horizon_months}, next={point[0]:,.0f}"
        )
        return WarrantyForecast(
            dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            point=point, lower=lower, upper=upper,
            metrics={
                "level_gbp": round(level, 2),
                "trend_gbp_per_month": round(slope, 2),
                "resid_std_gbp": round(resid_std, 2),
                "n_obs": int(len(series)),
            },
        )

    # ── 2. Accrual adequacy ───────────────────────────────────────────────────
    def assess_accrual_adequacy(self, warranty_df: pd.DataFrame) -> dict:
        """
        Compare cumulative accrual vs incurred warranty cost over the most recent
        12 months. ``adequacy_pct`` = accrual / incurred × 100.

        Status thresholds:
          • under   : adequacy < 95%  (reserve shortfall — actionable)
          • over    : adequacy > 110% (over-reserved)
          • adequate: otherwise
        """
        df = warranty_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        recent = df[df["date"] >= df["date"].max() - pd.DateOffset(months=12)]

        incurred = float(recent["warranty_cost_gbp"].sum())
        accrual = float(recent["accrual_gbp"].sum())
        adequacy_pct = (accrual / incurred * 100) if incurred > 0 else 100.0

        if adequacy_pct < 95.0:
            status = "under"
        elif adequacy_pct > 110.0:
            status = "over"
        else:
            status = "adequate"

        shortfall = max(incurred - accrual, 0.0)

        # Worst (most under-accrued) segment for targeting.
        seg_gap = (
            recent.groupby("segment")
            .apply(lambda g: g["accrual_gbp"].sum() - g["warranty_cost_gbp"].sum())
            .sort_values()
        )
        worst_segment = str(seg_gap.index[0]) if not seg_gap.empty else None

        return {
            "adequacy_pct": round(adequacy_pct, 2),
            "status": status,
            "shortfall_gbp": round(shortfall, 2),
            "incurred_gbp": round(incurred, 2),
            "accrual_gbp": round(accrual, 2),
            "worst_segment": worst_segment,
            "worst_segment_gap_gbp": round(float(seg_gap.iloc[0]), 2) if not seg_gap.empty else 0.0,
        }

    # ── 3. Failure-mode breakdown ─────────────────────────────────────────────
    def failure_mode_breakdown(self, warranty_df: pd.DataFrame) -> dict:
        """
        % share of warranty cost by dominant failure mode (most recent 12 months),
        and flag modes whose share is *rising* vs the prior 12 months.
        """
        df = warranty_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()

        recent = df[df["date"] > max_date - pd.DateOffset(months=12)]
        prior = df[
            (df["date"] <= max_date - pd.DateOffset(months=12))
            & (df["date"] > max_date - pd.DateOffset(months=24))
        ]

        def _shares(frame: pd.DataFrame) -> dict[str, float]:
            if frame.empty:
                return {}
            tot = frame["warranty_cost_gbp"].sum()
            if tot <= 0:
                return {}
            grp = frame.groupby("dominant_failure_mode")["warranty_cost_gbp"].sum()
            return {str(k): float(v / tot * 100) for k, v in grp.items()}

        recent_shares = _shares(recent)
        prior_shares = _shares(prior)

        rising = []
        for mode, share in recent_shares.items():
            prev = prior_shares.get(mode, 0.0)
            if share - prev > 2.0:  # >2 percentage-point increase
                rising.append({
                    "mode": mode,
                    "share_pct": round(share, 1),
                    "delta_pp": round(share - prev, 1),
                })
        rising.sort(key=lambda r: r["delta_pp"], reverse=True)

        breakdown = {
            m: round(s, 1)
            for m, s in sorted(recent_shares.items(), key=lambda kv: kv[1], reverse=True)
        }
        return {
            "breakdown_pct": breakdown,
            "rising_modes": rising,
            "dominant_mode": next(iter(breakdown), None),
        }

    # ── 4. Composite risk score ───────────────────────────────────────────────
    def warranty_risk_score(self, warranty_df: pd.DataFrame) -> dict:
        """
        0–100 composite warranty risk score blending three normalised components:

          • claims_trend   (40%): YoY change in claims-per-1000 (rising = worse)
          • severity_trend (30%): YoY change in avg claim severity (rising = worse)
          • accrual_gap    (30%): under-accrual relative to incurred (gap = worse)

        Higher score = higher warranty risk. Returns the score, a band, and the
        component contributions for transparency.
        """
        df = warranty_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()

        recent = df[df["date"] > max_date - pd.DateOffset(months=12)]
        prior = df[
            (df["date"] <= max_date - pd.DateOffset(months=12))
            & (df["date"] > max_date - pd.DateOffset(months=24))
        ]

        def _wmean(frame: pd.DataFrame, col: str) -> float:
            if frame.empty:
                return 0.0
            return float(frame[col].mean())

        # Claims trend (weighted by claims) — % change YoY.
        cpk_recent = _wmean(recent, "claims_per_1000")
        cpk_prior = _wmean(prior, "claims_per_1000")
        claims_chg = (cpk_recent - cpk_prior) / cpk_prior if cpk_prior > 0 else 0.0

        sev_recent = _wmean(recent, "avg_claim_gbp")
        sev_prior = _wmean(prior, "avg_claim_gbp")
        sev_chg = (sev_recent - sev_prior) / sev_prior if sev_prior > 0 else 0.0

        adequacy = self.assess_accrual_adequacy(df)
        # gap fraction: positive when under-accrued.
        gap_frac = max(0.0, (100.0 - adequacy["adequacy_pct"]) / 100.0)

        # Normalise each component to 0–100 with sensible saturation points.
        def _scale(x: float, sat: float) -> float:
            return float(np.clip(x / sat, -1.0, 1.0) * 50 + 50)

        claims_comp = _scale(claims_chg, 0.20)   # +20% YoY → ~100
        severity_comp = _scale(sev_chg, 0.20)
        accrual_comp = float(np.clip(gap_frac / 0.15, 0.0, 1.0) * 100)  # 15% gap → 100

        score = 0.40 * claims_comp + 0.30 * severity_comp + 0.30 * accrual_comp
        score = float(np.clip(score, 0, 100))

        if score >= 75:
            band = "critical"
        elif score >= 55:
            band = "high"
        elif score >= 35:
            band = "elevated"
        else:
            band = "low"

        return {
            "score": round(score, 1),
            "band": band,
            "components": {
                "claims_trend": round(claims_comp, 1),
                "severity_trend": round(severity_comp, 1),
                "accrual_gap": round(accrual_comp, 1),
            },
            "detail": {
                "claims_yoy_pct": round(claims_chg * 100, 1),
                "severity_yoy_pct": round(sev_chg * 100, 1),
                "accrual_adequacy_pct": adequacy["adequacy_pct"],
            },
        }
