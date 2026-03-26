"""
Price Elasticity Model (Layer 3 — Predictive Intelligence)

Estimates how demand responds to price changes (own-price elasticity)
and how commodity cost pressures affect pricing power.

Outputs:
  - Elasticity coefficients by segment and region
  - Optimal price point recommendations
  - Price sensitivity bands for scenario simulation

Feeds into: Revenue Drivers (Layer 2), Scenario Simulation (Layer 4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ElasticityResult:
    segment: str
    own_price_elasticity: float
    incentive_elasticity: float
    commodity_cross_elasticity: float
    r_squared: float
    coefficients: dict[str, float]


class PriceElasticityModel:
    """
    Log-log regression model for estimating price elasticity of demand.

    Model: ln(Q) = α + β₁·ln(P) + β₂·ln(Incentive) + β₃·ln(CommodityIndex) + controls
    β₁ = own-price elasticity (expected negative)
    β₂ = incentive elasticity (expected positive)
    β₃ = commodity cross-elasticity
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: dict[str, Ridge] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.results: dict[str, ElasticityResult] = {}

    def _prepare_log_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform data to log-log specification."""
        log_df = pd.DataFrame()
        log_df["ln_volume"] = np.log(df["volume"].clip(lower=1))
        log_df["ln_price"] = np.log(df["avg_price_usd"].clip(lower=1))
        log_df["ln_incentive"] = np.log((df["incentive_pct"] + 0.01).clip(lower=0.001))

        if "commodity_index" in df.columns:
            log_df["ln_commodity_idx"] = np.log(df["commodity_index"].clip(lower=1))

        # Add macro controls if available
        for col in ["gdp_growth_pct", "interest_rate_pct"]:
            if col in df.columns:
                log_df[col] = df[col]

        # Calendar
        if "month" in df.columns:
            log_df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            log_df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        y = log_df["ln_volume"]
        X = log_df.drop(columns=["ln_volume"])
        return X, y

    def fit(
        self,
        segment: str,
        sales_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity_df: pd.DataFrame,
    ) -> ElasticityResult:
        """Estimate elasticities for a vehicle segment."""
        seg_df = sales_df[sales_df["segment"] == segment].copy()

        # Aggregate monthly
        monthly = seg_df.groupby("date").agg(
            volume=("volume", "sum"),
            avg_price_usd=("avg_price_usd", "mean"),
            incentive_pct=("incentive_pct", "mean"),
        ).reset_index()

        # Merge macro + commodity index
        monthly = monthly.merge(macro_df, on="date", how="inner")

        comm_cols = [c for c in commodity_df.columns if c != "date"]
        commodity_df = commodity_df.copy()
        commodity_df["commodity_index"] = commodity_df[comm_cols].mean(axis=1)
        monthly = monthly.merge(commodity_df[["date", "commodity_index"]], on="date", how="inner")
        monthly["month"] = monthly["date"].dt.month

        monthly = monthly.dropna().reset_index(drop=True)

        X, y = self._prepare_log_features(monthly)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        model = Ridge(alpha=self.alpha)
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)

        # Extract elasticities (coefficients on log terms)
        coef_dict = dict(zip(X.columns, model.coef_))

        result = ElasticityResult(
            segment=segment,
            own_price_elasticity=coef_dict.get("ln_price", 0.0),
            incentive_elasticity=coef_dict.get("ln_incentive", 0.0),
            commodity_cross_elasticity=coef_dict.get("ln_commodity_idx", 0.0),
            r_squared=r2,
            coefficients=coef_dict,
        )

        self.models[segment] = model
        self.scalers[segment] = scaler
        self.results[segment] = result

        logger.info(
            f"Elasticity for {segment}: price={result.own_price_elasticity:.3f}, "
            f"incentive={result.incentive_elasticity:.3f}, R²={r2:.3f}"
        )
        return result

    def fit_all_segments(
        self,
        sales_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity_df: pd.DataFrame,
    ) -> dict[str, ElasticityResult]:
        from src.config import get_settings
        settings = get_settings()
        results = {}
        for seg in settings["vehicle_segments"]:
            segment = seg["segment"]
            try:
                results[segment] = self.fit(segment, sales_df, macro_df, commodity_df)
            except Exception as e:
                logger.error(f"Elasticity estimation failed for {segment}: {e}")
        return results

    def summary_table(self) -> pd.DataFrame:
        """Return a summary DataFrame of all elasticity estimates."""
        records = []
        for seg, result in self.results.items():
            records.append({
                "segment": seg,
                "own_price_elasticity": round(result.own_price_elasticity, 4),
                "incentive_elasticity": round(result.incentive_elasticity, 4),
                "commodity_cross_elasticity": round(result.commodity_cross_elasticity, 4),
                "r_squared": round(result.r_squared, 4),
            })
        return pd.DataFrame(records)
