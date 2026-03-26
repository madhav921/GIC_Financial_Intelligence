"""
Market Intelligence Engine — enriches forecasts with real-world signals.

Combines data from multiple sources into actionable intelligence:
  - Real-time commodity price monitoring vs. forecast
  - Macro regime detection (expansion / contraction / crisis)
  - Cross-asset correlation monitoring
  - Alert generation for material deviations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class MarketAlert:
    """Actionable alert from market monitoring."""
    timestamp: str
    severity: str        # "info", "warning", "critical"
    category: str        # "commodity", "macro", "crypto", "correlation"
    title: str
    description: str
    metric_value: float
    threshold: float


@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of market conditions."""
    timestamp: str
    commodity_signals: dict[str, dict]
    macro_regime: str
    risk_level: str
    alerts: list[MarketAlert] = field(default_factory=list)
    crypto_correlation: float | None = None


class MarketIntelligence:
    """
    Monitors real-world market data and generates intelligence for CFO decisions.
    """

    def __init__(self):
        self.alert_history: list[MarketAlert] = []

    def analyze_commodity_trends(self, prices_df: pl.DataFrame) -> dict[str, dict]:
        """
        Analyze commodity price trends from real market data.

        Returns per-commodity signals: trend direction, momentum, vs. historical range.
        """
        results = {}
        date_col = "date"
        value_cols = [c for c in prices_df.columns if c != date_col]

        for col in value_cols:
            series = prices_df.select(pl.col(col).drop_nulls()).to_series()
            if len(series) < 20:
                continue

            current = float(series[-1])
            avg_30 = float(series[-30:].mean()) if len(series) >= 30 else float(series.mean())
            avg_90 = float(series[-90:].mean()) if len(series) >= 90 else float(series.mean())
            high_52w = float(series[-252:].max()) if len(series) >= 252 else float(series.max())
            low_52w = float(series[-252:].min()) if len(series) >= 252 else float(series.min())

            # Momentum
            momentum_30d = (current - avg_30) / avg_30 * 100 if avg_30 else 0
            momentum_90d = (current - avg_90) / avg_90 * 100 if avg_90 else 0

            # Position in 52-week range
            range_52w = high_52w - low_52w
            position_in_range = (current - low_52w) / range_52w * 100 if range_52w else 50

            # Volatility (trailing 30-period)
            returns = series.pct_change().drop_nulls()
            vol_30 = float(returns[-30:].std()) if len(returns) >= 30 else float(returns.std())

            # Trend classification
            if momentum_30d > 5:
                trend = "bullish"
            elif momentum_30d < -5:
                trend = "bearish"
            else:
                trend = "neutral"

            results[col] = {
                "current_price": round(current, 2),
                "avg_30d": round(avg_30, 2),
                "avg_90d": round(avg_90, 2),
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "momentum_30d_pct": round(momentum_30d, 2),
                "momentum_90d_pct": round(momentum_90d, 2),
                "position_in_52w_range_pct": round(position_in_range, 1),
                "trailing_volatility": round(vol_30, 4),
                "trend": trend,
            }

        return results

    def detect_macro_regime(self, macro_df: pl.DataFrame) -> str:
        """
        Detect current macroeconomic regime from indicators.

        Returns: "expansion", "slowdown", "contraction", or "crisis"
        """
        if macro_df.is_empty():
            return "unknown"

        cols = macro_df.columns
        signals = []

        # Yield curve signal (if available)
        if "T10Y2Y" in cols:
            spread = float(macro_df.select(pl.col("T10Y2Y").drop_nulls()).to_series()[-1])
            if spread < 0:
                signals.append("contraction")
            elif spread < 0.5:
                signals.append("slowdown")
            else:
                signals.append("expansion")

        # Unemployment signal
        if "UNRATE" in cols:
            unemp = macro_df.select(pl.col("UNRATE").drop_nulls()).to_series()
            if len(unemp) >= 6:
                recent_trend = float(unemp[-1]) - float(unemp[-6])
                if recent_trend > 1.0:
                    signals.append("contraction")
                elif recent_trend > 0.3:
                    signals.append("slowdown")
                else:
                    signals.append("expansion")

        # Consumer sentiment
        if "UMCSENT" in cols:
            sent = macro_df.select(pl.col("UMCSENT").drop_nulls()).to_series()
            if len(sent) > 0:
                current_sent = float(sent[-1])
                if current_sent < 60:
                    signals.append("crisis")
                elif current_sent < 75:
                    signals.append("contraction")
                elif current_sent < 90:
                    signals.append("slowdown")
                else:
                    signals.append("expansion")

        if not signals:
            return "unknown"

        # Majority vote
        from collections import Counter
        counts = Counter(signals)
        return counts.most_common(1)[0][0]

    def generate_alerts(
        self,
        commodity_signals: dict[str, dict],
        macro_regime: str,
    ) -> list[MarketAlert]:
        """Generate actionable alerts based on market conditions."""
        alerts = []
        now = datetime.now().isoformat()

        # Commodity alerts
        for commodity, signal in commodity_signals.items():
            mom = signal.get("momentum_30d_pct", 0)
            vol = signal.get("trailing_volatility", 0)

            if abs(mom) > 15:
                alerts.append(MarketAlert(
                    timestamp=now,
                    severity="critical",
                    category="commodity",
                    title=f"{commodity} price {'surge' if mom > 0 else 'crash'}",
                    description=f"{commodity} moved {mom:+.1f}% in 30 days. Review COGS and margin forecasts.",
                    metric_value=mom,
                    threshold=15.0,
                ))
            elif abs(mom) > 8:
                alerts.append(MarketAlert(
                    timestamp=now,
                    severity="warning",
                    category="commodity",
                    title=f"{commodity} significant {'up' if mom > 0 else 'down'}trend",
                    description=f"{commodity} moved {mom:+.1f}% in 30 days.",
                    metric_value=mom,
                    threshold=8.0,
                ))

            if vol > 0.06:
                alerts.append(MarketAlert(
                    timestamp=now,
                    severity="warning",
                    category="commodity",
                    title=f"{commodity} high volatility",
                    description=f"{commodity} trailing volatility at {vol:.1%}, above 6% threshold.",
                    metric_value=vol,
                    threshold=0.06,
                ))

        # Macro regime alert
        if macro_regime in ("contraction", "crisis"):
            alerts.append(MarketAlert(
                timestamp=now,
                severity="critical" if macro_regime == "crisis" else "warning",
                category="macro",
                title=f"Macro regime: {macro_regime.upper()}",
                description=f"Economy detected in {macro_regime} phase. Review demand forecasts and scenario stress tests.",
                metric_value=0,
                threshold=0,
            ))

        self.alert_history.extend(alerts)
        return alerts

    def create_snapshot(
        self,
        commodity_prices_df: pl.DataFrame,
        macro_df: pl.DataFrame | None = None,
    ) -> MarketSnapshot:
        """Create a comprehensive market snapshot for executive reporting."""
        commodity_signals = self.analyze_commodity_trends(commodity_prices_df)
        macro_regime = self.detect_macro_regime(macro_df) if macro_df is not None and not macro_df.is_empty() else "unknown"

        alerts = self.generate_alerts(commodity_signals, macro_regime)

        # Determine overall risk level
        critical_count = sum(1 for a in alerts if a.severity == "critical")
        warning_count = sum(1 for a in alerts if a.severity == "warning")
        if critical_count >= 2 or macro_regime == "crisis":
            risk_level = "high"
        elif critical_count >= 1 or warning_count >= 3:
            risk_level = "elevated"
        elif warning_count >= 1:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return MarketSnapshot(
            timestamp=datetime.now().isoformat(),
            commodity_signals=commodity_signals,
            macro_regime=macro_regime,
            risk_level=risk_level,
            alerts=alerts,
        )
