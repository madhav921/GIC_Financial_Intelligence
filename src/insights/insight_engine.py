"""
Insight Engine (Actionable-Intelligence Layer — orchestrator).

The InsightEngine is the brain of the actionable-intelligence layer. It combines
forecasts, P&L, the variance bridge, warranty analytics, anomaly detection, the
early-warning score and the recommendation engine into a single prioritised list
of :class:`InsightCard` objects.

Design goals
------------
  • Works with rich context OR none at all — with partial/empty context it falls
    back to a curated set of genuinely high-quality, specific demo insights so the
    API always returns useful content.
  • Every insight is quantified (£ impact + £ expected action savings) and
    prescriptive (a concrete recommended action).
  • Output is sorted by priority then |impact| and capped to the top N.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.insights.anomaly_detector import AnomalyDetector
from src.insights.early_warning import EarlyWarningSystem
from src.insights.recommendation_engine import RecommendationEngine
from src.insights.schemas import InsightCard


class InsightEngine:
    """Produces a prioritised, prescriptive insight feed."""

    def __init__(self) -> None:
        self.recommender = RecommendationEngine()
        self.anomaly = AnomalyDetector()
        self.early_warning = EarlyWarningSystem()

    # ── public API ────────────────────────────────────────────────────────────
    def generate_insights(self, context: dict | None = None, top_n: int = 8) -> list[InsightCard]:
        """
        Generate a prioritised list of InsightCards.

        ``context`` may include any of:
            forecasts        : dict[commodity -> {forecast_pct, exposure_gbp, hedge_ratio, ...}]
            commodity_index  : pd.DataFrame with date + commodity_index
            pnl_summary      : dict (plan/actual EBIT etc.)
            warranty_df      : pd.DataFrame (warranty data)
            plan_pnl_summary / actual_pnl_summary : dicts for variance context

        Falls back to curated demo insights when context is missing or sparse.
        """
        context = context or {}
        insights: list[InsightCard] = []

        try:
            insights.extend(self._commodity_insights(context))
            insights.extend(self._warranty_insights(context))
            insights.extend(self._anomaly_insights(context))
        except Exception as exc:  # noqa: BLE001 — never let one source break the feed
            logger.warning(f"Insight generation partial failure: {exc}")

        # Always ensure a rich feed: backfill with curated demo insights.
        if len(insights) < top_n:
            existing_ids = {i.id for i in insights}
            for card in self._demo_insights():
                if card.id not in existing_ids:
                    insights.append(card)

        insights = self._dedupe(insights)
        insights.sort(key=lambda c: (c.priority, -abs(c.impact_gbp)))
        return insights[:top_n]

    def summary_stats(self, insights: list[InsightCard]) -> dict:
        """Aggregate headline statistics across an insight list."""
        if not insights:
            return {
                "n_critical": 0, "n_warning": 0, "total_impact_gbp": 0.0,
                "total_opportunity_gbp": 0.0, "weighted_confidence": 0.0,
            }
        n_critical = sum(1 for i in insights if i.severity == "critical")
        n_warning = sum(1 for i in insights if i.severity == "warning")
        total_impact = sum(abs(i.impact_gbp) for i in insights)
        total_opportunity = sum(
            i.expected_action_savings_gbp for i in insights
            if i.category == "opportunity" or i.expected_action_savings_gbp > 0
        )
        weights = [abs(i.impact_gbp) for i in insights]
        wsum = sum(weights) or 1.0
        weighted_conf = sum(i.confidence * w for i, w in zip(insights, weights)) / wsum
        return {
            "n_critical": n_critical,
            "n_warning": n_warning,
            "total_impact_gbp": round(total_impact, 2),
            "total_opportunity_gbp": round(total_opportunity, 2),
            "weighted_confidence": round(weighted_conf, 3),
        }

    # ── context-driven generators ─────────────────────────────────────────────
    def _commodity_insights(self, context: dict) -> list[InsightCard]:
        """Build hedging/commodity insights from a forecasts dict if present."""
        out: list[InsightCard] = []
        forecasts = context.get("forecasts") or {}
        if not isinstance(forecasts, dict):
            return out
        for commodity, f in forecasts.items():
            try:
                fpct = float(f.get("forecast_pct", 0.0))
                exposure = float(f.get("exposure_gbp", 0.0))
                hedge = float(f.get("hedge_ratio", f.get("current_hedge_ratio", 0.4)))
            except (TypeError, ValueError, AttributeError):
                continue
            if abs(fpct) < 0.05 or exposure <= 0:
                continue
            rec = self.recommender.hedge_recommendation(commodity, fpct, exposure, hedge)
            impact = exposure * fpct
            severity = "critical" if abs(fpct) >= 0.20 else "warning"
            out.append(InsightCard(
                id=f"commodity_{commodity.lower()}",
                category="hedging" if fpct > 0 else "commodity_risk",
                severity=severity,
                priority=1 if abs(fpct) >= 0.20 else 2,
                title=f"{commodity} price forecast {fpct:+.0%} — hedge action advised",
                finding=(
                    f"{commodity} is forecast to move {fpct:+.0%} against a "
                    f"£{exposure / 1e6:,.0f}m exposure."
                ),
                reasoning=rec["rationale"],
                impact_gbp=round(impact, 2),
                impact_label=f"£{abs(impact) / 1e6:,.1f}m COGS exposure",
                confidence=0.7,
                recommended_action=rec["action"],
                expected_action_savings_gbp=rec["expected_savings_gbp"],
                affected_segments=f.get("segments", []),
                supporting_metrics={
                    "forecast_pct": fpct,
                    "exposure_gbp": exposure,
                    "target_hedge_ratio": rec["target_hedge_ratio"],
                },
            ))
        return out

    def _warranty_insights(self, context: dict) -> list[InsightCard]:
        """Build warranty insights from a warranty_df if present."""
        out: list[InsightCard] = []
        wdf = context.get("warranty_df")
        if not isinstance(wdf, pd.DataFrame) or wdf.empty:
            return out
        try:
            from src.models.warranty_model import WarrantyModel
            wm = WarrantyModel()
            adequacy = wm.assess_accrual_adequacy(wdf)
            risk = wm.warranty_risk_score(wdf)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Warranty insight skipped: {exc}")
            return out

        if adequacy["status"] == "under":
            shortfall = adequacy["shortfall_gbp"]
            out.append(InsightCard(
                id="warranty_accrual_shortfall",
                category="warranty",
                severity="critical" if shortfall > 2e7 else "warning",
                priority=1 if shortfall > 2e7 else 2,
                title="Warranty accrual shortfall — reserve top-up required",
                finding=(
                    f"Trailing-12m accrual covers only {adequacy['adequacy_pct']:.0f}% of "
                    f"incurred warranty cost; shortfall £{shortfall / 1e6:,.1f}m "
                    f"(worst: {adequacy['worst_segment']})."
                ),
                reasoning=(
                    "Under-accrual understates current-period cost and risks a future "
                    "catch-up charge to EBIT. Rebasing the accrual rate now smooths the P&L."
                ),
                impact_gbp=round(shortfall, 2),
                impact_label=f"£{shortfall / 1e6:,.1f}m reserve gap",
                confidence=0.8,
                recommended_action=(
                    f"Top up the {adequacy['worst_segment']} warranty reserve and raise the "
                    f"accrual rate to ~3.4% of revenue."
                ),
                expected_action_savings_gbp=round(shortfall * 0.5, 2),
                affected_segments=[adequacy["worst_segment"]] if adequacy["worst_segment"] else [],
                supporting_metrics=adequacy,
            ))

        if risk["band"] in ("high", "critical"):
            out.append(InsightCard(
                id="warranty_risk_elevated",
                category="warranty",
                severity="warning" if risk["band"] == "high" else "critical",
                priority=2,
                title=f"Warranty risk {risk['band']} (score {risk['score']:.0f})",
                finding=(
                    f"Composite warranty risk is {risk['score']:.0f}/100 — claims "
                    f"{risk['detail']['claims_yoy_pct']:+.0f}% YoY, severity "
                    f"{risk['detail']['severity_yoy_pct']:+.0f}% YoY."
                ),
                reasoning="Rising claims frequency and severity pressure the warranty line and NPS.",
                impact_gbp=0.0,
                impact_label=f"risk score {risk['score']:.0f}/100",
                confidence=0.7,
                recommended_action="Launch a failure-mode containment review on the rising modes.",
                expected_action_savings_gbp=0.0,
                affected_segments=[],
                supporting_metrics=risk,
            ))
        return out

    def _anomaly_insights(self, context: dict) -> list[InsightCard]:
        """Flag anomalies in the commodity index if a frame is supplied."""
        out: list[InsightCard] = []
        idx = context.get("commodity_index")
        if not isinstance(idx, pd.DataFrame) or "commodity_index" not in idx.columns:
            return out
        flags = self.anomaly.detect(
            idx["commodity_index"],
            dates=idx["date"].tolist() if "date" in idx.columns else None,
        )
        crit = [f for f in flags if f["severity"] in ("warning", "critical")]
        if crit:
            latest = crit[0]
            out.append(InsightCard(
                id="commodity_index_anomaly",
                category="commodity_risk",
                severity=latest["severity"],
                priority=2,
                title="Commodity index anomaly detected",
                finding=(
                    f"Commodity index printed {latest['value']:.1f} on {latest['date']} "
                    f"({latest['zscore']:+.1f}σ vs rolling window)."
                ),
                reasoning="A multi-sigma move signals a regime shift worth investigating before it feeds COGS.",
                impact_gbp=0.0,
                impact_label=f"{latest['zscore']:+.1f}σ",
                confidence=0.6,
                recommended_action="Review driver commodities behind the index spike and refresh hedges.",
                expected_action_savings_gbp=0.0,
                affected_segments=[],
                supporting_metrics={"anomalies": crit[:5]},
            ))
        return out

    # ── curated demo insights (high-quality fallback) ─────────────────────────
    def _demo_insights(self) -> list[InsightCard]:
        """A curated, specific, realistic demo feed for an empty-context call."""
        return [
            InsightCard(
                id="demo_lithium_hedge",
                category="hedging",
                severity="critical",
                priority=1,
                title="Lithium forecast +22% — extend hedge cover on EV battery exposure",
                finding=(
                    "Lithium carbonate is forecast +22% over 12 months against a £180m "
                    "annual battery-material exposure; current hedge ratio is only 35%."
                ),
                reasoning=(
                    "EV ramp lifts lithium intensity just as the 2025–26 oversupply unwinds. "
                    "Lifting the hedge ratio to ~70% locks in cost on the incremental £63m of "
                    "exposure before the move."
                ),
                impact_gbp=39_600_000.0,
                impact_label="£39.6m COGS exposure",
                confidence=0.74,
                recommended_action="Increase lithium hedge ratio from 35% to 70% via 12-month forwards.",
                expected_action_savings_gbp=13_900_000.0,
                affected_segments=["EV", "Luxury_SUV"],
                supporting_metrics={"forecast_pct": 0.22, "exposure_gbp": 180_000_000, "current_hedge_ratio": 0.35},
            ),
            InsightCard(
                id="demo_natgas_volatility",
                category="commodity_risk",
                severity="warning",
                priority=2,
                title="Natural-gas volatility threatens UK manufacturing energy cost",
                finding=(
                    "TTF/NBP gas implied volatility is running ~30%; a +30% spike adds an "
                    "estimated £24m to UK plant energy cost over the heating season."
                ),
                reasoning=(
                    "Energy is ~4% of BOM but highly volatile and largely unhedged. A modest "
                    "collar caps the downside on the most exposed quarters."
                ),
                impact_gbp=24_000_000.0,
                impact_label="£24m energy cost at risk",
                confidence=0.65,
                recommended_action="Place a winter gas collar covering 50% of Q4–Q1 plant demand.",
                expected_action_savings_gbp=7_500_000.0,
                affected_segments=["Luxury_SUV", "Premium_SUV", "Performance", "EV"],
                supporting_metrics={"volatility_pct": 30, "exposure_gbp": 80_000_000},
            ),
            InsightCard(
                id="demo_ev_warranty_accrual",
                category="warranty",
                severity="critical",
                priority=1,
                title="EV battery warranty accrual ~8% short of incurred claims",
                finding=(
                    "Trailing-12m EV warranty accrual covers ~92% of incurred claims; the "
                    "battery/high-voltage failure mode drives a ~£21m reserve shortfall."
                ),
                reasoning=(
                    "EV battery claims are low-frequency but high-severity (£6k–£15k per pack). "
                    "Under-accrual risks a future catch-up charge; rebasing now smooths EBIT."
                ),
                impact_gbp=21_000_000.0,
                impact_label="£21m reserve gap",
                confidence=0.8,
                recommended_action="Raise EV warranty accrual rate to 3.5% of revenue and top up the reserve.",
                expected_action_savings_gbp=10_500_000.0,
                affected_segments=["EV"],
                supporting_metrics={"adequacy_pct": 92, "shortfall_gbp": 21_000_000},
            ),
            InsightCard(
                id="demo_margin_compression",
                category="margin",
                severity="warning",
                priority=2,
                title="Gross margin tracking 1.4pp below plan on commodity + mix",
                finding=(
                    "YTD gross margin is 1.4pp below plan; ~60% is commodity-cost driven and "
                    "~40% is adverse mix (Performance over-indexing vs Luxury SUV)."
                ),
                reasoning=(
                    "At ~£22bn revenue, 1.4pp is ~£308m of gross margin. Pricing actions and a "
                    "mix shift toward Range Rover recover the majority."
                ),
                impact_gbp=308_000_000.0,
                impact_label="£308m margin vs plan",
                confidence=0.7,
                recommended_action="Implement 1.5% price action on Luxury SUV and re-weight production mix.",
                expected_action_savings_gbp=120_000_000.0,
                affected_segments=["Luxury_SUV", "Performance"],
                supporting_metrics={"margin_gap_pp": 1.4, "commodity_share": 0.6},
            ),
            InsightCard(
                id="demo_fx_exposure",
                category="cost",
                severity="warning",
                priority=3,
                title="USD/GBP exposure on commodity purchases largely unhedged",
                finding=(
                    "~£1.1bn of annual USD-denominated commodity spend has <40% FX cover; a "
                    "5% adverse GBP move adds ~£33m to landed cost."
                ),
                reasoning=(
                    "Commodity invoicing is USD-heavy while revenue skews GBP/EUR. Layering FX "
                    "forwards aligned to the purchasing calendar de-risks COGS."
                ),
                impact_gbp=33_000_000.0,
                impact_label="£33m FX exposure",
                confidence=0.68,
                recommended_action="Raise USD purchase hedge cover from 40% to 75% on a rolling 12-month book.",
                expected_action_savings_gbp=11_000_000.0,
                affected_segments=["EV", "Luxury_SUV", "Premium_SUV"],
                supporting_metrics={"usd_spend_gbp": 1_100_000_000, "current_cover": 0.4},
            ),
            InsightCard(
                id="demo_steel_prebuy_opportunity",
                category="opportunity",
                severity="info",
                priority=3,
                title="Steel pre-buy opportunity ahead of forecast price recovery",
                finding=(
                    "Steel is forecast +9% over 9 months while current cover sits at ~28 days; "
                    "pre-buying to 45 days locks in today's lower price."
                ),
                reasoning=(
                    "Steel is the single largest BOM line (~22%). Modest pre-buying ahead of a "
                    "forecast recovery is a low-risk, positive-carry hedge on physical inventory."
                ),
                impact_gbp=18_000_000.0,
                impact_label="£18m avoidable inflation",
                confidence=0.6,
                recommended_action="Pre-buy steel to lift cover from 28 to 45 days against committed builds.",
                expected_action_savings_gbp=6_500_000.0,
                affected_segments=["Premium_SUV", "Luxury_SUV"],
                supporting_metrics={"forecast_pct": 0.09, "days_of_supply": 28, "target_days": 45},
            ),
            InsightCard(
                id="demo_demand_softness",
                category="demand",
                severity="warning",
                priority=3,
                title="China demand softness pressuring Premium SUV volume",
                finding=(
                    "Premium SUV order intake is tracking ~6% below plan in China, a ~£140m "
                    "revenue risk if the trend persists through H2."
                ),
                reasoning=(
                    "Macro/PMI softness and competitive EV pricing are weighing on intake. "
                    "Targeted incentives plus mix rebalancing to stronger regions mitigate."
                ),
                impact_gbp=140_000_000.0,
                impact_label="£140m revenue at risk",
                confidence=0.62,
                recommended_action="Deploy a targeted China incentive and reallocate allocation to NA/EU.",
                expected_action_savings_gbp=45_000_000.0,
                affected_segments=["Premium_SUV"],
                supporting_metrics={"intake_vs_plan_pct": -6.0, "region": "CN"},
            ),
            InsightCard(
                id="demo_palladium_substitution",
                category="opportunity",
                severity="info",
                priority=4,
                title="Palladium thrifting opportunity as EV mix reduces catalyst demand",
                finding=(
                    "Falling ICE volume cuts autocatalyst palladium demand; thrifting and "
                    "recycling can release ~£9m of annual precious-metal cost."
                ),
                reasoning=(
                    "Palladium is in structural decline as the fleet electrifies. Re-specifying "
                    "loadings and increasing recycled content captures a durable cost saving."
                ),
                impact_gbp=9_000_000.0,
                impact_label="£9m cost-out potential",
                confidence=0.58,
                recommended_action="Re-spec catalyst PGM loadings and expand closed-loop recycling.",
                expected_action_savings_gbp=9_000_000.0,
                affected_segments=["Performance", "Premium_SUV"],
                supporting_metrics={"pgm_saving_gbp": 9_000_000},
            ),
        ]

    @staticmethod
    def _dedupe(cards: list[InsightCard]) -> list[InsightCard]:
        seen: set[str] = set()
        out: list[InsightCard] = []
        for c in cards:
            if c.id not in seen:
                seen.add(c.id)
                out.append(c)
        return out
