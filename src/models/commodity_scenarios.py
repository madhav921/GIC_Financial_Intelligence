"""
Commodity Scenario Framework & Variance Tracking (Layer 4)

Implements the JLR Commodity Price Forecast Model specification:
  - Bear / Base / Bull scenario analysis per commodity
  - Macro assumptions table driven scenarios
  - Monthly variance tracking and forecast logging
  - Escalation triggers (>5% variance alert, >10% governance review)
  - Feeds Commodity Index(t) into Financial Driver Engine (Layer 2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_project_root, get_settings


@dataclass
class MacroAssumptions:
    """Macro assumptions for scenario analysis."""
    global_manufacturing_pmi: float = 51.0
    china_gdp_growth: float = 4.75
    usd_dxy: float = 102.0
    energy_ttf: float = 35.0
    ev_demand_growth: float = 0.175
    supply_disruption_risk: str = "low"

    def scenario_label(self) -> str:
        if self.global_manufacturing_pmi < 49:
            return "bear"
        elif self.global_manufacturing_pmi > 52.5:
            return "bull"
        return "base"


@dataclass
class CommodityScenarioResult:
    """Result of a commodity scenario analysis."""
    commodity: str
    current_price: float
    bear_price: float
    base_price: float
    bull_price: float
    scenario_weights: dict[str, float]  # probability weights
    weighted_forecast: float
    horizon_months: int = 12


@dataclass
class VarianceRecord:
    """Monthly variance tracking record."""
    month: str
    commodity: str
    prior_forecast: float
    actual_price: float
    variance_pct: float
    reason: str = ""
    action_taken: str = ""
    escalation_level: str = "none"  # none, alert, governance_review


class CommodityScenarioEngine:
    """
    Commodity-specific scenario analysis engine.

    Translates macro assumptions into Bear/Base/Bull price paths
    for each JLR-relevant commodity. Integrates with the main
    ScenarioEngine for Monte Carlo overlay.
    """

    def __init__(self):
        self.settings = get_settings()
        self._scenario_config = self.settings.get("commodity_scenarios", {})
        self._price_scenarios = self._scenario_config.get("price_scenarios_12m", {})

    def get_scenario_prices(
        self, commodity: str
    ) -> dict[str, float]:
        """Get Bear/Base/Bull price targets for a commodity."""
        if commodity in self._price_scenarios:
            return self._price_scenarios[commodity]
        return {}

    def run_commodity_scenario(
        self,
        commodity: str,
        current_price: float | None = None,
        macro: MacroAssumptions | None = None,
        scenario_weights: dict[str, float] | None = None,
    ) -> CommodityScenarioResult:
        """
        Run Bear/Base/Bull scenario for a single commodity.

        Scenario weights default to (20% bear, 60% base, 20% bull)
        but shift based on macro assumptions.
        """
        prices = self.get_scenario_prices(commodity)
        if not prices:
            logger.warning(f"No scenario config for {commodity}, using ±20% from current")
            cp = current_price or 100.0
            prices = {
                "current": cp,
                "bear": cp * 0.80,
                "base": cp * 1.05,
                "bull": cp * 1.30,
            }

        cp = current_price or prices.get("current", 100.0)

        # Determine scenario weights from macro assumptions
        if scenario_weights is None:
            if macro is not None:
                scenario_weights = self._derive_weights_from_macro(macro)
            else:
                scenario_weights = {"bear": 0.20, "base": 0.60, "bull": 0.20}

        weighted = (
            scenario_weights["bear"] * prices["bear"]
            + scenario_weights["base"] * prices["base"]
            + scenario_weights["bull"] * prices["bull"]
        )

        return CommodityScenarioResult(
            commodity=commodity,
            current_price=cp,
            bear_price=prices["bear"],
            base_price=prices["base"],
            bull_price=prices["bull"],
            scenario_weights=scenario_weights,
            weighted_forecast=round(weighted, 2),
        )

    def run_all_commodity_scenarios(
        self,
        current_prices: dict[str, float] | None = None,
        macro: MacroAssumptions | None = None,
    ) -> dict[str, CommodityScenarioResult]:
        """Run scenario analysis for all configured commodities."""
        results = {}
        for commodity in self._price_scenarios:
            cp = (current_prices or {}).get(commodity)
            results[commodity] = self.run_commodity_scenario(commodity, cp, macro)
        return results

    def scenario_comparison_table(
        self,
        current_prices: dict[str, float] | None = None,
        macro: MacroAssumptions | None = None,
    ) -> pd.DataFrame:
        """Generate a comparison table of all commodity scenarios."""
        results = self.run_all_commodity_scenarios(current_prices, macro)
        records = []
        for commodity, r in results.items():
            records.append({
                "commodity": r.commodity,
                "current": r.current_price,
                "bear_12m": r.bear_price,
                "base_12m": r.base_price,
                "bull_12m": r.bull_price,
                "weighted_12m": r.weighted_forecast,
                "bear_chg_pct": round((r.bear_price / r.current_price - 1) * 100, 1),
                "bull_chg_pct": round((r.bull_price / r.current_price - 1) * 100, 1),
            })
        return pd.DataFrame(records)

    def _derive_weights_from_macro(self, macro: MacroAssumptions) -> dict[str, float]:
        """Shift probability weights based on macro environment."""
        # Base weights
        w_bear, w_base, w_bull = 0.20, 0.60, 0.20

        # PMI signal
        if macro.global_manufacturing_pmi < 48:
            w_bear += 0.15
            w_base -= 0.10
            w_bull -= 0.05
        elif macro.global_manufacturing_pmi > 53:
            w_bull += 0.15
            w_base -= 0.10
            w_bear -= 0.05

        # DXY signal (strong USD = commodity bearish)
        if macro.usd_dxy > 106:
            w_bear += 0.05
            w_bull -= 0.05
        elif macro.usd_dxy < 98:
            w_bull += 0.05
            w_bear -= 0.05

        # Supply disruption
        if macro.supply_disruption_risk == "high":
            w_bull += 0.10
            w_base -= 0.05
            w_bear -= 0.05

        # Normalize
        total = w_bear + w_base + w_bull
        return {
            "bear": round(max(w_bear / total, 0.05), 3),
            "base": round(max(w_base / total, 0.10), 3),
            "bull": round(max(w_bull / total, 0.05), 3),
        }


class VarianceTracker:
    """
    Monthly forecast variance tracking and escalation.

    Tracks forecast vs actual, flags variances, and triggers
    governance reviews when thresholds are breached.
    """

    def __init__(self):
        self.settings = get_settings()
        self._log_path = (
            get_project_root()
            / self.settings["paths"]["audit_trail"]
            / "variance_log.jsonl"
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_variance(
        self,
        month: str,
        commodity: str,
        prior_forecast: float,
        actual_price: float,
        reason: str = "",
        action_taken: str = "",
    ) -> VarianceRecord:
        """Record a monthly forecast variance."""
        variance_pct = ((actual_price - prior_forecast) / prior_forecast) * 100

        # Determine escalation level
        if abs(variance_pct) > 10:
            escalation = "governance_review"
        elif abs(variance_pct) > 5:
            escalation = "alert"
        else:
            escalation = "none"

        record = VarianceRecord(
            month=month,
            commodity=commodity,
            prior_forecast=prior_forecast,
            actual_price=actual_price,
            variance_pct=round(variance_pct, 2),
            reason=reason,
            action_taken=action_taken,
            escalation_level=escalation,
        )

        # Persist to JSONL
        self._append_log(record)

        if escalation == "governance_review":
            logger.warning(
                f"ESCALATION: {commodity} variance {variance_pct:+.1f}% "
                f"exceeds 10% — L6 Governance review triggered"
            )
        elif escalation == "alert":
            logger.warning(
                f"ALERT: {commodity} variance {variance_pct:+.1f}% exceeds 5%"
            )

        return record

    def get_history(
        self, commodity: str | None = None, limit: int = 100
    ) -> list[VarianceRecord]:
        """Read variance history from log."""
        if not self._log_path.exists():
            return []

        records = []
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if commodity is None or data.get("commodity") == commodity:
                    records.append(VarianceRecord(**{
                        k: v for k, v in data.items()
                        if k in VarianceRecord.__dataclass_fields__
                    }))

        return records[-limit:]

    def variance_summary(self, commodity: str | None = None) -> pd.DataFrame:
        """Return variance history as a DataFrame."""
        records = self.get_history(commodity)
        if not records:
            return pd.DataFrame(columns=[
                "month", "commodity", "prior_forecast", "actual_price",
                "variance_pct", "escalation_level",
            ])
        return pd.DataFrame([
            {
                "month": r.month,
                "commodity": r.commodity,
                "prior_forecast": r.prior_forecast,
                "actual_price": r.actual_price,
                "variance_pct": r.variance_pct,
                "escalation_level": r.escalation_level,
            }
            for r in records
        ])

    def _append_log(self, record: VarianceRecord) -> None:
        """Append a variance record to JSONL log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "month": record.month,
            "commodity": record.commodity,
            "prior_forecast": record.prior_forecast,
            "actual_price": record.actual_price,
            "variance_pct": record.variance_pct,
            "reason": record.reason,
            "action_taken": record.action_taken,
            "escalation_level": record.escalation_level,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


class MonthlyUpdatePipeline:
    """
    Orchestrates the monthly commodity price update cycle.

    Step-by-step workflow from the spec:
    1. Pull latest prices
    2. Check macro signals
    3. Run model refresh
    4. Revise scenarios
    5. Flag variance (>5% change)
    6. Escalate (>10% → L6 Governance review)
    7. Feed L2 (update Commodity Index(t))
    """

    def __init__(self):
        self.settings = get_settings()
        self.scenario_engine = CommodityScenarioEngine()
        self.variance_tracker = VarianceTracker()

    def run_monthly_update(
        self,
        current_prices: dict[str, float],
        prior_forecasts: dict[str, float],
        macro: MacroAssumptions | None = None,
        update_month: str | None = None,
    ) -> dict:
        """
        Execute the full monthly update cycle.

        Returns dict with:
          - scenario_table: DataFrame of updated scenarios
          - variance_records: list of VarianceRecord
          - escalations: list of commodities requiring governance review
          - updated_index_change: commodity index change vs prior month
        """
        if update_month is None:
            update_month = datetime.now().strftime("%b %Y")

        logger.info(f"Running monthly commodity update for {update_month}")

        # Step 1-2: Latest prices and macro signals already provided
        logger.info(f"Step 1-2: {len(current_prices)} commodity prices received")

        # Step 4: Revise scenarios
        scenario_table = self.scenario_engine.scenario_comparison_table(
            current_prices, macro
        )
        logger.info(f"Step 4: Scenarios revised for {len(scenario_table)} commodities")

        # Step 5-6: Flag variance and escalate
        variance_records = []
        escalations = []
        for commodity, actual_price in current_prices.items():
            if commodity in prior_forecasts:
                record = self.variance_tracker.record_variance(
                    month=update_month,
                    commodity=commodity,
                    prior_forecast=prior_forecasts[commodity],
                    actual_price=actual_price,
                )
                variance_records.append(record)
                if record.escalation_level == "governance_review":
                    escalations.append(commodity)

        if escalations:
            logger.warning(f"Step 6: ESCALATION required for: {escalations}")
        else:
            logger.info("Step 6: No escalations required")

        return {
            "update_month": update_month,
            "scenario_table": scenario_table,
            "variance_records": variance_records,
            "escalations": escalations,
            "num_commodities_updated": len(current_prices),
        }
