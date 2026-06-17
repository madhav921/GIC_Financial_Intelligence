"""
Shared dataclasses for the Actionable-Intelligence layer.

These are the canonical contracts consumed by the API (``routes/insights.py``)
and by Layer 5 (LLM governance) when it writes executive narratives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Allowed vocab (kept here so producers/consumers agree on terms).
CATEGORIES = (
    "commodity_risk",
    "margin",
    "demand",
    "warranty",
    "hedging",
    "cost",
    "opportunity",
)
SEVERITIES = ("info", "warning", "critical")


@dataclass
class InsightCard:
    """
    A single prioritised, prescriptive insight.

    An InsightCard is the atomic unit of the actionable-intelligence feed: it
    pairs a quantified *finding* with the *reasoning* behind it, the £ *impact*,
    a *recommended action*, and the £ *savings* that action is expected to unlock.
    """
    id: str
    category: str                 # one of CATEGORIES
    severity: str                 # info | warning | critical
    priority: int                 # 1 = highest .. 5 = lowest
    title: str
    finding: str
    reasoning: str
    impact_gbp: float
    impact_label: str
    confidence: float             # 0-1
    recommended_action: str
    expected_action_savings_gbp: float
    affected_segments: list[str] = field(default_factory=list)
    supporting_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VarianceBridge:
    """
    EBIT plan→actual variance decomposition (waterfall).

    ``bridge`` is an ordered list of driver contributions. Drivers are emitted in
    canonical order: Volume, Price/Mix, Commodity, FX, Warranty, Other. Each entry
    is ``{driver, delta_gbp, pct, direction, explanation}`` where ``direction`` is
    "favourable" or "adverse" and ``pct`` is the driver's share of total variance.
    """
    plan_ebit: float
    actual_ebit: float
    total_variance_gbp: float
    total_variance_pct: float
    bridge: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
