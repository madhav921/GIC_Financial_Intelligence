"""
GIC Plan-to-Perform — 5-Layer Architecture

  Layer 1: Data Architecture      (layers/layer1_data/)
  Layer 2: Predictive Intelligence (layers/layer2_intelligence/)
  Layer 3: Financial Drivers       (layers/layer3_financial/)
  Layer 4: Simulation & Risk       (layers/layer4_simulation/)
  Layer 5: Governance & LLM        (layers/layer5_governance/)

Entry point: orchestrator.py → GICOrchestrator
"""
from layers.layer1_data.controller import DataLayerController
from layers.layer2_intelligence.controller import IntelligenceLayerController
from layers.layer3_financial.controller import FinancialLayerController
from layers.layer4_simulation.controller import SimulationLayerController
from layers.layer5_governance.controller import GovernanceLayerController

__all__ = [
    "DataLayerController",
    "IntelligenceLayerController",
    "FinancialLayerController",
    "SimulationLayerController",
    "GovernanceLayerController",
]
