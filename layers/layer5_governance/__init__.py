"""
Layer 5 — Governance & Explainability

Responsibilities:
  - Append-only JSONL audit trail (immutable, CFO-ready compliance logging)
  - Open-source LLM narrative generation (HuggingFace flan-t5 / Ollama llama3.2)
  - Forecast bias tracking: >5% → alert, >10% → L6 escalation
  - Feature importance explainability (XGBoost shap-like top-drivers)
  - Market intelligence alerts (regime, volatility, correlation breaks)

LLM Integration:
  Primary:   HuggingFace transformers — google/flan-t5-base (free, CPU-compatible)
  Enhanced:  Ollama — llama3.2:1b (better quality, requires Ollama installation)
  Fallback:  Template-based narratives (always available, no model needed)

Audit trail format (JSONL, append-only):
  {
    "id": "<uuid>",
    "timestamp": "<ISO8601>",
    "event_type": "forecast_generated | manual_override | scenario_run | ...",
    "details": {...},
    "user": "<user_id>"
  }

Key modules:
  layers/layer5_governance/llm_engine.py  — LLM narrative generation (NEW)
  src/governance/audit_trail.py           — JSONL audit logging
  src/governance/bias_tracking.py         — Forecast residual monitoring
  src/governance/explainability.py        — Feature importance narratives
"""
from layers.layer5_governance.controller import GovernanceLayerController
from layers.layer5_governance.llm_engine import GICLLMEngine
__all__ = ["GovernanceLayerController", "GICLLMEngine"]
