# Layer 5 — Governance & Explainability

**Controller:** `GovernanceLayerController` (`controller.py`)
**LLM Engine:** `GICLLMEngine` (`llm_engine.py`)
**Entry point:** `controller.log_event(event_type, details)` | `controller.generate_narrative(forecast_result)`

## What this layer does

Provides the compliance, explainability, and AI narrative layer for the GIC platform.
Every significant system action is appended to an immutable JSONL audit trail.
An open-source LLM generates human-readable narratives without sending data to external APIs.

## LLM backend auto-detection

```
1. Ollama (llama3.2:1b)           — Best quality; run: ollama pull llama3.2:1b
2. HuggingFace transformers        — google/flan-t5-base; free, ~300MB, CPU-compatible
3. Template-based fallback         — Always available; zero external dependencies
```

## Bias tracking thresholds

```
>5%  forecast variance  →  alert logged to audit trail
>10% forecast variance  →  escalation event + CFO notification narrative
```

## Key modules

| Module | Role |
|---|---|
| `layers/layer5_governance/llm_engine.py` | Open-source LLM integration (Ollama / HuggingFace / template) |
| `src/governance/audit_trail.py` | Append-only JSONL audit log with UUID + ISO8601 timestamps |
| `src/governance/bias_tracking.py` | Forecast residual monitoring with alert/escalation thresholds |
| `src/governance/explainability.py` | XGBoost feature importance narratives and top-driver extraction |
