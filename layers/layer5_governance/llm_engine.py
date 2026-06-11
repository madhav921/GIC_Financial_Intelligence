"""
GIC LLM Engine — Open-Source Language Model Integration

Provides natural language narrative generation for the Governance layer.

Supported backends (in priority order):
  1. Ollama (llama3.2:1b) — Best quality, requires: `ollama pull llama3.2:1b`
  2. HuggingFace transformers (google/flan-t5-base) — Free, no GPU, ~300MB
  3. Template-based fallback — Always available, zero dependencies

Usage:
    llm = GICLLMEngine()
    narrative = llm.explain_forecast("Copper", forecast_pct=7.2, drivers=["PMI", "DXY"])
    report    = llm.generate_risk_summary(scenario_results)
    alert     = llm.explain_alert("Lithium", alert_type="variance", variance_pct=12.3)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class LLMConfig:
    backend: str = "auto"          # "ollama" | "transformers" | "template"
    ollama_model: str = "llama3.2:1b"
    hf_model: str = "google/flan-t5-base"
    max_new_tokens: int = 256
    temperature: float = 0.3       # Low temperature for factual financial text
    ollama_host: str = "http://localhost:11434"


class GICLLMEngine:
    """
    Open-source LLM engine for GIC governance narrative generation.

    Auto-detects available backend:
      Ollama → HuggingFace transformers → Template fallback
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._backend: Optional[str] = None
        self._hf_pipeline = None
        self._backend = self._init_backend()
        logger.info(f"GIC LLM Engine initialized — backend: {self._backend}")

    # ── Backend initialization ────────────────────────────────────────────────

    def _init_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        # Try Ollama first
        if self._check_ollama():
            return "ollama"

        # Try HuggingFace transformers
        if self._init_transformers():
            return "transformers"

        # Template fallback
        logger.info("GIC LLM Engine: Using template-based fallback (no LLM available)")
        return "template"

    def _check_ollama(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self.config.ollama_host}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(self.config.ollama_model in m for m in models):
                    logger.info(f"GIC LLM Engine: Ollama available — {self.config.ollama_model}")
                    return True
                else:
                    logger.info(
                        f"GIC LLM Engine: Ollama running but {self.config.ollama_model} not pulled. "
                        f"Run: ollama pull {self.config.ollama_model}"
                    )
        except Exception:
            pass
        return False

    def _init_transformers(self) -> bool:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(
                f"GIC LLM Engine: Loading {self.config.hf_model} "
                f"(first run downloads ~300MB)"
            )
            self._hf_pipeline = pipeline(
                "text2text-generation",
                model=self.config.hf_model,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # Deterministic for financial reporting
            )
            logger.info(f"GIC LLM Engine: HuggingFace {self.config.hf_model} loaded")
            return True
        except ImportError:
            logger.warning(
                "GIC LLM Engine: transformers not installed. "
                "Run: pip install transformers torch sentencepiece"
            )
        except Exception as e:
            logger.warning(f"GIC LLM Engine: HuggingFace init failed — {e}")
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def explain_forecast(
        self,
        commodity: str,
        forecast_pct: float,
        drivers: list[str],
        current_price: float | None = None,
        forecast_price: float | None = None,
        mape: float | None = None,
    ) -> str:
        """Generate a natural language explanation of a commodity forecast."""
        direction = "rise" if forecast_pct > 0 else "fall"
        prompt = (
            f"Write a 2-sentence financial analysis for {commodity} commodity price forecast. "
            f"The model predicts a {abs(forecast_pct):.1f}% {direction} over the next 12 months. "
            f"Key drivers: {', '.join(drivers[:3])}. "
            f"{'Current price: $' + str(round(current_price, 2)) + '.' if current_price else ''} "
            f"{'Forecast accuracy (MAPE): ' + str(round(mape, 1)) + '%.' if mape else ''} "
            f"Focus on the business impact for an automotive manufacturer."
        )
        return self._generate(prompt)

    def explain_alert(
        self,
        commodity: str,
        alert_type: str,
        variance_pct: float,
        threshold_pct: float = 5.0,
    ) -> str:
        """Generate a governance alert narrative."""
        severity = "critical escalation" if variance_pct > 10 else "alert"
        prompt = (
            f"Write a 2-sentence governance {severity} for {commodity} commodity. "
            f"The forecast variance is {variance_pct:.1f}% against the {threshold_pct:.0f}% threshold. "
            f"Alert type: {alert_type}. "
            f"Recommend an immediate action for the finance team."
        )
        return self._generate(prompt)

    def generate_risk_summary(
        self,
        scenario_name: str,
        ebit_impact_pct: float,
        var_95: float,
        commodity_risk_pct: float,
        demand_risk_pct: float,
        fx_risk_pct: float,
    ) -> str:
        """Generate a risk scenario summary narrative."""
        direction = "downside" if ebit_impact_pct < 0 else "upside"
        prompt = (
            f"Write a 3-sentence CFO-level risk summary for the '{scenario_name}' scenario. "
            f"EBIT impact: {ebit_impact_pct:+.1f}% ({direction}). "
            f"VaR(95%): £{abs(var_95/1e6):.0f}M. "
            f"Risk sources: commodity {commodity_risk_pct:.0f}%, "
            f"demand {demand_risk_pct:.0f}%, FX {fx_risk_pct:.0f}%. "
            f"Include a hedging recommendation."
        )
        return self._generate(prompt)

    def generate_executive_insight(
        self,
        total_revenue: float,
        gross_margin_pct: float,
        ebit: float,
        commodity_index: float,
        top_risk_commodity: str,
    ) -> str:
        """Generate a top-line executive insight for the dashboard header."""
        prompt = (
            f"Write a 2-sentence executive insight for a board dashboard. "
            f"Total revenue: £{total_revenue/1e9:.2f}B. "
            f"Gross margin: {gross_margin_pct:.1f}%. "
            f"EBIT: £{ebit/1e6:.0f}M. "
            f"Commodity index: {commodity_index:.1f} (base=100). "
            f"Top risk: {top_risk_commodity}. "
            f"Tone: concise and strategic."
        )
        return self._generate(prompt)

    def explain_audit_event(self, event_type: str, details: dict) -> str:
        """Generate a human-readable description of an audit trail event."""
        details_str = ", ".join(f"{k}={v}" for k, v in list(details.items())[:5])
        prompt = (
            f"Write a 1-sentence audit log description. "
            f"Event: {event_type}. Details: {details_str}. "
            f"Be factual and specific."
        )
        return self._generate(prompt)

    def health_check(self) -> dict:
        """Return LLM engine status and backend info."""
        return {
            "backend": self._backend,
            "model": (
                self.config.ollama_model if self._backend == "ollama"
                else self.config.hf_model if self._backend == "transformers"
                else "template"
            ),
            "status": "healthy",
        }

    # ── Internal generation ───────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """Route prompt to the active backend."""
        try:
            if self._backend == "ollama":
                return self._generate_ollama(prompt)
            elif self._backend == "transformers":
                return self._generate_transformers(prompt)
            else:
                return self._generate_template(prompt)
        except Exception as e:
            logger.warning(f"LLM generation failed ({self._backend}): {e} — using template")
            return self._generate_template(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        import httpx
        resp = httpx.post(
            f"{self.config.ollama_host}/api/generate",
            json={
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_new_tokens,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        result = self._hf_pipeline(prompt, max_new_tokens=self.config.max_new_tokens)
        return result[0]["generated_text"].strip()

    def _generate_template(self, prompt: str) -> str:
        """Template-based fallback — extracts key info from prompt keywords."""
        prompt_lower = prompt.lower()

        # Detect context from prompt and return appropriate template
        if "forecast" in prompt_lower and "%" in prompt:
            # Extract commodity and direction
            words = prompt.split()
            commodity = next(
                (w for w in words if w[0].isupper() and len(w) > 3), "commodity"
            )
            pct = next((w for w in words if "%" in w), "N/A")
            direction = "upward" if "rise" in prompt_lower else "downward"
            return (
                f"{commodity} is forecast to trend {direction} ({pct}) over the next 12 months "
                f"based on current macro conditions and historical price patterns. "
                f"This is driven by supply-demand dynamics and macroeconomic indicators. "
                f"Monitor variance against actuals monthly and escalate if >10%."
            )
        elif "risk" in prompt_lower or "scenario" in prompt_lower:
            return (
                "This scenario represents a significant risk event requiring immediate attention "
                "from the finance team. "
                "The Monte Carlo simulation indicates elevated tail risk beyond normal operating "
                "parameters. "
                "Recommend reviewing hedge positions and adjusting procurement strategy accordingly."
            )
        elif "alert" in prompt_lower or "variance" in prompt_lower:
            return (
                "Forecast variance has exceeded the governance threshold, triggering a review "
                "requirement. "
                "The finance team should investigate the root cause and update the model assumptions. "
                "Document findings in the audit trail and notify the CFO."
            )
        elif "executive" in prompt_lower or "board" in prompt_lower:
            return (
                "Financial performance is within expected parameters with commodity costs as the "
                "primary risk factor. "
                "Hedging strategy is recommended to protect EBIT margins from commodity volatility."
            )
        else:
            return (
                "Analysis complete. The system has processed the financial data and generated "
                "insights based on current market conditions and model outputs."
            )
