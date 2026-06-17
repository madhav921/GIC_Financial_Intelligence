"""
GIC LLM Engine — Open-Source Language Model Integration

Provides natural language narrative generation for the Governance layer.

Supported backends (in priority order):
  1. Ollama (llama3.2:1b) — Best quality, requires: `ollama pull llama3.2:1b`
  2. HuggingFace transformers (google/flan-t5-base) — Free, no GPU, ~300MB
  3. Template-based fallback — Always available, zero dependencies,
     produces rich JLR CFO-grade output using the actual data parameters

Usage:
    llm = GICLLMEngine()
    narrative = llm.explain_forecast("Copper", forecast_pct=0.072, drivers=["PMI", "DXY"],
                                      exposure_gbp=198_000_000, hedge_ratio=0.40)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class LLMConfig:
    backend: str = "auto"           # "ollama" | "transformers" | "template"
    ollama_model: str = "llama3.2:1b"
    hf_model: str = "google/flan-t5-base"
    max_new_tokens: int = 350
    temperature: float = 0.25       # Low for consistent, factual financial text
    ollama_host: str = "http://localhost:11434"


# ── JLR commodity context ─────────────────────────────────────────────────────
# Maps each commodity to the JLR vehicle programmes it affects, its use in the
# vehicle and the key market drivers. Used by every narrative backend.

_COMMODITY_CONTEXT: dict[str, dict] = {
    "Steel": {
        "programs": ["Defender", "Discovery", "Range Rover Sport"],
        "usage": "body-in-white stampings, underbody and chassis structural components",
        "drivers": ["global hot-rolled coil demand", "EU blast-furnace energy costs", "Chinese export volumes"],
        "note": "Steel is the single largest line in JLR's commodity basket at ~22% of strategic spend.",
        "action": (
            "Extend HRC forward contracts to 70% cover, aligned with committed Defender and "
            "Discovery build schedules. Consider a mix of quarterly LME swaps and fixed-price "
            "mill agreements to lock cost ahead of the annual model year change."
        ),
    },
    "Aluminum": {
        "programs": ["Range Rover", "Range Rover Sport", "Range Rover SV"],
        "usage": "lightweight all-aluminium body-in-white (40% weight saving vs steel) and closures",
        "drivers": ["LME aluminium spot", "European smelter energy tariffs", "automotive sheet order book"],
        "note": "Aluminium intensity is highest on the Luxury SUV platform, which carries JLR's highest gross margin per unit.",
        "action": (
            "Increase aluminium sheet hedging to 65% via LME swaps and negotiate fixed-price "
            "supply agreements with tier-1 sheet suppliers ahead of the Range Rover model year "
            "change. Evaluate domestic UK supply to reduce logistics and FX exposure."
        ),
    },
    "Copper": {
        "programs": ["I-Pace", "Range Rover PHEV", "Defender PHEV"],
        "usage": "EV and PHEV wiring harnesses, e-motor windings and on-board charging systems",
        "drivers": ["LME copper spot", "EV adoption acceleration", "Chilean and Peruvian mine supply"],
        "note": "Copper intensity rises approximately 4× from ICE to full BEV — exposure grows with the EV mix.",
        "action": (
            "Layer copper forwards across EV programme launch windows. Target 60% cover on "
            "committed I-Pace and Range Rover EV builds via LME futures. Align procurement "
            "cadence with wiring harness supplier (Aptiv/Lear) call-off schedules."
        ),
    },
    "Lithium": {
        "programs": ["I-Pace", "Range Rover Electric", "Defender PHEV"],
        "usage": "lithium-ion battery cells and cathode active material across all BEV and PHEV packs",
        "drivers": ["EV demand outpacing Australian and Chilean supply ramp", "Chinese cathode processing demand", "spodumene concentrate spot pricing"],
        "note": "Lithium is the most price-volatile material in JLR's portfolio; point forecasts carry inherent uncertainty. Multi-year offtake is preferred over spot exposure.",
        "action": (
            "Negotiate multi-year cathode offtake agreements indexed to spodumene rather than "
            "spot lithium carbonate. Target 70% coverage over the Range Rover Electric and "
            "next-generation I-Pace build horizon. Engage cell suppliers (Samsung SDI, LG Energy) "
            "to share price risk within the battery supply agreement."
        ),
    },
    "Cobalt": {
        "programs": ["I-Pace", "Range Rover PHEV", "Defender PHEV"],
        "usage": "NMC cathode active material in all lithium-ion battery packs",
        "drivers": ["DRC mining output and political risk (70% of global supply)", "Chinese battery demand", "NMC811 cobalt-reduction substitution"],
        "note": "Cobalt supply is critically DRC-concentrated — JLR's cobalt exposure carries a structural geopolitical risk premium.",
        "action": (
            "Accelerate transition to low-cobalt NMC811 and cobalt-free LFP chemistry in "
            "partnership with cell suppliers. Maintain a 3-month physical inventory buffer "
            "and hedge residual cobalt requirement via LME forwards."
        ),
    },
    "Nickel": {
        "programs": ["I-Pace", "Range Rover Electric", "Range Rover PHEV"],
        "usage": "high-nickel NMC811 cathode for energy density and driving range",
        "drivers": ["LME nickel spot", "Indonesian HPAL (Class 1) supply ramp", "stainless steel demand"],
        "note": "High-nickel cathode is JLR's strategic direction for next-generation platforms where range is a competitive differentiator.",
        "action": (
            "Align nickel procurement within battery cell supply agreements, specifying "
            "Class-1 nickel feedstock from certified Indonesian HPAL sources. Establish "
            "a rolling 12-month nickel forward book to protect EV programme margins."
        ),
    },
    "Natural_Gas": {
        "programs": ["All programmes (Solihull & Castle Bromwich)"],
        "usage": "paint shop, body shop and general services energy at JLR's UK manufacturing sites",
        "drivers": ["TTF European spot", "NBP UK domestic gas price", "seasonal heating demand and LNG import capacity"],
        "note": "Natural gas is the most volatile commodity in the basket; MAPE routinely exceeds 25%. Scenario analysis should accompany any point estimate.",
        "action": (
            "Implement a winter gas price collar (cap+floor) covering 50% of Q4–Q1 Solihull "
            "and Castle Bromwich consumption. Accelerate the renewable electricity transition "
            "programme to structurally reduce gas dependency over the planning horizon."
        ),
    },
    "Natural Gas": {
        "programs": ["All programmes (Solihull & Castle Bromwich)"],
        "usage": "paint shop, body shop and general services energy at JLR's UK manufacturing sites",
        "drivers": ["TTF European spot", "NBP UK domestic gas price", "seasonal heating demand and LNG import capacity"],
        "note": "Natural gas is the most volatile commodity in the basket; MAPE routinely exceeds 25%.",
        "action": (
            "Implement a winter gas price collar covering 50% of Q4–Q1 Solihull and Castle "
            "Bromwich consumption. Accelerate the renewable electricity transition programme "
            "to structurally reduce gas dependency."
        ),
    },
    "Platinum": {
        "programs": ["F-Pace", "E-Pace", "Range Rover Sport (ICE)"],
        "usage": "three-way catalytic converter PGM loading for NOx, HC and CO emission control",
        "drivers": ["South African supply concentration", "ICE production volumes", "EU7 emission tightening"],
        "note": "Platinum demand from JLR declines structurally as the fleet electrifies; the closed-loop recycling programme is increasing in value.",
        "action": (
            "Maintain current hedge cover and expand the closed-loop autocatalyst recycling "
            "programme to recover PGM from end-of-life vehicles. Review loadings against "
            "planned ICE volume decline on a quarterly basis."
        ),
    },
    "Palladium": {
        "programs": ["Defender", "Discovery", "F-Pace (petrol)"],
        "usage": "gasoline three-way catalytic converter PGM loading",
        "drivers": ["Russian supply risk (40% of global mine supply)", "ICE production decline", "platinum substitution progress"],
        "note": "Palladium is in long-run structural decline as fleet electrification accelerates; near-term Russian supply risk remains elevated.",
        "action": (
            "Accelerate platinum-for-palladium substitution in catalyst specification where "
            "homologation permits. Maintain a minimum 3-month physical inventory buffer "
            "to manage any supply disruption through the Russian risk window."
        ),
    },
    "Rhodium": {
        "programs": ["F-Pace", "Range Rover (petrol)"],
        "usage": "three-way catalyst for NOx emission reduction — smallest BOM loading, highest unit price",
        "drivers": ["South African supply concentration (>80%)", "ICE production volumes", "regulatory NOx limits"],
        "note": "Rhodium spot has historically ranged $1,000–$30,000/oz; it is the most price-volatile precious metal in the basket.",
        "action": (
            "Minimise rhodium loading through catalyst formulation optimisation with JLR's "
            "tier-1 catalyst supplier. Maintain a 3-month physical buffer and hedge any "
            "incremental requirements via OTC forwards with specialist PGM dealers."
        ),
    },
    "Polypropylene": {
        "programs": ["All programmes"],
        "usage": "interior trim panels, bumper fascias, door cards and underbody protection",
        "drivers": ["naphtha and propylene feedstock pricing", "European cracker utilisation", "crude oil"],
        "note": "Polypropylene tracks crude oil with a ~2–4 week lag; energy market conditions are the primary cost driver.",
        "action": (
            "Monitor naphtha forward curves and hedge up to 50% of annual PP requirement "
            "via feedstock-indexed pricing agreements with BASF and SABIC. Consolidate "
            "suppliers to improve negotiating leverage on fixed-price tranches."
        ),
    },
    "ABS_Resin": {
        "programs": ["Range Rover", "Discovery", "I-Pace"],
        "usage": "exterior body cladding, grille surrounds and interior structural hard plastics",
        "drivers": ["styrene monomer pricing", "acrylonitrile supply tightness", "butadiene availability"],
        "note": "ABS resin is a three-monomer blend; pricing can disconnect from crude oil when any monomer is in supply deficit.",
        "action": (
            "Negotiate 18–24 month fixed-price supply agreements with LG Chem and Trinseo "
            "to lock out spot market volatility. Target 60% fixed-price coverage, "
            "with the balance managed via formula-priced spot supply."
        ),
    },
    "ABS Resin": {
        "programs": ["Range Rover", "Discovery", "I-Pace"],
        "usage": "exterior body cladding, grille surrounds and interior structural hard plastics",
        "drivers": ["styrene monomer pricing", "acrylonitrile supply tightness", "butadiene availability"],
        "note": "ABS resin is a three-monomer blend; pricing can disconnect from crude oil when any monomer is in supply deficit.",
        "action": (
            "Negotiate 18–24 month fixed-price supply agreements with LG Chem and Trinseo "
            "to target 60% fixed-price coverage."
        ),
    },
}


class GICLLMEngine:
    """
    Open-source LLM engine for GIC governance narrative generation.

    Auto-detects available backend:
      Ollama → HuggingFace transformers → JLR CFO template (zero-dependency, always works)
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
        if self._check_ollama():
            return "ollama"
        if self._init_transformers():
            return "transformers"
        logger.info("GIC LLM Engine: Using JLR CFO template (no external LLM available)")
        return "template"

    def _check_ollama(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self.config.ollama_host}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(self.config.ollama_model in m for m in models):
                    logger.info(f"GIC LLM Engine: Ollama — {self.config.ollama_model}")
                    return True
        except Exception:
            pass
        return False

    def _init_transformers(self) -> bool:
        try:
            from transformers import pipeline
            logger.info(f"GIC LLM Engine: Loading {self.config.hf_model} (~300MB first run)")
            self._hf_pipeline = pipeline(
                "text2text-generation",
                model=self.config.hf_model,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
            logger.info(f"GIC LLM Engine: {self.config.hf_model} loaded")
            return True
        except ImportError:
            logger.warning("GIC LLM Engine: transformers not installed — pip install transformers torch")
        except Exception as e:
            logger.warning(f"GIC LLM Engine: HuggingFace init failed — {e}")
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def explain_forecast(
        self,
        commodity: str,
        forecast_pct: float,
        drivers: list[str],
        exposure_gbp: float = 0.0,
        hedge_ratio: float = 0.40,
        current_price: float | None = None,
        forecast_price: float | None = None,
        mape: float | None = None,
    ) -> str:
        """
        Generate a JLR CFO-grade procurement intelligence note for a commodity forecast.

        Args:
            commodity:    commodity name (matches _COMMODITY_CONTEXT keys)
            forecast_pct: fractional price change forecast (0.15 = +15%)
            drivers:      top 3 market drivers from the forecast model
            exposure_gbp: JLR annual GBP exposure to this commodity
            hedge_ratio:  current forward hedge coverage (0.40 = 40%)
            mape:         model MAPE% — shown as confidence caveat when >20%
        """
        if self._backend == "template":
            return self._template_forecast(
                commodity, forecast_pct, drivers, exposure_gbp, hedge_ratio, mape
            )
        prompt = self._prompt_forecast(
            commodity, forecast_pct, drivers, exposure_gbp, hedge_ratio, mape
        )
        return self._generate(prompt)

    def explain_alert(
        self,
        commodity: str,
        alert_type: str,
        variance_pct: float,
        threshold_pct: float = 5.0,
    ) -> str:
        """Generate a JLR governance escalation note for a bias or variance alert."""
        if self._backend == "template":
            return self._template_alert(commodity, alert_type, variance_pct, threshold_pct)
        prompt = self._prompt_alert(commodity, alert_type, variance_pct, threshold_pct)
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
        """Generate a CFO-level risk scenario summary for the Scenario Simulation page."""
        if self._backend == "template":
            return self._template_risk(
                scenario_name, ebit_impact_pct, var_95,
                commodity_risk_pct, demand_risk_pct, fx_risk_pct
            )
        prompt = self._prompt_risk(
            scenario_name, ebit_impact_pct, var_95,
            commodity_risk_pct, demand_risk_pct, fx_risk_pct
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
        """Generate a board-level executive insight for the dashboard header."""
        if self._backend == "template":
            return self._template_executive(
                total_revenue, gross_margin_pct, ebit, commodity_index, top_risk_commodity
            )
        prompt = self._prompt_executive(
            total_revenue, gross_margin_pct, ebit, commodity_index, top_risk_commodity
        )
        return self._generate(prompt)

    def explain_audit_event(self, event_type: str, details: dict) -> str:
        """Generate a human-readable description of an audit trail event."""
        if self._backend == "template":
            return self._template_audit(event_type, details)
        details_str = ", ".join(f"{k}={v}" for k, v in list(details.items())[:5])
        prompt = (
            f"You are a JLR governance system. Write one sentence describing this audit event.\n"
            f"Event type: {event_type}\nDetails: {details_str}\nBe factual and concise."
        )
        return self._generate(prompt)

    def health_check(self) -> dict:
        return {
            "backend": self._backend,
            "model": (
                self.config.ollama_model if self._backend == "ollama"
                else self.config.hf_model if self._backend == "transformers"
                else "template"
            ),
            "status": "healthy",
        }

    # ── Prompt builders (for Ollama / transformers) ───────────────────────────

    def _prompt_forecast(
        self, commodity: str, forecast_pct: float, drivers: list[str],
        exposure_gbp: float, hedge_ratio: float, mape: float | None
    ) -> str:
        ctx = _COMMODITY_CONTEXT.get(commodity, {})
        programs = ", ".join(ctx.get("programs", ["JLR programmes"])[:2])
        usage = ctx.get("usage", "production components")
        unhedged = exposure_gbp * (1 - hedge_ratio)
        ebit_risk = abs(unhedged * forecast_pct)
        direction = "increase" if forecast_pct > 0 else "decrease"
        action = ctx.get("action", "Review hedge positions and procurement strategy.")

        return (
            "You are a senior procurement intelligence analyst at Jaguar Land Rover (JLR), "
            "a British premium automotive OEM with £22B annual revenue and ~7% EBIT margin.\n\n"
            f"Write a 3-sentence CFO procurement briefing note on {commodity}.\n\n"
            f"Data points:\n"
            f"• Forecast: {abs(forecast_pct * 100):.0f}% {direction} over 12 months\n"
            f"• Primary drivers: {', '.join(drivers[:3])}\n"
            f"• JLR annual exposure: £{exposure_gbp / 1e6:.0f}M — {usage}\n"
            f"• Affected programmes: {programs}\n"
            f"• Current hedge cover: {hedge_ratio * 100:.0f}% → unhedged: £{unhedged / 1e6:.0f}M\n"
            f"• Estimated EBIT risk on unhedged position: £{ebit_risk / 1e6:.0f}M\n"
            f"{'• Forecast model MAPE: ' + str(round(mape, 1)) + '% (treat point estimate with caution)' if mape and mape > 15 else ''}\n\n"
            f"Cover: (1) price move and market context, (2) EBIT and programme risk, "
            f"(3) specific procurement action.\n"
            f"Suggested action to consider: {action}\n"
            f"Write in concise, factual business English. No jargon. No fluff."
        )

    def _prompt_alert(
        self, commodity: str, alert_type: str, variance_pct: float, threshold_pct: float
    ) -> str:
        ctx = _COMMODITY_CONTEXT.get(commodity, {})
        severity = "critical governance escalation" if variance_pct > 10 else "governance alert"
        return (
            f"You are a JLR governance officer. Write a 2-sentence {severity} note "
            f"for the CFO and Finance Director.\n\n"
            f"Commodity: {commodity}\n"
            f"Alert type: {alert_type.replace('_', ' ')}\n"
            f"Forecast variance: {variance_pct:.1f}% (governance threshold: {threshold_pct:.0f}%)\n"
            f"Affected programmes: {', '.join(ctx.get('programs', ['JLR programmes'])[:2])}\n\n"
            f"State the specific risk and the required immediate action. Be direct."
        )

    def _prompt_risk(
        self, scenario_name: str, ebit_impact_pct: float, var_95: float,
        commodity_risk_pct: float, demand_risk_pct: float, fx_risk_pct: float
    ) -> str:
        BASE_EBIT = 1_580_000_000
        ebit_impact_gbp = BASE_EBIT * ebit_impact_pct / 100
        direction = "downside" if ebit_impact_pct < 0 else "upside"
        return (
            "You are JLR's CFO decision-support system. "
            "Write a 3-sentence board-ready risk scenario note.\n\n"
            f"Scenario: '{scenario_name}'\n"
            f"EBIT impact: £{abs(ebit_impact_gbp / 1e6):.0f}M {direction} ({ebit_impact_pct:+.1f}%)\n"
            f"VaR(95%): £{abs(var_95 / 1e6):.0f}M (1-year loss not exceeded in 95% of simulations)\n"
            f"Risk decomposition: commodity {commodity_risk_pct:.0f}%, "
            f"demand {demand_risk_pct:.0f}%, FX {fx_risk_pct:.0f}%\n\n"
            f"Cover: (1) scenario headline, (2) primary risk driver and EBIT magnitude, "
            f"(3) hedging and mitigation recommendation.\n"
            f"Write in concise business English suitable for a board risk committee."
        )

    def _prompt_executive(
        self, total_revenue: float, gross_margin_pct: float, ebit: float,
        commodity_index: float, top_risk_commodity: str
    ) -> str:
        return (
            "You are JLR's CFO. Write a 2-sentence executive summary for the board dashboard.\n\n"
            f"Revenue: £{total_revenue / 1e9:.1f}B | Gross margin: {gross_margin_pct:.1f}% | "
            f"EBIT: £{ebit / 1e6:.0f}M\n"
            f"Commodity basket index: {commodity_index:.1f} (base=100) | "
            f"Highest risk: {top_risk_commodity}\n\n"
            f"Write: (1) P&L headline, (2) primary risk or opportunity requiring leadership attention.\n"
            f"Tone: board-level, strategic, factual. Under 60 words."
        )

    # ── Template generators (zero-dependency, JLR CFO-grade) ─────────────────

    def _template_forecast(
        self, commodity: str, forecast_pct: float, drivers: list[str],
        exposure_gbp: float, hedge_ratio: float, mape: float | None
    ) -> str:
        ctx = _COMMODITY_CONTEXT.get(commodity, {})
        programs = ctx.get("programs", ["all JLR programmes"])
        usage = ctx.get("usage", "production components")
        note = ctx.get("note", "")
        action = ctx.get("action", "Review hedge positions and procurement strategy.")

        # Compute from settings if exposure not passed
        if exposure_gbp <= 0:
            try:
                from src.config import get_settings
                for cfg in get_settings().get("commodities", []):
                    if cfg["name"].replace(" ", "_") == commodity.replace(" ", "_"):
                        exposure_gbp = 3_300_000_000 * float(cfg.get("bom_weight", 0.05))
                        break
            except Exception:
                exposure_gbp = 3_300_000_000 * 0.05

        unhedged_gbp = exposure_gbp * (1 - hedge_ratio)
        ebit_risk_gbp = abs(unhedged_gbp * forecast_pct)
        direction_word = "rise" if forecast_pct > 0 else "fall"
        pct_str = f"{abs(forecast_pct * 100):.0f}%"

        # Two affected programmes
        prog1 = programs[0] if programs else "key programmes"
        prog2 = programs[1] if len(programs) > 1 else ""
        prog_text = f"{prog1} and {prog2}" if prog2 else prog1

        # Driver text (first two)
        driver_text = (
            ", ".join(drivers[:2]) if drivers
            else "supply-demand fundamentals and macroeconomic conditions"
        )

        # Accuracy caveat
        mape_caveat = (
            f" (model MAPE {mape:.0f}% — treat as directional signal rather than precise point estimate)"
            if mape and mape > 20 else ""
        )

        # Sentence 1: market move
        s1 = (
            f"{commodity} is forecast to {direction_word} {pct_str} over the next 12 months, "
            f"driven by {driver_text}.{mape_caveat}"
        )

        # Sentence 2: JLR exposure and hedge position
        s2 = (
            f"Against JLR's £{exposure_gbp / 1e6:.0f}M annual {commodity.lower().replace('_', ' ')} "
            f"spend — {usage} across the {prog_text} programmes — the current "
            f"{hedge_ratio * 100:.0f}% forward cover leaves £{unhedged_gbp / 1e6:.0f}M unhedged."
        )

        # Sentence 3: EBIT impact
        if ebit_risk_gbp >= 1e6:
            s3 = (
                f"An unhedged {pct_str} move would add approximately "
                f"£{ebit_risk_gbp / 1e6:.0f}M of incremental cost pressure to EBIT — "
                f"{note}"
            )
        else:
            s3 = note or f"The EBIT impact is contained at current hedge levels."

        # Sentence 4: recommended action
        s4 = f"Recommended action: {action}"

        return f"{s1} {s2} {s3} {s4}"

    def _template_alert(
        self, commodity: str, alert_type: str, variance_pct: float, threshold_pct: float
    ) -> str:
        ctx = _COMMODITY_CONTEXT.get(commodity, {})
        programs = ", ".join(ctx.get("programs", ["JLR programmes"])[:2])
        severity_label = "Critical escalation" if variance_pct > 10 else "Governance alert"
        action_type = "require CFO sign-off" if variance_pct > 10 else "be reviewed by the Finance Director"

        return (
            f"{severity_label}: {commodity} forecast variance of {variance_pct:.1f}% has breached "
            f"the {threshold_pct:.0f}% governance threshold — the {alert_type.replace('_', ' ')} "
            f"for the {programs} programmes must {action_type} before the next procurement decision. "
            f"Finance team should update model assumptions, document root-cause analysis in the "
            f"audit trail and refresh the hedge recommendation within 5 business days."
        )

    def _template_risk(
        self, scenario_name: str, ebit_impact_pct: float, var_95: float,
        commodity_risk_pct: float, demand_risk_pct: float, fx_risk_pct: float
    ) -> str:
        BASE_EBIT = 1_580_000_000
        ebit_impact_gbp = BASE_EBIT * ebit_impact_pct / 100
        direction = "downside" if ebit_impact_pct < 0 else "upside"
        primary_driver = (
            "commodity cost inflation" if commodity_risk_pct >= demand_risk_pct and commodity_risk_pct >= fx_risk_pct
            else "demand softness" if demand_risk_pct >= fx_risk_pct
            else "FX movements"
        )
        hedge_rec = (
            "Extend forward commodity hedges to 65–70% cover and review FX book alignment "
            "with the purchasing calendar."
            if commodity_risk_pct >= 40
            else "Stress-test the demand plan against order intake and review FX natural hedge position."
        )

        return (
            f"The '{scenario_name}' scenario shows a £{abs(ebit_impact_gbp / 1e6):.0f}M EBIT "
            f"{direction} ({ebit_impact_pct:+.1f}%), with a 95th-percentile loss of "
            f"£{abs(var_95 / 1e6):.0f}M under the Monte Carlo distribution. "
            f"The dominant risk driver is {primary_driver} ({commodity_risk_pct:.0f}% of simulated "
            f"variance), followed by demand ({demand_risk_pct:.0f}%) and FX ({fx_risk_pct:.0f}%). "
            f"Finance recommendation: {hedge_rec}"
        )

    def _template_executive(
        self, total_revenue: float, gross_margin_pct: float, ebit: float,
        commodity_index: float, top_risk_commodity: str
    ) -> str:
        ebit_margin = ebit / total_revenue * 100 if total_revenue else 0
        index_signal = (
            "elevated commodity basket cost" if commodity_index > 105
            else "normalising commodity basket cost" if commodity_index < 95
            else "commodity basket broadly on plan"
        )
        return (
            f"JLR revenue of £{total_revenue / 1e9:.1f}B delivers a {gross_margin_pct:.1f}% "
            f"gross margin and £{ebit / 1e6:.0f}M EBIT ({ebit_margin:.1f}% margin), "
            f"with {index_signal} (index {commodity_index:.1f}). "
            f"Primary leadership focus: {top_risk_commodity} price trajectory — "
            f"current hedge position warrants a CFO review ahead of the next procurement cycle."
        )

    def _template_audit(self, event_type: str, details: dict) -> str:
        label = {
            "forecast_generated": "Commodity forecast generated",
            "narrative_generated": "LLM narrative generated",
            "bias_alert": "Forecast bias alert triggered",
            "bias_escalation": "Bias escalation raised to CFO",
            "pipeline_complete": "Full pipeline run completed",
            "scenario_run": "Monte Carlo scenario simulation executed",
            "data_ingestion": "Data ingestion event recorded",
            "manual_override": "Manual model override applied",
        }.get(event_type, event_type.replace("_", " ").capitalize())
        parts = [f"{k}: {v}" for k, v in list(details.items())[:3] if k not in ("entry_id", "timestamp", "event_type", "user")]
        detail_str = "; ".join(parts) if parts else ""
        return f"{label}{' — ' + detail_str if detail_str else ''}."

    # ── Internal generation ───────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        try:
            if self._backend == "ollama":
                return self._generate_ollama(prompt)
            elif self._backend == "transformers":
                return self._generate_transformers(prompt)
            else:
                return self._generate_template(prompt)
        except Exception as e:
            logger.warning(f"LLM generation failed ({self._backend}): {e} — using template fallback")
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
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        result = self._hf_pipeline(prompt, max_new_tokens=self.config.max_new_tokens)
        return result[0]["generated_text"].strip()

    def _generate_template(self, prompt: str) -> str:
        """Generic template fallback for unknown prompt types — extracts context from prompt text."""
        pl = prompt.lower()
        if "scenario" in pl or "simulation" in pl or "var" in pl:
            return (
                "This scenario represents a significant deviation from JLR's base financial plan, "
                "with commodity cost inflation as the primary EBIT driver. "
                "The Monte Carlo distribution indicates elevated tail risk in the 90th–95th percentile band. "
                "Recommended action: extend forward commodity hedges to 65–70% cover and stress-test "
                "the demand plan against current order intake."
            )
        elif "alert" in pl or "variance" in pl or "bias" in pl or "escalation" in pl:
            return (
                "Governance alert: forecast variance has exceeded the JLR Finance governance threshold. "
                "The Finance Director should review model assumptions, document root-cause analysis in "
                "the audit trail and refresh the procurement recommendation within 5 business days."
            )
        elif "board" in pl or "executive" in pl or "revenue" in pl:
            return (
                "JLR financial performance is tracking within plan, with commodity basket cost "
                "as the primary variance driver. "
                "Recommended leadership focus: hedge coverage adequacy ahead of the next "
                "commodity procurement cycle."
            )
        else:
            return (
                "Analysis complete. The GIC platform has processed current commodity and financial data "
                "and generated an updated risk position for JLR's procurement and finance teams. "
                "Review the detailed breakdown in the Commodity Intelligence and Scenario Simulation pages."
            )
