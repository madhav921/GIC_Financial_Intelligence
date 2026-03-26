"""
Audit Trail (Layer 5 — Governance & Control)

Immutable logging of all model predictions, overrides, and decisions.
Every forecast, scenario run, and manual override is recorded with
full context for regulatory compliance and model governance.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_project_root, get_settings


class AuditTrail:
    """Append-only audit log for model governance."""

    def __init__(self):
        settings = get_settings()
        self.audit_dir = get_project_root() / settings["paths"]["audit_trail"]
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.audit_dir / "audit_log.jsonl"

    def _write_entry(self, entry: dict[str, Any]) -> str:
        """Append an audit entry. Returns the entry ID."""
        entry_id = str(uuid.uuid4())
        entry["entry_id"] = entry_id
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        return entry_id

    def log_forecast(
        self,
        model_name: str,
        commodity: str,
        forecast_values: list[float],
        metrics: dict[str, float],
        model_version: str = "",
        user: str = "system",
    ) -> str:
        """Log a forecast generation event."""
        return self._write_entry({
            "event_type": "forecast_generated",
            "model_name": model_name,
            "commodity": commodity,
            "forecast_horizon": len(forecast_values),
            "forecast_mean": sum(forecast_values) / len(forecast_values) if forecast_values else 0,
            "metrics": metrics,
            "model_version": model_version,
            "user": user,
        })

    def log_override(
        self,
        model_name: str,
        original_value: float,
        override_value: float,
        reason: str,
        user: str,
    ) -> str:
        """Log a manual override of a model prediction."""
        override_pct = abs(override_value - original_value) / max(abs(original_value), 1e-10) * 100
        settings = get_settings()
        max_override = settings["governance"]["max_override_pct"]

        entry = {
            "event_type": "manual_override",
            "model_name": model_name,
            "original_value": original_value,
            "override_value": override_value,
            "override_pct": round(override_pct, 2),
            "within_limit": override_pct <= max_override,
            "reason": reason,
            "user": user,
        }

        if override_pct > max_override:
            logger.warning(
                f"Override exceeds limit: {override_pct:.1f}% > {max_override}% "
                f"(user={user}, model={model_name})"
            )
            entry["alert"] = f"Override exceeds {max_override}% threshold"

        return self._write_entry(entry)

    def log_scenario_run(
        self,
        scenario_name: str,
        parameters: dict,
        result_summary: dict,
        user: str = "system",
    ) -> str:
        """Log a scenario simulation run."""
        return self._write_entry({
            "event_type": "scenario_run",
            "scenario_name": scenario_name,
            "parameters": parameters,
            "result_summary": result_summary,
            "user": user,
        })

    def log_data_ingestion(
        self,
        source: str,
        dataset: str,
        row_count: int,
        date_range: str = "",
    ) -> str:
        """Log a data ingestion event."""
        return self._write_entry({
            "event_type": "data_ingestion",
            "source": source,
            "dataset": dataset,
            "row_count": row_count,
            "date_range": date_range,
        })

    def get_entries(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Read audit entries, optionally filtered by event type."""
        if not self._log_file.exists():
            return []

        entries = []
        with open(self._log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if event_type is None or entry.get("event_type") == event_type:
                    entries.append(entry)

        return entries[-limit:]  # Return most recent

    def override_report(self) -> list[dict]:
        """Generate a report of all manual overrides."""
        return self.get_entries(event_type="manual_override")
