"""Tests for governance layer."""

import pandas as pd
import pytest

from src.governance.audit_trail import AuditTrail
from src.governance.bias_tracking import BiasTracker


class TestAuditTrail:
    def test_log_forecast(self, tmp_path, monkeypatch):
        # Patch audit dir to temp
        audit = AuditTrail()
        audit.audit_dir = tmp_path
        audit._log_file = tmp_path / "test_audit.jsonl"

        entry_id = audit.log_forecast(
            model_name="test_model",
            commodity="Lithium",
            forecast_values=[100, 110, 120],
            metrics={"mae": 5.0},
        )
        assert entry_id is not None

        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "forecast_generated"

    def test_log_override(self, tmp_path):
        audit = AuditTrail()
        audit.audit_dir = tmp_path
        audit._log_file = tmp_path / "test_audit.jsonl"

        audit.log_override(
            model_name="test", original_value=100, override_value=110,
            reason="Market insight", user="analyst",
        )
        entries = audit.get_entries(event_type="manual_override")
        assert len(entries) == 1
        assert entries[0]["override_pct"] == 10.0


class TestBiasTracker:
    def test_compute_bias(self):
        actuals = pd.Series([100, 110, 105, 115, 120, 108, 112, 118])
        forecasts = pd.Series([102, 112, 108, 118, 125, 112, 116, 122])

        tracker = BiasTracker()
        report = tracker.compute_bias(actuals, forecasts, "test", "Lithium")

        assert report.mean_bias_pct > 0  # Over-forecasting
        assert report.bias_direction == "over"
        assert report.n_observations == 8

    def test_bias_alert(self):
        actuals = pd.Series([100] * 10)
        forecasts = pd.Series([115] * 10)  # 15% over

        tracker = BiasTracker()
        report = tracker.compute_bias(actuals, forecasts, "test", "Steel")

        assert report.is_alert is True  # Exceeds 5% threshold
