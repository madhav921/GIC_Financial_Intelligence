"""
Anomaly Detector (Actionable-Intelligence Layer).

Lightweight statistical anomaly detection over time series — used to flag
unexpected jumps in commodity prices, margins, warranty claims, or demand so the
InsightEngine can surface them as early-warning signals.

Two methods:
  • "zscore" — rolling-window z-score; flags points beyond ``threshold`` σ.
  • "iqr"    — rolling inter-quartile-range fence (robust to outliers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class AnomalyDetector:
    """Rolling-window anomaly detection for univariate series."""

    def __init__(self, window: int = 12, threshold: float = 2.5) -> None:
        self.window = window
        self.threshold = threshold

    def detect(
        self,
        series: pd.Series | list[float] | np.ndarray,
        method: str = "zscore",
        dates: list | None = None,
    ) -> list[dict]:
        """
        Detect anomalies in ``series``. Returns a list of
        ``{date, value, zscore, severity}`` for flagged points, newest first.
        """
        s = self._coerce(series, dates)
        if len(s) < max(4, self.window // 2):
            return []

        if method == "iqr":
            flags = self._iqr(s)
        else:
            flags = self._zscore(s)

        flags.sort(key=lambda f: f["date"], reverse=True)
        if flags:
            logger.info(f"Anomaly detect ({method}): {len(flags)} point(s) flagged")
        return flags

    def detect_in_frame(
        self,
        df: pd.DataFrame,
        value_cols: list[str],
        date_col: str = "date",
        method: str = "zscore",
    ) -> dict[str, list[dict]]:
        """Run :meth:`detect` across multiple columns of a DataFrame."""
        out: dict[str, list[dict]] = {}
        dates = df[date_col].tolist() if date_col in df.columns else None
        for col in value_cols:
            if col in df.columns:
                out[col] = self.detect(df[col], method=method, dates=dates)
        return out

    # ── internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _coerce(series, dates) -> pd.Series:
        if isinstance(series, pd.Series):
            s = series.copy()
            if dates is not None:
                s.index = pd.to_datetime(dates)
        else:
            idx = pd.to_datetime(dates) if dates is not None else pd.RangeIndex(len(series))
            s = pd.Series(np.asarray(series, dtype=float), index=idx)
        return s.dropna()

    def _zscore(self, s: pd.Series) -> list[dict]:
        roll_mean = s.rolling(self.window, min_periods=max(3, self.window // 2)).mean()
        roll_std = s.rolling(self.window, min_periods=max(3, self.window // 2)).std()
        z = (s - roll_mean) / roll_std.replace(0, np.nan)
        flags = []
        for idx, zv in z.items():
            if pd.notna(zv) and abs(zv) >= self.threshold:
                flags.append(self._record(idx, float(s.loc[idx]), float(zv)))
        return flags

    def _iqr(self, s: pd.Series) -> list[dict]:
        flags = []
        for i in range(self.window, len(s)):
            win = s.iloc[i - self.window:i]
            q1, q3 = np.percentile(win, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            val = float(s.iloc[i])
            if val < lo or val > hi:
                centre = (q1 + q3) / 2
                pseudo_z = (val - centre) / (iqr / 1.349 + 1e-9)
                flags.append(self._record(s.index[i], val, float(pseudo_z)))
        return flags

    @staticmethod
    def _record(idx, value: float, z: float) -> dict:
        az = abs(z)
        severity = "critical" if az >= 3.5 else "warning" if az >= 2.5 else "info"
        date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        return {
            "date": date,
            "value": round(value, 4),
            "zscore": round(z, 2),
            "severity": severity,
        }
