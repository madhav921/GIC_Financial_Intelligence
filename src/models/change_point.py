"""
Online Change-Point / Structural-Break Detection (Layer 3 — AI Forecast Layer)

Complements the existing Hurst-exponent RegimeDetector (`regime_detector.py`) by
detecting *abrupt* structural breaks in commodity price series — events the slow,
window-averaged Hurst statistic reacts to only with a lag.

Two complementary detectors are provided:

  1. CUSUM (Cumulative Sum control chart; Page 1954) — a classic sequential
     change detector. We accumulate signed deviations from a running mean and
     flag a break when the cumulative sum exceeds a data-scaled threshold.
     Robust, parameter-light, excellent for mean shifts. Direction (up/down)
     follows the sign of the triggering arm.

  2. BOCPD (Bayesian Online Change-Point Detection; Adams & MacKay 2007,
     "Bayesian Online Changepoint Detection", arXiv:0710.3742). We implement
     the simplified Gaussian-observation variant: maintain a run-length
     posterior P(r_t | x_{1:t}) with a constant hazard H = 1/lambda and a
     Normal predictive model with running sufficient statistics. The change
     probability at each step is the mass that the run length collapses to 0.

Why this beats Hurst alone for dashboard alerting:
  * CUSUM/BOCPD fire on the *step* a regime shifts; Hurst needs a full window
    (~12+ obs) of the new regime before H crosses its threshold.
  * BOCPD yields a calibrated per-step change probability — directly usable as
    an alert confidence on the dashboard.

References:
  - Page, E.S. (1954). "Continuous Inspection Schemes." Biometrika 41(1/2).
  - Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint
    Detection." arXiv:0710.3742.

Degrades gracefully: depends only on numpy/pandas (always present). scipy is
used opportunistically for the Student-t predictive; falls back to a Gaussian
predictive if scipy is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ChangePoint:
    """A single detected structural break."""
    index: int
    date: str | None
    direction: str          # "up" | "down"
    magnitude: float        # signed shift estimate in series units
    confidence: float       # 0..1


@dataclass
class RegimeShiftAlert:
    """Dashboard-facing summary of the most recent break."""
    shifted: bool
    last_break_index: int | None = None
    last_break_date: str | None = None
    confidence: float = 0.0
    direction: str | None = None
    n_breaks: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ChangePointDetector:
    """
    Online change-point / structural-break detector for commodity series.

    Methods
    -------
    detect_cusum(series, threshold=None, drift=0.0)
        Cumulative-sum breaks with direction.
    detect_online(series, hazard_lambda=250.0)
        Simplified BOCPD: per-step change probability + run-length expectation.
    latest_regime_shift(series, ...)
        Compact dict-style alert for dashboards.
    """

    def __init__(self, min_segment: int = 8):
        # Minimum spacing between reported breaks (avoid chatter).
        self.min_segment = max(2, int(min_segment))

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _coerce(series: pd.Series | np.ndarray | list) -> tuple[np.ndarray, list[str] | None]:
        """Return (values, optional ISO date labels) from any series-like input."""
        dates: list[str] | None = None
        if isinstance(series, pd.Series):
            if isinstance(series.index, pd.DatetimeIndex):
                dates = [d.strftime("%Y-%m-%d") for d in series.index]
            values = series.to_numpy(dtype=float)
        else:
            values = np.asarray(series, dtype=float)
        values = values[~np.isnan(values)] if values.ndim == 1 else values.astype(float)
        return values, dates

    def _date_at(self, dates: list[str] | None, idx: int) -> str | None:
        if dates is not None and 0 <= idx < len(dates):
            return dates[idx]
        return None

    # ──────────────────────────────────────────────────────────────────────
    # 1. CUSUM
    # ──────────────────────────────────────────────────────────────────────
    def detect_cusum(
        self,
        series: pd.Series | np.ndarray | list,
        threshold: float | None = None,
        drift: float = 0.0,
    ) -> list[ChangePoint]:
        """
        Tabular two-sided CUSUM (Page 1954).

        Parameters
        ----------
        threshold : float | None
            Decision threshold in *units of series std*. If None, defaults to
            5.0 * sigma which gives few false alarms on noisy commodity data.
        drift : float
            Allowance (slack) k in units of series std; deviations smaller than
            this are treated as noise. Default 0.5*sigma is applied if 0.

        Returns
        -------
        list[ChangePoint] sorted by index.
        """
        values, dates = self._coerce(series)
        n = len(values)
        if n < self.min_segment * 2:
            logger.warning(f"detect_cusum: series too short (n={n}); returning [].")
            return []

        sigma = float(np.std(values)) or 1.0
        k = (drift if drift > 0 else 0.5) * sigma
        h = (threshold if threshold is not None else 5.0) * sigma

        s_pos = 0.0
        s_neg = 0.0
        ref = float(values[0])           # running reference mean
        breaks: list[ChangePoint] = []
        last_break = -self.min_segment

        for i in range(1, n):
            diff = values[i] - ref
            s_pos = max(0.0, s_pos + diff - k)
            s_neg = min(0.0, s_neg + diff + k)

            triggered = None
            if s_pos > h:
                triggered = "up"
            elif s_neg < -h:
                triggered = "down"

            if triggered and (i - last_break) >= self.min_segment:
                # Estimate magnitude as mean(after) - mean(before) over local window.
                w = min(self.min_segment, i, n - i)
                before = float(np.mean(values[max(0, i - w):i]))
                after = float(np.mean(values[i:i + w]))
                mag = after - before
                conf = float(min(1.0, max(s_pos, -s_neg) / (h + 1e-9)))
                breaks.append(
                    ChangePoint(
                        index=i,
                        date=self._date_at(dates, i),
                        direction=triggered,
                        magnitude=mag,
                        confidence=conf,
                    )
                )
                last_break = i
                # Reset accumulators and re-anchor reference to new regime.
                s_pos = s_neg = 0.0
                ref = after
            else:
                # Slowly track the running mean to adapt to drift.
                ref += (values[i] - ref) / max(2, i)

        logger.info(f"detect_cusum: {len(breaks)} break(s) found (n={n}, h={h:.3f}).")
        return breaks

    # ──────────────────────────────────────────────────────────────────────
    # 2. Simplified BOCPD (Adams & MacKay 2007)
    # ──────────────────────────────────────────────────────────────────────
    def detect_online(
        self,
        series: pd.Series | np.ndarray | list,
        hazard_lambda: float = 250.0,
        mu0: float | None = None,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Simplified Bayesian Online Change-Point Detection with a Normal
        observation model and Normal-Gamma conjugate prior (Student-t
        predictive). Returns per-step change probability and expected run
        length.

        Returns
        -------
        dict with:
            "change_prob"   : np.ndarray, P(run length resets to 0) per step.
            "run_length"    : np.ndarray, expected run length (E[r_t]).
            "map_run_length": np.ndarray, argmax run length per step.
        """
        values, _ = self._coerce(series)
        n = len(values)
        if n == 0:
            return {"change_prob": np.array([]), "run_length": np.array([]),
                    "map_run_length": np.array([])}

        try:
            from scipy.stats import t as student_t  # type: ignore
            have_scipy = True
        except Exception:  # pragma: no cover - scipy almost always present
            have_scipy = False

        H = 1.0 / float(hazard_lambda)        # constant hazard
        if mu0 is None:
            mu0 = float(np.mean(values[: min(n, 10)]))

        # Run-length posterior; index r = run length.
        R = np.zeros(n + 1)
        R[0] = 1.0
        # Conjugate Normal-Gamma sufficient stats, one entry per run length.
        mu = np.array([mu0])
        kappa = np.array([kappa0])
        alpha = np.array([alpha0])
        beta = np.array([beta0])

        change_prob = np.zeros(n)
        exp_run = np.zeros(n)
        map_run = np.zeros(n, dtype=int)

        for t in range(n):
            x = values[t]

            # Predictive probability under each run-length hypothesis.
            if have_scipy:
                df = 2.0 * alpha
                scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
                pred = student_t.pdf(x, df=df, loc=mu, scale=scale)
            else:
                var = beta * (kappa + 1.0) / (alpha * kappa)
                pred = np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
            pred = np.clip(pred, 1e-300, None)

            growth = R[: t + 1] * pred * (1.0 - H)
            cp = float(np.sum(R[: t + 1] * pred * H))

            new_R = np.zeros(t + 2)
            new_R[0] = cp
            new_R[1: t + 2] = growth
            total = new_R.sum()
            if total <= 0:
                new_R[0] = 1.0
                total = 1.0
            new_R /= total
            R = new_R

            change_prob[t] = new_R[0]
            exp_run[t] = float(np.dot(np.arange(t + 2), new_R))
            map_run[t] = int(np.argmax(new_R))

            # Update sufficient statistics (prepend the r=0 prior).
            new_mu = (kappa * mu + x) / (kappa + 1.0)
            new_kappa = kappa + 1.0
            new_alpha = alpha + 0.5
            new_beta = beta + (kappa * (x - mu) ** 2) / (2.0 * (kappa + 1.0))

            mu = np.concatenate(([mu0], new_mu))
            kappa = np.concatenate(([kappa0], new_kappa))
            alpha = np.concatenate(([alpha0], new_alpha))
            beta = np.concatenate(([beta0], new_beta))

        return {
            "change_prob": change_prob,
            "run_length": exp_run,
            "map_run_length": map_run,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 3. Dashboard alert
    # ──────────────────────────────────────────────────────────────────────
    def latest_regime_shift(
        self,
        series: pd.Series | np.ndarray | list,
        recent_window: int = 6,
        cusum_threshold: float | None = None,
    ) -> RegimeShiftAlert:
        """
        Produce a compact alert describing whether a structural break has
        occurred within the last `recent_window` observations, combining CUSUM
        (for the break location/direction) with BOCPD (for confidence).
        """
        values, dates = self._coerce(series)
        n = len(values)
        if n < self.min_segment * 2:
            return RegimeShiftAlert(shifted=False)

        breaks = self.detect_cusum(values if dates is None else series,
                                   threshold=cusum_threshold)
        online = self.detect_online(values)
        cp = online["change_prob"]

        if not breaks:
            # Fall back to BOCPD-only signal if CUSUM found nothing.
            recent_cp = cp[-recent_window:] if len(cp) else np.array([0.0])
            peak = float(np.max(recent_cp)) if recent_cp.size else 0.0
            shifted = peak > 0.5
            return RegimeShiftAlert(
                shifted=shifted,
                confidence=round(peak, 3),
                n_breaks=0,
                extra={"source": "bocpd_only"},
            )

        last = breaks[-1]
        within = (n - 1 - last.index) <= recent_window
        # Blend CUSUM confidence with BOCPD change prob near the break.
        lo, hi = max(0, last.index - 1), min(len(cp), last.index + 2)
        bocpd_conf = float(np.max(cp[lo:hi])) if hi > lo else 0.0
        confidence = round(0.5 * last.confidence + 0.5 * bocpd_conf, 3)

        return RegimeShiftAlert(
            shifted=bool(within),
            last_break_index=last.index,
            last_break_date=last.date,
            confidence=confidence,
            direction=last.direction,
            n_breaks=len(breaks),
            extra={"magnitude": round(last.magnitude, 4), "source": "cusum+bocpd"},
        )


def quick_demo() -> None:
    """Fabricate a series with an injected mean shift and verify detection."""
    rng = np.random.default_rng(7)
    shift_at = 100
    series = np.r_[rng.normal(0.0, 1.0, shift_at),
                   rng.normal(5.0, 1.0, 100)]

    det = ChangePointDetector()
    breaks = det.detect_cusum(series)
    assert breaks, "CUSUM found no breaks on a clear mean shift"
    first = breaks[0].index
    print(f"[CUSUM] first break at index {first} (injected at {shift_at}), "
          f"dir={breaks[0].direction}, conf={breaks[0].confidence:.2f}")
    assert abs(first - shift_at) <= 15, f"break {first} far from injection {shift_at}"

    online = det.detect_online(series)
    peak_idx = int(np.argmax(online["change_prob"][1:]) + 1)
    print(f"[BOCPD] peak change-prob at index {peak_idx} "
          f"(p={online['change_prob'][peak_idx]:.3f})")

    alert = det.latest_regime_shift(series)
    print(f"[alert] shifted={alert.shifted} dir={alert.direction} "
          f"conf={alert.confidence} n_breaks={alert.n_breaks}")
    print("change_point quick_demo OK")


if __name__ == "__main__":
    quick_demo()
