"""
Split-Conformal Prediction Intervals (Layer 3 — Uncertainty Calibration)

DISTRIBUTION-FREE, provably-calibrated prediction intervals that wrap ANY point
forecaster (SARIMAX, XGBoost, the ensemble, ...). Unlike parametric SARIMAX
confidence intervals — which assume Gaussian innovations and a correctly
specified model — split conformal prediction guarantees marginal coverage of
(1 - alpha) regardless of model misspecification, as long as calibration and
test residuals are exchangeable.

Why this beats parametric SARIMAX CIs for GIC:
    * SARIMAX CIs widen/narrow based on a (usually wrong) Gaussian innovation
      assumption; under fat-tailed commodity shocks they systematically
      UNDER-cover (claimed 80% but achieved ~70%).
    * Conformal intervals are calibrated empirically on held-out residuals, so
      they deliver the promised coverage *by construction* — a finite-sample,
      assumption-light guarantee.
    * Adaptive Conformal Inference (ACI) further adjusts alpha online to defend
      coverage when the residual distribution drifts (regime shifts), which is
      exactly when commodity forecasters fail.

Methods implemented:
    * Split conformal prediction — store sorted absolute residuals from a
      held-out calibration set; the (1 - alpha) empirical quantile sets the
      half-width. Optional per-horizon widening by sqrt(h) reflects accumulating
      multi-step uncertainty.
    * Adaptive Conformal Inference (ACI) — online update of the working alpha to
      track a target long-run coverage under distribution drift.

References:
    Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a Random World."
    Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction
        and Distribution-Free Uncertainty Quantification." arXiv:2107.07511.
    Gibbs & Candès (2021). "Adaptive Conformal Inference Under Distribution
        Shift." NeurIPS 2021. arXiv:2106.00170.

This module is ADDITIVE — nothing in the existing pipeline is modified. It is
wired opportunistically: hand it any point forecast and a calibration residual
set to obtain calibrated bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class ConformalForecaster:
    """
    Split-conformal wrapper producing distribution-free prediction intervals.

    Usage:
        c = ConformalForecaster()
        c.calibrate(holdout_residuals)            # 1-D array of (actual - pred)
        lower, upper = c.interval(point_forecasts, alpha=0.2)   # 80% bands
        cov = c.coverage(actuals, lower, upper)   # empirical check

    Attributes:
        sorted_abs_residuals: sorted absolute calibration residuals (ascending).
        n_calibration:        number of calibration points seen.
    """

    sorted_abs_residuals: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    n_calibration: int = 0

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self, residuals: np.ndarray) -> "ConformalForecaster":
        """
        Store sorted absolute residuals from a held-out calibration set.

        Args:
            residuals: 1-D array of calibration residuals (actual - forecast).
                       Sign is irrelevant — absolute values are used for
                       symmetric two-sided intervals.

        Returns:
            self (for chaining).
        """
        res = np.asarray(residuals, dtype=float).ravel()
        res = res[np.isfinite(res)]
        if res.size == 0:
            logger.warning("ConformalForecaster.calibrate: no finite residuals; "
                           "intervals will be degenerate (zero width).")
            self.sorted_abs_residuals = np.array([0.0])
            self.n_calibration = 0
            return self

        self.sorted_abs_residuals = np.sort(np.abs(res))
        self.n_calibration = int(res.size)
        logger.info(
            f"ConformalForecaster calibrated on n={self.n_calibration} residuals "
            f"(median |r|={np.median(self.sorted_abs_residuals):.4g})"
        )
        return self

    def quantile(self, alpha: float = 0.2) -> float:
        """
        Finite-sample conformal quantile of the absolute residuals.

        Uses the rank index ceil((n + 1)(1 - alpha)) / n (Angelopoulos & Bates),
        which provides the marginal-coverage guarantee. Clipped to a valid index.
        """
        if self.sorted_abs_residuals.size == 0:
            raise RuntimeError("Call calibrate() before requesting a quantile.")
        n = self.sorted_abs_residuals.size
        # Conformal level with finite-sample correction.
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        idx = int(np.ceil(level * n)) - 1
        idx = int(np.clip(idx, 0, n - 1))
        return float(self.sorted_abs_residuals[idx])

    # ── Interval construction ────────────────────────────────────────────────
    def interval(
        self,
        point_forecasts: np.ndarray,
        alpha: float = 0.2,
        horizon_widening: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (1 - alpha) prediction intervals around point forecasts.

        Args:
            point_forecasts: array of point forecasts, one per horizon step.
            alpha:           miscoverage rate (alpha=0.2 → 80% interval).
            horizon_widening: if True, widen the band for step h (1-indexed) by
                              sqrt(h) to reflect accumulating multi-step error.

        Returns:
            (lower, upper) numpy arrays the same shape as point_forecasts.
        """
        point = np.asarray(point_forecasts, dtype=float).ravel()
        q = self.quantile(alpha)

        if horizon_widening and point.size > 1:
            steps = np.arange(1, point.size + 1, dtype=float)
            half_width = q * np.sqrt(steps)
        else:
            half_width = np.full(point.shape, q, dtype=float)

        lower = point - half_width
        upper = point + half_width
        return lower, upper

    # ── Diagnostics ──────────────────────────────────────────────────────────
    @staticmethod
    def coverage(
        actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> float:
        """
        Empirical coverage: fraction of actuals falling within [lower, upper].
        """
        a = np.asarray(actuals, dtype=float).ravel()
        lo = np.asarray(lower, dtype=float).ravel()
        hi = np.asarray(upper, dtype=float).ravel()
        n = min(a.size, lo.size, hi.size)
        if n == 0:
            return float("nan")
        inside = (a[:n] >= lo[:n]) & (a[:n] <= hi[:n])
        return float(np.mean(inside))

    # ── Adaptive Conformal Inference (ACI) ───────────────────────────────────
    def adaptive_conformal(
        self,
        residuals_stream: np.ndarray,
        target_alpha: float = 0.2,
        gamma: float = 0.01,
    ) -> dict:
        """
        Adaptive Conformal Inference (Gibbs & Candès 2021).

        Runs an online loop over a stream of residuals, updating the working
        alpha_t to keep long-run coverage near (1 - target_alpha) even under
        distribution drift. Update rule:

            alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t)

        where err_t = 1 if the realised point fell OUTSIDE the interval at
        time t, else 0. A growing-window conformal quantile is recomputed each
        step from residuals seen so far.

        Args:
            residuals_stream: ordered residuals (actual - forecast) over time.
            target_alpha:     desired long-run miscoverage (0.2 → 80% target).
            gamma:            ACI learning rate (typical 0.005–0.05).

        Returns:
            dict with achieved_coverage, final_alpha, alpha_path (list),
            and per-step half-widths.
        """
        stream = np.asarray(residuals_stream, dtype=float).ravel()
        stream = stream[np.isfinite(stream)]
        if stream.size < 5:
            logger.warning("adaptive_conformal: need >=5 residuals; returning empty.")
            return {
                "achieved_coverage": float("nan"),
                "final_alpha": target_alpha,
                "alpha_path": [],
                "half_widths": [],
            }

        alpha_t = float(target_alpha)
        seen: list[float] = []
        covered: list[bool] = []
        alpha_path: list[float] = []
        half_widths: list[float] = []

        warmup = max(5, stream.size // 10)
        for t, r in enumerate(stream):
            if t < warmup:
                seen.append(abs(float(r)))
                continue
            arr = np.sort(np.asarray(seen, dtype=float))
            n = arr.size
            eff_alpha = float(np.clip(alpha_t, 1e-3, 0.999))
            level = np.ceil((n + 1) * (1.0 - eff_alpha)) / n
            idx = int(np.clip(int(np.ceil(np.clip(level, 0, 1) * n)) - 1, 0, n - 1))
            half_w = float(arr[idx])
            half_widths.append(half_w)

            inside = abs(float(r)) <= half_w
            covered.append(inside)
            err_t = 0.0 if inside else 1.0
            # ACI update toward target miscoverage.
            alpha_t = alpha_t + gamma * (target_alpha - err_t)
            alpha_t = float(np.clip(alpha_t, 1e-3, 0.999))
            alpha_path.append(alpha_t)

            seen.append(abs(float(r)))

        achieved = float(np.mean(covered)) if covered else float("nan")
        logger.info(
            f"ACI: target coverage={1 - target_alpha:.0%}, "
            f"achieved={achieved:.0%}, final_alpha={alpha_t:.3f}"
        )
        return {
            "achieved_coverage": achieved,
            "final_alpha": alpha_t,
            "alpha_path": alpha_path,
            "half_widths": half_widths,
        }


def quick_demo() -> None:
    """Fabricate residuals and confirm achieved coverage ~= target."""
    rng = np.random.default_rng(42)

    # Split-conformal demo: heavy-tailed residuals (Student-t) where Gaussian
    # CIs would under-cover; conformal still hits the target.
    cal_residuals = rng.standard_t(df=3, size=1000)
    c = ConformalForecaster().calibrate(cal_residuals)

    target = 0.8
    test_point = np.zeros(2000)
    test_actual = rng.standard_t(df=3, size=2000)  # same distribution
    lower, upper = c.interval(test_point, alpha=1 - target, horizon_widening=False)
    cov = c.coverage(test_actual, lower, upper)
    print(f"[split-conformal] target={target:.0%}  achieved={cov:.1%}  "
          f"half_width={float(upper[0]):.2f}")

    # ACI demo under drift: residual scale doubles halfway through.
    drift = np.concatenate([
        rng.normal(0, 1.0, 1000),
        rng.normal(0, 2.0, 1000),
    ])
    aci = c.adaptive_conformal(drift, target_alpha=0.2, gamma=0.02)
    print(f"[ACI under drift] target=80%  achieved={aci['achieved_coverage']:.1%}  "
          f"final_alpha={aci['final_alpha']:.3f}")


if __name__ == "__main__":
    quick_demo()
