"""
Gradient-Boosted Quantile Regression (Layer 3 — AI Forecast Layer)

Produces ASYMMETRIC, non-parametric prediction intervals for commodity prices
by directly regressing conditional quantiles, rather than assuming the
symmetric Gaussian errors baked into SARIMAX confidence bands.

Why this matters for GIC: commodity downside risk is fat-tailed and skewed —
metals (Lithium, Cobalt, Rhodium) spike violently on supply shocks but grind
down slowly. A symmetric ±1.96·sigma band systematically misprices tail risk.
Quantile regression learns P(y <= q | X) at each requested level, so the
resulting [q05, q95] band can be wider on the downside than the upside (or vice
versa) exactly where the data demands it.

Method
------
For a quantile tau, the model minimises the pinball (quantile) loss
    L_tau(y, f) = max( tau·(y-f), (tau-1)·(y-f) ).
Gradient boosting on this loss yields a consistent estimator of the conditional
tau-quantile (Koenker & Bassett 1978; Meinshausen 2006 "Quantile Regression
Forests"; Friedman 2001 "Greedy Function Approximation").

Backends (selected automatically, heavy imports are lazy):
  1. XGBoost >= 2.0 with `objective="reg:quantilederror"` and `quantile_alpha`
     — one model handles all quantiles jointly (XGBoost 2.x feature).
  2. XGBoost (older) — one booster per quantile via a custom pinball objective.
  3. sklearn GradientBoostingRegressor(loss="quantile", alpha=tau) — one model
     per quantile (robust fallback, always available with scikit-learn).

A monotonic post-sort guarantees q05 <= q50 <= q95 (avoids quantile crossing).

References:
  - Koenker, R. & Bassett, G. (1978). "Regression Quantiles." Econometrica.
  - Friedman, J. (2001). "Greedy Function Approximation: A Gradient Boosting
    Machine." Annals of Statistics.
  - Meinshausen, N. (2006). "Quantile Regression Forests." JMLR 7.

Degrades gracefully: if neither xgboost nor sklearn is importable, `fit` raises
a clear, documented RuntimeError — but the *module imports* with no heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class QuantileForecaster:
    """
    Gradient-boosted multi-quantile regressor.

    Examples
    --------
    >>> qf = QuantileForecaster(quantiles=(0.05, 0.5, 0.95))
    >>> qf.fit(X_train, y_train)              # doctest: +SKIP
    >>> bands = qf.predict(X_future)          # {0.05: ..., 0.5: ..., 0.95: ...}
    """

    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    random_state: int = 42

    # populated by fit()
    backend: str = field(default="", init=False)
    _models: dict[float, Any] = field(default_factory=dict, init=False)
    _joint_model: Any = field(default=None, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)

    # ──────────────────────────────────────────────────────────────────────
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        quantiles: tuple[float, ...] | None = None,
    ) -> "QuantileForecaster":
        """Fit one model per quantile (or a joint model where supported)."""
        if quantiles is not None:
            self.quantiles = tuple(sorted(quantiles))
        else:
            self.quantiles = tuple(sorted(self.quantiles))

        Xv, self.feature_names = self._as_matrix(X)
        yv = np.asarray(y, dtype=float).ravel()
        if len(Xv) != len(yv):
            raise ValueError(f"X/y length mismatch: {len(Xv)} vs {len(yv)}")

        # ── Backend 1/2: XGBoost ──────────────────────────────────────────
        try:
            import xgboost as xgb  # noqa: F401
            self._fit_xgboost(Xv, yv)
            logger.info(f"QuantileForecaster fitted via {self.backend} "
                        f"on q={self.quantiles}")
            return self
        except ImportError:
            logger.warning("xgboost unavailable — falling back to sklearn GBR.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"xgboost quantile path failed ({exc}); "
                           "falling back to sklearn GBR.")

        # ── Backend 3: sklearn ────────────────────────────────────────────
        try:
            self._fit_sklearn(Xv, yv)
            logger.info(f"QuantileForecaster fitted via {self.backend} "
                        f"on q={self.quantiles}")
            return self
        except ImportError as exc:
            raise RuntimeError(
                "QuantileForecaster requires either xgboost or scikit-learn; "
                "neither is installed."
            ) from exc

    # ──────────────────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[float, np.ndarray]:
        """
        Return {quantile: predictions}. Quantiles are sorted per-row to remove
        any crossing (q_lower <= q_mid <= q_upper).
        """
        if not self._models and self._joint_model is None:
            raise RuntimeError("QuantileForecaster.predict called before fit().")
        Xv, _ = self._as_matrix(X)

        preds: dict[float, np.ndarray] = {}
        if self._joint_model is not None:
            import xgboost as xgb
            dm = xgb.DMatrix(Xv)
            raw = np.asarray(self._joint_model.predict(dm))
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            for j, q in enumerate(self.quantiles):
                preds[q] = raw[:, j]
        else:
            for q, model in self._models.items():
                preds[q] = np.asarray(model.predict(Xv)).ravel()

        return self._enforce_monotone(preds)

    # ──────────────────────────────────────────────────────────────────────
    # Backends
    # ──────────────────────────────────────────────────────────────────────
    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb

        # Try XGBoost 2.x joint multi-quantile objective first.
        ver = tuple(int(p) for p in xgb.__version__.split(".")[:2])
        if ver >= (2, 0):
            try:
                model = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=np.array(self.quantiles),
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    tree_method="hist",
                )
                model.fit(X, y)
                # sklearn wrapper returns (n, n_quantiles) for multi-alpha.
                self._models = {}
                self._joint_model = None
                self._sk_joint = model
                self.backend = "xgboost-2.x-joint"
                # Store as per-quantile callables for a uniform predict path.
                self._wrap_sk_joint(model)
                return
            except Exception as exc:
                logger.warning(f"XGBoost 2.x joint objective failed ({exc}); "
                               "using per-quantile pinball boosters.")

        # Older XGBoost: one booster per quantile via custom pinball gradient.
        self.backend = "xgboost-pinball-per-quantile"
        for q in self.quantiles:
            self._models[q] = self._fit_xgb_pinball(xgb, X, y, q)

    def _wrap_sk_joint(self, model: Any) -> None:
        """Adapt a multi-alpha sklearn-API XGB model to per-quantile predict."""
        quantiles = self.quantiles

        class _Col:
            def __init__(self, mdl, j):
                self.mdl, self.j = mdl, j

            def predict(self, X):
                out = np.asarray(self.mdl.predict(X))
                if out.ndim == 1:
                    return out
                return out[:, self.j]

        self._models = {q: _Col(model, j) for j, q in enumerate(quantiles)}

    def _fit_xgb_pinball(self, xgb, X: np.ndarray, y: np.ndarray, q: float) -> Any:
        """Train one booster with a custom pinball (quantile) objective."""
        dtrain = xgb.DMatrix(X, label=y)

        def _obj(preds: np.ndarray, dmat) -> tuple[np.ndarray, np.ndarray]:
            labels = dmat.get_label()
            err = labels - preds
            # Smoothed pinball gradient/hessian.
            grad = np.where(err >= 0, -q, -(q - 1.0))
            hess = np.full_like(preds, 1.0)
            return grad, hess

        params = {
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "tree_method": "hist",
            "seed": self.random_state,
        }
        booster = xgb.train(params, dtrain, num_boost_round=self.n_estimators,
                            obj=_obj)

        class _Booster:
            def __init__(self, b):
                self.b = b

            def predict(self, X):
                return self.b.predict(xgb.DMatrix(X))

        return _Booster(booster)

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.backend = "sklearn-gbr-quantile"
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            model.fit(X, y)
            self._models[q] = model

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────
    def _enforce_monotone(self, preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
        """Sort predictions across quantiles row-wise to prevent crossing."""
        qs = sorted(preds.keys())
        mat = np.column_stack([preds[q] for q in qs])
        mat = np.sort(mat, axis=1)
        return {q: mat[:, j] for j, q in enumerate(qs)}

    @staticmethod
    def _as_matrix(X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float), list(X.columns)
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, [f"f{i}" for i in range(arr.shape[1])]


def quick_demo() -> None:
    """Fit on heteroscedastic, skewed data and show asymmetric bands."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("quantile quick_demo skipped: scikit-learn not installed.")
        return

    rng = np.random.default_rng(0)
    n = 600
    x = np.linspace(0, 4, n)
    # Skewed, heteroscedastic noise (downside-heavy).
    noise = rng.standard_gamma(2.0, n) - 2.0 + 0.5 * x * rng.standard_normal(n)
    y = 10.0 + 2.0 * x + noise
    X = x.reshape(-1, 1)

    qf = QuantileForecaster(quantiles=(0.05, 0.5, 0.95), n_estimators=200)
    qf.fit(X, y)
    bands = qf.predict(X)

    lo, mid, hi = bands[0.05], bands[0.5], bands[0.95]
    # Empirical coverage of the [5%,95%] band should be ~90%.
    cov = float(np.mean((y >= lo) & (y <= hi)))
    up = float(np.mean(hi - mid))
    down = float(np.mean(mid - lo))
    print(f"[quantile] backend={qf.backend}  band coverage={cov:.1%} "
          f"(target 90%)")
    print(f"[quantile] mean upper half-width={up:.2f}  "
          f"lower half-width={down:.2f}  asymmetry ratio={down / (up + 1e-9):.2f}")
    print("quantile_forecast quick_demo OK")


if __name__ == "__main__":
    quick_demo()
