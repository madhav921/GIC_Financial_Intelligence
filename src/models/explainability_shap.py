"""
SHAP Feature Attribution (Layer 3 — Explainability)

Per-forecast driver attribution for the tree-based commodity models (XGBoost).
Turns an opaque gradient-boosted forecast into a ranked, signed set of drivers
("Manufacturing PMI contributed +2.3% to the forecast") that feeds directly
into the LLM narrative layer.

Primary method — SHAP (SHapley Additive exPlanations):
    SHAP values are the game-theoretic Shapley values of a model's prediction,
    the unique attribution that is locally accurate (sum of attributions +
    base value = prediction), consistent, and missingness-respecting. For tree
    ensembles, `shap.TreeExplainer` computes them exactly in polynomial time.

Graceful degradation:
    If `shap` is not installed, this module falls back to:
        1. XGBoost gain-based feature importance (global), and
        2. a single-pass permutation-importance approximation around the row
           being explained (local sensitivity),
    combined into a normalized contribution dict. The public API is identical,
    so callers never branch on whether SHAP is present.

References:
    Lundberg & Lee (2017). "A Unified Approach to Interpreting Model
        Predictions." NeurIPS 2017. arXiv:1705.07874.
    Lundberg et al. (2020). "From local explanations to global understanding
        with explainable AI for trees." Nature Machine Intelligence.

This module is ADDITIVE — heavy/optional deps (shap, xgboost) are imported
lazily inside methods so the module always imports cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Driver:
    """A single signed feature attribution for one forecast."""
    feature: str
    value: float
    contribution: float          # signed, in model output units
    direction: str               # "up" | "down"


@dataclass
class ShapExplainer:
    """
    Driver attribution for tree models with a SHAP-first, importance-fallback path.

    Usage:
        ex = ShapExplainer()
        result = ex.explain(xgb_model, X_row)          # dict
        drivers = ex.narrative_drivers(xgb_model, X_row, base_value=last_price)
    """

    method_used: str = field(default="unknown")

    # ── Core attribution ─────────────────────────────────────────────────────
    def explain(self, model: Any, X: pd.DataFrame | np.ndarray) -> dict:
        """
        Attribute a forecast (or batch) to its features.

        Args:
            model: a fitted tree model (XGBoost sklearn wrapper or Booster).
            X:     feature matrix (DataFrame preferred for names). A single row
                   or a batch; batch attributions are mean-|value| aggregated
                   for the global view plus the first row for the local view.

        Returns:
            dict with:
                method            : "shap" | "importance_fallback"
                contributions     : {feature: normalized_contribution in [0,1]}
                signed            : list[Driver] sorted by |contribution| desc
                base_value        : float SHAP base value (NaN in fallback)
        """
        feature_names = self._feature_names(model, X)
        X_arr = self._as_array(X)

        shap_out = self._try_shap(model, X_arr, feature_names)
        if shap_out is not None:
            self.method_used = "shap"
            return shap_out

        logger.info("SHAP unavailable — using gain + permutation importance fallback.")
        self.method_used = "importance_fallback"
        return self._importance_fallback(model, X_arr, feature_names)

    def narrative_drivers(
        self,
        model: Any,
        X: pd.DataFrame | np.ndarray,
        base_value: float | None = None,
        top_k: int = 5,
        as_pct: bool = True,
    ) -> list[dict]:
        """
        Produce a compact, LLM-ready list of the top drivers of a forecast.

        Each item: {feature, value, contribution, direction, text}, e.g.
            "Manufacturing PMI contributed +2.3% to the forecast".

        Args:
            base_value: reference level for percentage framing (e.g. last price).
                        If None, percentages are relative to the summed
                        absolute contribution.
            top_k:      number of drivers to surface.
            as_pct:     express contribution as a percentage in the text.
        """
        result = self.explain(model, X)
        signed: list[Driver] = result["signed"][:top_k]

        if base_value is not None and abs(base_value) > 1e-9:
            denom = abs(base_value)
        else:
            denom = sum(abs(d.contribution) for d in signed) or 1.0

        narratives: list[dict] = []
        for d in signed:
            pct = 100.0 * d.contribution / denom
            sign = "+" if d.contribution >= 0 else "-"
            pretty = d.feature.replace("macro_", "").replace("_", " ").strip().title()
            if as_pct:
                text = f"{pretty} contributed {sign}{abs(pct):.1f}% to the forecast"
            else:
                text = (f"{pretty} contributed {sign}{abs(d.contribution):.3g} "
                        f"to the forecast")
            narratives.append({
                "feature": d.feature,
                "value": round(float(d.value), 4),
                "contribution": round(float(d.contribution), 6),
                "direction": d.direction,
                "text": text,
            })
        return narratives

    # ── SHAP path ────────────────────────────────────────────────────────────
    def _try_shap(
        self, model: Any, X_arr: np.ndarray, feature_names: list[str]
    ) -> dict | None:
        try:
            import shap  # type: ignore
        except Exception as exc:  # ImportError or backend issues
            logger.debug(f"shap import failed: {exc}")
            return None

        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_arr)
            sv = np.asarray(sv, dtype=float)
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)

            local = sv[0]                       # signed attribution for first row
            global_abs = np.mean(np.abs(sv), axis=0)

            base_value = float(np.atleast_1d(explainer.expected_value)[0])
            contributions = self._normalize(dict(zip(feature_names, global_abs)))
            signed = self._signed_drivers(feature_names, local, X_arr[0])
            return {
                "method": "shap",
                "contributions": contributions,
                "signed": signed,
                "base_value": base_value,
            }
        except Exception as exc:
            logger.warning(f"SHAP TreeExplainer failed ({exc}); falling back.")
            return None

    # ── Fallback path ────────────────────────────────────────────────────────
    def _importance_fallback(
        self, model: Any, X_arr: np.ndarray, feature_names: list[str]
    ) -> dict:
        n_features = len(feature_names)

        # 1) Global gain-based importance.
        gain = self._gain_importance(model, n_features)

        # 2) Local permutation sensitivity around the explained row.
        row = X_arr[0].astype(float).copy()

        def _predict(arr2d: np.ndarray) -> float:
            """Name-agnostic predict: bypass the sklearn wrapper's feature-name
            validation (which raises when a model fit on a named DataFrame is
            scored with a bare numpy array, silently zeroing the slope)."""
            try:
                booster = model.get_booster()
                import xgboost as xgb  # local; only reached for XGB models
                dm = xgb.DMatrix(arr2d)
                # The booster validates feature names; supply them when the
                # model was trained on a named DataFrame so prediction succeeds.
                bnames = getattr(booster, "feature_names", None)
                if bnames is not None:
                    dm.feature_names = list(bnames)
                return float(np.asarray(booster.predict(dm)).ravel()[0])
            except Exception:
                try:
                    out = model.predict(arr2d, validate_features=False)
                except TypeError:
                    out = model.predict(arr2d)
                return float(np.asarray(out).ravel()[0])

        try:
            base_pred = _predict(row.reshape(1, -1))
        except Exception:
            base_pred = 0.0

        local_signed = np.zeros(n_features, dtype=float)
        X_f = X_arr.astype(float)
        multi_row = X_f.shape[0] > 1
        col_means = np.nanmean(X_f, axis=0) if multi_row else row
        col_std = np.nanstd(X_f, axis=0) if multi_row else np.abs(row) + 1.0
        col_std = np.where(col_std > 1e-9, col_std, 1.0)
        for j in range(n_features):
            # Symmetric finite-difference local model slope w.r.t. feature j.
            step = 0.5 * col_std[j]
            up_row, dn_row = row.copy(), row.copy()
            up_row[j] += step
            dn_row[j] -= step
            try:
                p_up = _predict(up_row.reshape(1, -1))
                p_dn = _predict(dn_row.reshape(1, -1))
            except Exception:
                p_up = p_dn = base_pred
            slope = (p_up - p_dn) / (2.0 * step)
            if multi_row:
                # Linearised contribution: slope x deviation from the population
                # mean (this row's feature value relative to the batch).
                local_signed[j] = slope * (row[j] - col_means[j])
            else:
                # Single-row input has no population reference, so the deviation
                # is undefined; use the signed local sensitivity (slope x step)
                # — the model's directional response to a 0.5-std move.
                local_signed[j] = slope * step

        # Blend gain (global) with local sensitivity magnitude for ranking.
        abs_local = np.abs(local_signed)
        blend = 0.5 * self._unit(gain) + 0.5 * self._unit(abs_local)
        contributions = self._normalize(dict(zip(feature_names, blend)))
        signed = self._signed_drivers(feature_names, local_signed, row)
        return {
            "method": "importance_fallback",
            "contributions": contributions,
            "signed": signed,
            "base_value": float("nan"),
        }

    @staticmethod
    def _gain_importance(model: Any, n_features: int) -> np.ndarray:
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
            if imp.size == n_features:
                return imp
        except Exception:
            pass
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            vec = np.zeros(n_features, dtype=float)
            for k, v in score.items():
                # XGBoost keys like "f12".
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < n_features:
                        vec[idx] = float(v)
            return vec
        except Exception:
            return np.ones(n_features, dtype=float)

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _signed_drivers(
        feature_names: list[str], signed_vals: np.ndarray, row_values: np.ndarray
    ) -> list[Driver]:
        drivers = [
            Driver(
                feature=feature_names[i],
                value=float(row_values[i]) if i < len(row_values) else float("nan"),
                contribution=float(signed_vals[i]),
                direction=(
                    "flat" if abs(signed_vals[i]) < 1e-9
                    else "up" if signed_vals[i] > 0 else "down"
                ),
            )
            for i in range(len(feature_names))
        ]
        drivers.sort(key=lambda d: abs(d.contribution), reverse=True)
        return drivers

    @staticmethod
    def _normalize(d: dict[str, float]) -> dict[str, float]:
        total = sum(abs(v) for v in d.values())
        if total <= 1e-12:
            return {k: 0.0 for k in d}
        return {k: abs(v) / total for k, v in d.items()}

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.abs(np.asarray(v, dtype=float))
        s = v.sum()
        return v / s if s > 1e-12 else v

    @staticmethod
    def _as_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(float)

    @staticmethod
    def _feature_names(model: Any, X: pd.DataFrame | np.ndarray) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        n = X.shape[1] if np.asarray(X).ndim == 2 else np.asarray(X).reshape(1, -1).shape[1]
        names = getattr(model, "feature_names_in_", None)
        if names is not None and len(names) == n:
            return list(names)
        return [f"f{i}" for i in range(n)]


def quick_demo() -> None:
    """Fit a tiny synthetic XGBoost and explain one row (guards missing deps)."""
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        print(f"[shap-demo] xgboost unavailable ({exc}) — skipping fit demo.")
        return

    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "manufacturing_pmi": rng.normal(52, 3, 300),
        "usd_index": rng.normal(100, 5, 300),
        "noise": rng.normal(0, 1, 300),
    })
    # Target driven mostly by PMI (positive) and USD (negative).
    y = 3.0 * X["manufacturing_pmi"] - 1.5 * X["usd_index"] + rng.normal(0, 2, 300)

    model = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.1,
                         random_state=0, tree_method="hist")
    # Fit on the DataFrame so XGBoost records feature names natively
    # (feature_names_in_ is a read-only property and cannot be assigned).
    model.fit(X, y.values)

    ex = ShapExplainer()
    # Frame contributions relative to the typical target level, not a single
    # (possibly near-zero) observation, so percentages are interpretable.
    base = float(np.mean(np.abs(y)))
    drivers = ex.narrative_drivers(model, X.iloc[[0]], base_value=base)
    print(f"[shap-demo] attribution method = {ex.method_used}")
    for d in drivers[:3]:
        print("   ", d["text"])


if __name__ == "__main__":
    quick_demo()
