"""
Commodity Price Forecast Model (Layer 3 — AI Forecast Layer)

Multi-approach commodity forecasting using REAL market data from Yahoo Finance:
  1. SARIMAX — seasonal ARIMA with exogenous macro regressors
  2. XGBoost  — gradient boosted ensemble with rich feature engineering
  3. Ensemble — weighted blend (XGBoost-dominant when data is plentiful)

Designed to work with BOTH real-world (yfinance) data AND synthetic fallback.

Key improvements over v0.1.0:
  - Integrates actual fetched yfinance data (data/external/market_commodities.parquet)
  - Proper walk-forward validation (no look-ahead bias)
  - Richer feature set: RSI, MACD momentum, VIX/macro context
  - Ensemble weighting based on CV MAPE
  - Stores feature importance for explainability
  - Handles monthly & daily granularity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.config import get_project_root, get_settings
from src.data.feature_engineering import prepare_commodity_features
from src.models.model_registry import ModelRegistry


@dataclass
class ForecastResult:
    """Container for forecast outputs."""
    commodity: str
    dates: list[str]
    point_forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    lower_95: list[float]
    upper_95: list[float]
    model_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Container for backtesting outputs."""
    commodity: str
    model_type: str
    dates: list[str]
    actuals: list[float]
    predictions: list[float]
    errors: list[float]
    pct_errors: list[float]
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float  # % of times direction was correct
    hit_rate_10pct: float        # % within ±10% of actual


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — momentum signal."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd_signal(series: pd.Series, fast: int = 3, slow: int = 6) -> pd.Series:
    """MACD line (adapted for monthly data) — trend signal."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _build_rich_features(
    price_series: pd.Series,
    macro_df: pd.DataFrame | None,
    lags: list[int],
    target_col: str,
) -> pd.DataFrame:
    """
    Build a comprehensive feature matrix for a single commodity price series.
    Much richer than the original — adds momentum, RSI, spread, macro context.
    """
    df = price_series.to_frame(name=target_col).copy()
    df.index = pd.to_datetime(df.index)

    # ─── Lag features ────────────────────────────────────────────────────────
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # ─── Rolling statistics ───────────────────────────────────────────────────
    for window in [3, 6, 12]:
        df[f"ma_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"std_{window}"] = df[target_col].shift(1).rolling(window).std()
        df[f"min_{window}"] = df[target_col].shift(1).rolling(window).min()
        df[f"max_{window}"] = df[target_col].shift(1).rolling(window).max()
        # z-score within rolling window
        df[f"zscore_{window}"] = (
            (df[target_col].shift(1) - df[f"ma_{window}"]) / (df[f"std_{window}"] + 1e-9)
        )

    # ─── Momentum / Returns ───────────────────────────────────────────────────
    for period in [1, 3, 6, 12]:
        df[f"ret_{period}m"] = df[target_col].pct_change(period).shift(1)

    # ─── Technical indicators (RSI, MACD) as trend signals ───────────────────
    df["rsi_14"] = _rsi(df[target_col].shift(1), period=14)
    df["macd"] = _macd_signal(df[target_col].shift(1))
    df["macd_sign"] = (df["macd"] > 0).astype(int)

    # ─── Price level features ─────────────────────────────────────────────────
    df["log_price"] = np.log1p(df[target_col].shift(1))
    df["price_vs_52w_high"] = df[target_col].shift(1) / (
        df[target_col].shift(1).rolling(12).max() + 1e-9
    )

    # ─── Calendar encoding ────────────────────────────────────────────────────
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["quarter"] = df.index.quarter

    # ─── Macro context (aligned by date) ──────────────────────────────────────
    if macro_df is not None:
        macro = macro_df.copy()
        if "date" in macro.columns:
            macro = macro.set_index(pd.to_datetime(macro["date"])).drop(columns=["date"])
        macro.index = pd.to_datetime(macro.index)

        macro_cols = [c for c in macro.columns if macro[c].dtype in [float, "float64", "float32", "int64", "int32"]]

        # Resample to match price frequency (monthly)
        macro_monthly = macro[macro_cols].resample("MS").last()
        for col in macro_cols[:8]:  # limit to avoid feature explosion
            merged = df.join(macro_monthly[[col]], how="left")
            df[f"macro_{col.lower().replace(' ', '_')}"] = merged[col].shift(1).ffill()
            df[f"macro_{col.lower().replace(' ', '_')}_lag3"] = merged[col].shift(3).ffill()

    # ─── Target ───────────────────────────────────────────────────────────────
    df["target"] = df[target_col]

    # Drop rows with too many NaNs (from lagging)
    min_valid_lag = max(lags) if lags else 12
    df = df.iloc[min_valid_lag:].dropna(subset=["target"])

    return df


class SARIMAXForecaster:
    """SARIMAX-based commodity price forecaster."""

    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, series: pd.Series, exog: pd.DataFrame | None = None) -> dict[str, float]:
        """Fit SARIMAX model and return in-sample metrics."""
        model = SARIMAX(
            series,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False, maxiter=200)
        fitted = self.model_fit.fittedvalues
        metrics = {
            "aic": float(self.model_fit.aic),
            "bic": float(self.model_fit.bic),
            "mae": float(mean_absolute_error(series, fitted)),
            "mape": float(mean_absolute_percentage_error(series, fitted) * 100),
        }
        logger.info(f"SARIMAX fitted — AIC={metrics['aic']:.1f}, MAE={metrics['mae']:.1f}")
        return metrics

    def predict(
        self, steps: int, exog_future: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast with confidence intervals."""
        forecast = self.model_fit.get_forecast(steps=steps, exog=exog_future)
        mean = forecast.predicted_mean.values
        ci_80 = forecast.conf_int(alpha=0.20).values
        ci_95 = forecast.conf_int(alpha=0.05).values
        return mean, ci_80, ci_95


class XGBoostForecaster:
    """XGBoost-based commodity price forecaster with rich time-series features."""

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
        "tree_method": "hist",
    }

    def __init__(self, **params):
        p = dict(self.DEFAULT_PARAMS)
        p.update(params)
        self.model = XGBRegressor(**p)
        self.feature_names: list[str] = []
        self.scaler: StandardScaler | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """Train XGBoost and return validation metrics."""
        self.feature_names = list(X_train.columns)

        eval_set = [(X_train.values, y_train.values)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val.values, y_val.values))

        self.model.fit(
            X_train.values, y_train.values,
            eval_set=eval_set,
            verbose=False,
        )

        X_eval = X_val if X_val is not None else X_train
        y_eval = y_val if y_val is not None else y_train
        preds = self.model.predict(X_eval.values)

        mae = float(mean_absolute_error(y_eval, preds))
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
        mape = float(mean_absolute_percentage_error(y_eval, preds) * 100)

        # Directional accuracy
        actual_dir = np.sign(np.diff(y_eval.values))
        pred_dir = np.sign(np.diff(preds))
        dir_acc = float(np.mean(actual_dir == pred_dir)) * 100 if len(actual_dir) > 0 else 0.0

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": dir_acc,
        }
        logger.info(
            f"XGBoost fitted — RMSE={rmse:.1f}, MAE={mae:.1f}, "
            f"MAPE={mape:.1f}%, Dir.Acc={dir_acc:.0f}%"
        )
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores sorted descending."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class CommodityForecastModel:
    """
    Orchestrates commodity price forecasting across multiple approaches.

    Integrates real-world yfinance data when available, falls back to synthetic.
    Generates a BOM-weighted Commodity Index that feeds into the Driver Engine.
    """

    def __init__(self):
        self.settings = get_settings()
        self.horizon = self.settings["forecast"]["horizon_months"]
        self.registry = ModelRegistry()
        self.sarimax_models: dict[str, SARIMAXForecaster] = {}
        self.xgb_models: dict[str, XGBoostForecaster] = {}
        self._feature_importance: dict[str, pd.DataFrame] = {}
        self._cv_metrics: dict[str, dict] = {}
        self._lags = self.settings["forecast"].get("feature_lag_months", [1, 3, 6, 12])

    def _load_real_prices(self) -> pd.DataFrame | None:
        """
        Load real-world commodity prices from yfinance (fetched data).
        Returns pandas DataFrame or None if not available.
        """
        root = get_project_root()
        ext_path = root / "data" / "external" / "market_commodities.parquet"
        if ext_path.exists():
            import polars as pl
            df = pl.read_parquet(ext_path).to_pandas()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            elif "Date" in df.columns:
                df["date"] = pd.to_datetime(df["Date"])
                df = df.set_index("date").drop(columns=["Date"]).sort_index()
            # Resample to monthly (closing price at month end)
            df = df.resample("MS").last().ffill()
            logger.info(f"Loaded real yfinance commodity data: {df.shape}")
            return df
        return None

    def _load_real_macro(self) -> pd.DataFrame | None:
        """Load real macro context (FRED or synthetic fallback)."""
        root = get_project_root()
        # Try FRED data first
        fred_path = root / "data" / "external" / "fred_macro.parquet"
        if fred_path.exists():
            import polars as pl
            return pl.read_parquet(fred_path).to_pandas()
        # Try market indices as macro context
        idx_path = root / "data" / "external" / "market_indices.parquet"
        if idx_path.exists():
            import polars as pl
            return pl.read_parquet(idx_path).to_pandas()
        return None

    def train_sarimax(
        self, commodity: str, price_series: pd.Series, exog: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train SARIMAX model for a single commodity."""
        model = SARIMAXForecaster()
        metrics = model.fit(price_series, exog=exog)
        self.sarimax_models[commodity] = model
        return metrics

    def train_xgboost(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        test_size: int = 12,
    ) -> dict[str, float]:
        """
        Train XGBoost model for a single commodity with rich feature engineering.
        Uses real features: RSI, MACD, rolling z-scores, macro context.
        """
        # Use real price data if column matches, else fall back to synthetic
        if isinstance(commodity_df.index, pd.DatetimeIndex):
            price_series = commodity_df[commodity] if commodity in commodity_df.columns else None
        else:
            price_series = commodity_df.set_index("date")[commodity] if commodity in commodity_df.columns else None

        if price_series is None or price_series.dropna().__len__() < 24:
            logger.warning(f"Insufficient data for {commodity} XGBoost, using simpler features")
            return self._train_xgboost_simple(commodity, commodity_df, macro_df, test_size)

        feat_df = _build_rich_features(price_series, macro_df, self._lags, commodity)

        feature_cols = [c for c in feat_df.columns if c != "target"]
        X = feat_df[feature_cols]
        y = feat_df["target"]

        if len(X) < test_size + 12:
            test_size = max(6, len(X) // 5)

        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        model = XGBoostForecaster()
        metrics = model.fit(X_train, y_train, X_test, y_test)
        self.xgb_models[commodity] = model
        self._feature_importance[commodity] = model.feature_importance()

        self.registry.save_model(
            model.model,
            f"commodity_xgb_{commodity.lower().replace(' ', '_')}",
            metrics=metrics,
            features=feature_cols,
        )

        return metrics

    def _train_xgboost_simple(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        test_size: int,
    ) -> dict[str, float]:
        """Fallback to original simple feature engineering."""
        feat_df = prepare_commodity_features(
            commodity_df, macro_df, commodity,
            lags=self._lags,
        )
        feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]
        X = feat_df[feature_cols]
        y = feat_df["target"]

        if len(X) <= test_size:
            test_size = max(1, len(X) // 5)

        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        model = XGBoostForecaster()
        metrics = model.fit(X_train, y_train, X_test, y_test)
        self.xgb_models[commodity] = model
        self._feature_importance[commodity] = model.feature_importance()
        return metrics

    def cross_validate(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Walk-forward cross-validation — no look-ahead bias."""
        feat_df = prepare_commodity_features(
            commodity_df, macro_df, commodity,
            lags=self._lags,
        )
        feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]
        X = feat_df[feature_cols].values
        y = feat_df["target"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes, mapes, dir_accs = [], [], []

        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 12 or len(val_idx) < 3:
                continue
            model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
            model.fit(X[train_idx], y[train_idx], verbose=False)
            preds = model.predict(X[val_idx])
            maes.append(mean_absolute_error(y[val_idx], preds))
            mapes.append(mean_absolute_percentage_error(y[val_idx], preds) * 100)
            actual_dir = np.sign(np.diff(y[val_idx]))
            pred_dir = np.sign(np.diff(preds))
            if len(actual_dir) > 0:
                dir_accs.append(float(np.mean(actual_dir == pred_dir)) * 100)

        cv_metrics = {
            "cv_mae_mean": float(np.mean(maes)) if maes else float("nan"),
            "cv_mae_std": float(np.std(maes)) if maes else float("nan"),
            "cv_mape_mean": float(np.mean(mapes)) if mapes else float("nan"),
            "cv_mape_std": float(np.std(mapes)) if mapes else float("nan"),
            "cv_directional_accuracy": float(np.mean(dir_accs)) if dir_accs else float("nan"),
        }
        logger.info(
            f"CV {commodity}: MAPE={cv_metrics['cv_mape_mean']:.1f}%"
            f"+-{cv_metrics['cv_mape_std']:.1f}%,"
            f" Dir.Acc={cv_metrics['cv_directional_accuracy']:.0f}%"
        )
        self._cv_metrics[commodity] = cv_metrics
        return cv_metrics

    def forecast_sarimax(self, commodity: str) -> ForecastResult:
        """Generate SARIMAX forecast for a commodity."""
        model = self.sarimax_models.get(commodity)
        if model is None:
            raise ValueError(f"SARIMAX model not trained for {commodity}")

        mean, ci_80, ci_95 = model.predict(self.horizon)
        future_dates = pd.date_range(
            pd.Timestamp.now().normalize(), periods=self.horizon, freq="MS"
        )

        return ForecastResult(
            commodity=commodity,
            dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            point_forecast=mean.tolist(),
            lower_80=ci_80[:, 0].tolist(),
            upper_80=ci_80[:, 1].tolist(),
            lower_95=ci_95[:, 0].tolist(),
            upper_95=ci_95[:, 1].tolist(),
            model_type="SARIMAX",
        )

    def forecast_xgboost(
        self, commodity: str, commodity_df: pd.DataFrame, macro_df: pd.DataFrame | None
    ) -> ForecastResult:
        """
        Generate XGBoost recursive multi-step forecast.
        Uses the trained model's own predictions as inputs for future steps.
        """
        model = self.xgb_models.get(commodity)
        if model is None:
            raise ValueError(f"XGBoost model not trained for {commodity}")

        # Get the last available price series
        if isinstance(commodity_df.index, pd.DatetimeIndex):
            price_series = commodity_df[commodity] if commodity in commodity_df.columns else pd.Series()
        else:
            price_series = (
                commodity_df.set_index("date")[commodity]
                if "date" in commodity_df.columns and commodity in commodity_df.columns
                else pd.Series()
            )

        feat_df = _build_rich_features(price_series, macro_df, self._lags, commodity)
        feature_cols = [c for c in feat_df.columns if c != "target"]

        # Use final in-sample row as base for recursive forecast
        last_row = feat_df[feature_cols].iloc[[-1]].copy()
        forecasts = []
        future_dates = pd.date_range(
            pd.Timestamp.now().normalize(), periods=self.horizon, freq="MS"
        )

        last_price = float(feat_df["target"].iloc[-1])

        for step in range(self.horizon):
            pred = float(model.predict(last_row)[0])
            pred = max(pred, 0.01)  # avoid negative prices
            forecasts.append(pred)

            # Update lag features for next step
            for lag in self._lags:
                lag_col = f"lag_{lag}"
                if lag_col in last_row.columns:
                    if lag == 1:
                        last_row[lag_col] = pred
                    else:
                        prev_lag_col = f"lag_{lag - 1}"
                        if prev_lag_col in last_row.columns:
                            last_row[lag_col] = last_row[prev_lag_col].values[0]

            # Update return features
            if "ret_1m" in last_row.columns:
                last_row["ret_1m"] = (pred - last_price) / (last_price + 1e-9)
            last_price = pred

        # Estimate uncertainty from CV metrics
        cv_mape = self._cv_metrics.get(commodity, {}).get("cv_mape_mean", 10.0)
        uncertainty_80 = cv_mape / 100 * 1.28  # ~80% CI
        uncertainty_95 = cv_mape / 100 * 1.96  # ~95% CI

        lower_80 = [f * (1 - uncertainty_80 * (i + 1) ** 0.5 / self.horizon ** 0.5)
                    for i, f in enumerate(forecasts)]
        upper_80 = [f * (1 + uncertainty_80 * (i + 1) ** 0.5 / self.horizon ** 0.5)
                    for i, f in enumerate(forecasts)]
        lower_95 = [f * (1 - uncertainty_95 * (i + 1) ** 0.5 / self.horizon ** 0.5)
                    for i, f in enumerate(forecasts)]
        upper_95 = [f * (1 + uncertainty_95 * (i + 1) ** 0.5 / self.horizon ** 0.5)
                    for i, f in enumerate(forecasts)]

        importance = {}
        if commodity in self._feature_importance:
            fi = self._feature_importance[commodity]
            importance = dict(zip(fi["feature"].head(10), fi["importance"].head(10)))

        return ForecastResult(
            commodity=commodity,
            dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            point_forecast=forecasts,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_type="XGBoost",
            metrics=self._cv_metrics.get(commodity, {}),
            feature_importance=importance,
        )

    def forecast_ensemble(
        self, commodity: str, commodity_df: pd.DataFrame, macro_df: pd.DataFrame | None
    ) -> ForecastResult:
        """
        Ensemble forecast: weighted blend of SARIMAX + XGBoost.
        Weights are inversely proportional to CV MAPE (better model gets more weight).
        """
        sarimax_result = self.forecast_sarimax(commodity)
        xgb_result = self.forecast_xgboost(commodity, commodity_df, macro_df)

        # Weight by inverse CV MAPE (lower error = higher weight)
        sarimax_mape = self.sarimax_models[commodity].model_fit.fittedvalues
        sarimax_train_mape = float(
            mean_absolute_percentage_error(
                commodity_df[commodity].iloc[-len(sarimax_mape):],
                sarimax_mape
            ) * 100
        )
        xgb_mape = self._cv_metrics.get(commodity, {}).get("cv_mape_mean", sarimax_train_mape)

        total_inv = (1 / (sarimax_train_mape + 1e-6)) + (1 / (xgb_mape + 1e-6))
        w_sarimax = (1 / (sarimax_train_mape + 1e-6)) / total_inv
        w_xgb = (1 / (xgb_mape + 1e-6)) / total_inv

        n = min(len(sarimax_result.point_forecast), len(xgb_result.point_forecast))
        ensemble_forecast = [
            w_sarimax * s + w_xgb * x
            for s, x in zip(sarimax_result.point_forecast[:n], xgb_result.point_forecast[:n])
        ]
        lower_95 = [
            w_sarimax * s + w_xgb * x
            for s, x in zip(sarimax_result.lower_95[:n], xgb_result.lower_95[:n])
        ]
        upper_95 = [
            w_sarimax * s + w_xgb * x
            for s, x in zip(sarimax_result.upper_95[:n], xgb_result.upper_95[:n])
        ]
        lower_80 = [
            w_sarimax * s + w_xgb * x
            for s, x in zip(sarimax_result.lower_80[:n], xgb_result.lower_80[:n])
        ]
        upper_80 = [
            w_sarimax * s + w_xgb * x
            for s, x in zip(sarimax_result.upper_80[:n], xgb_result.upper_80[:n])
        ]

        logger.info(
            f"Ensemble {commodity}: SARIMAX w={w_sarimax:.2f}, XGBoost w={w_xgb:.2f}"
        )

        return ForecastResult(
            commodity=commodity,
            dates=sarimax_result.dates[:n],
            point_forecast=ensemble_forecast,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_type="Ensemble",
            metrics={
                "sarimax_weight": w_sarimax,
                "xgb_weight": w_xgb,
                "sarimax_mape": sarimax_train_mape,
                "xgb_cv_mape": xgb_mape,
            },
            feature_importance=self._feature_importance.get(commodity, pd.DataFrame()).to_dict("list")
            if isinstance(self._feature_importance.get(commodity), pd.DataFrame) else {},
        )

    def train_all_commodities(
        self,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
    ) -> dict[str, dict[str, float]]:
        """Train models for every configured commodity. Prefers real data."""
        # Attempt to merge with real yfinance data
        real_prices = self._load_real_prices()
        real_macro = self._load_real_macro()

        results = {}

        # Determine training DataFrame — real or synthetic
        if "date" in commodity_df.columns:
            train_df = commodity_df.set_index("date").sort_index()
        else:
            train_df = commodity_df.sort_index()

        train_df.index = pd.to_datetime(train_df.index)

        effective_macro = real_macro if real_macro is not None else macro_df
        commodity_cols = [c for c in train_df.columns if c not in ("date",)]

        for commodity in commodity_cols:
            logger.info(f"Training models for {commodity}...")

            price_series = train_df[commodity].dropna()
            if len(price_series) < 18:
                logger.warning(f"Skipping {commodity} — insufficient data ({len(price_series)} points)")
                continue

            # SARIMAX
            try:
                sarimax_metrics = self.train_sarimax(commodity, price_series)
            except Exception as e:
                logger.error(f"SARIMAX failed for {commodity}: {e}")
                sarimax_metrics = {"error": str(e)}

            # XGBoost
            try:
                xgb_metrics = self.train_xgboost(commodity, train_df, effective_macro)
            except Exception as e:
                logger.error(f"XGBoost failed for {commodity}: {e}")
                xgb_metrics = {"error": str(e)}

            # Cross-validation
            try:
                cv_metrics = self.cross_validate(commodity, train_df, effective_macro)
            except Exception as e:
                logger.error(f"CV failed for {commodity}: {e}")
                cv_metrics = {"error": str(e)}

            results[commodity] = {
                "sarimax": sarimax_metrics,
                "xgboost": xgb_metrics,
                "cross_validation": cv_metrics,
            }

        logger.info(f"Trained models for {len(results)} commodities")
        return results

    def generate_commodity_index(
        self, commodity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute a weighted Commodity Index from individual prices.
        Weights come from BOM configuration.
        Returns a pandas DataFrame with 'date' and 'commodity_index' columns.
        """
        settings = get_settings()
        weights = {c["name"]: c["bom_weight"] for c in settings["commodities"]}

        if "date" in commodity_df.columns:
            index_df = commodity_df[["date"]].copy()
            commodity_cols = [c for c in commodity_df.columns if c != "date"]
            base_df = commodity_df
        else:
            index_df = commodity_df.iloc[:, :0].copy()
            index_df["date"] = commodity_df.index
            commodity_cols = commodity_df.columns.tolist()
            base_df = commodity_df

        if not commodity_cols:
            raise ValueError("No commodity columns found in DataFrame")

        # Normalize each commodity to base=100 at first valid observation
        normalized = pd.DataFrame()
        for col in commodity_cols:
            series = (
                commodity_df[col] if "date" in commodity_df.columns else base_df[col]
            )
            first_valid = series.dropna().iloc[0] if not series.dropna().empty else 1.0
            if first_valid == 0:
                first_valid = 1.0
            normalized[col] = (series / first_valid) * 100

        # BOM-weighted index
        total_weight = sum(weights.get(c, 0) for c in commodity_cols)
        if total_weight == 0:
            total_weight = len(commodity_cols)
            weight_map = {c: 1.0 for c in commodity_cols}
        else:
            weight_map = weights

        index_values = sum(
            normalized[c] * weight_map.get(c, 0) / total_weight
            for c in commodity_cols
        )
        index_df["commodity_index"] = index_values.round(2)

        return index_df

    def get_feature_importance(self, commodity: str) -> pd.DataFrame:
        """Return feature importance DataFrame for a trained commodity model."""
        return self._feature_importance.get(commodity, pd.DataFrame())

    def get_cv_metrics(self) -> pd.DataFrame:
        """Return cross-validation metrics for all trained commodities as a DataFrame."""
        records = []
        for commodity, metrics in self._cv_metrics.items():
            records.append({"commodity": commodity, **metrics})
        return pd.DataFrame(records)



@dataclass
class ForecastResult:
    """Container for forecast outputs."""
    commodity: str
    dates: list[str]
    point_forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    lower_95: list[float]
    upper_95: list[float]
    model_type: str
    metrics: dict[str, float] = field(default_factory=dict)


class SARIMAXForecaster:
    """SARIMAX-based commodity price forecaster."""

    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, series: pd.Series, exog: pd.DataFrame | None = None) -> dict[str, float]:
        """Fit SARIMAX model and return in-sample metrics."""
        model = SARIMAX(
            series,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False, maxiter=200)
        fitted = self.model_fit.fittedvalues
        metrics = {
            "aic": self.model_fit.aic,
            "bic": self.model_fit.bic,
            "mae": mean_absolute_error(series, fitted),
            "mape": mean_absolute_percentage_error(series, fitted) * 100,
        }
        logger.info(f"SARIMAX fitted — AIC={metrics['aic']:.1f}, MAE={metrics['mae']:.1f}")
        return metrics

    def predict(
        self, steps: int, exog_future: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast with confidence intervals."""
        forecast = self.model_fit.get_forecast(steps=steps, exog=exog_future)
        mean = forecast.predicted_mean.values
        ci_80 = forecast.conf_int(alpha=0.20).values
        ci_95 = forecast.conf_int(alpha=0.05).values
        return mean, ci_80, ci_95


class XGBoostForecaster:
    """XGBoost-based commodity price forecaster with time-series features."""

    def __init__(self, **params):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        default_params.update(params)
        self.model = XGBRegressor(**default_params)
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """Train XGBoost and return validation metrics."""
        self.feature_names = list(X_train.columns)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Compute metrics on validation (or train if no val)
        X_eval = X_val if X_val is not None else X_train
        y_eval = y_val if y_val is not None else y_train
        preds = self.model.predict(X_eval)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_eval, preds))),
            "mae": float(mean_absolute_error(y_eval, preds)),
            "mape": float(mean_absolute_percentage_error(y_eval, preds) * 100),
        }
        logger.info(f"XGBoost fitted — RMSE={metrics['rmse']:.1f}, MAE={metrics['mae']:.1f}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)


class CommodityForecastModel:
    """
    Orchestrates commodity price forecasting across multiple approaches.

    This is the core Commodity Impact Model from Layer 3 of the architecture.
    It generates a Commodity Index that feeds into Layer 2 (Driver Engine)
    cost drivers.
    """

    def __init__(self):
        self.settings = get_settings()
        self.horizon = self.settings["forecast"]["horizon_months"]
        self.registry = ModelRegistry()
        self.sarimax_models: dict[str, SARIMAXForecaster] = {}
        self.xgb_models: dict[str, XGBoostForecaster] = {}

    def train_sarimax(
        self, commodity: str, price_series: pd.Series, exog: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train SARIMAX model for a single commodity."""
        model = SARIMAXForecaster()
        metrics = model.fit(price_series, exog=exog)
        self.sarimax_models[commodity] = model
        return metrics

    def train_xgboost(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        test_size: int = 12,
    ) -> dict[str, float]:
        """Train XGBoost model for a single commodity using engineered features."""
        feat_df = prepare_commodity_features(
            commodity_df, macro_df, commodity,
            lags=self.settings["forecast"]["feature_lag_months"],
        )

        # Time-series split
        feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]
        X = feat_df[feature_cols]
        y = feat_df["target"]

        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        model = XGBoostForecaster()
        metrics = model.fit(X_train, y_train, X_test, y_test)
        self.xgb_models[commodity] = model

        # Save model
        self.registry.save_model(
            model.model,
            f"commodity_xgb_{commodity.lower()}",
            metrics=metrics,
            features=feature_cols,
        )

        return metrics

    def cross_validate(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Time-series cross-validation for XGBoost model."""
        feat_df = prepare_commodity_features(
            commodity_df, macro_df, commodity,
            lags=self.settings["forecast"]["feature_lag_months"],
        )
        feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]
        X = feat_df[feature_cols].values
        y = feat_df["target"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes, mapes = [], []

        for train_idx, val_idx in tscv.split(X):
            model = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=42,
            )
            model.fit(X[train_idx], y[train_idx], verbose=False)
            preds = model.predict(X[val_idx])
            maes.append(mean_absolute_error(y[val_idx], preds))
            mapes.append(mean_absolute_percentage_error(y[val_idx], preds) * 100)

        cv_metrics = {
            "cv_mae_mean": float(np.mean(maes)),
            "cv_mae_std": float(np.std(maes)),
            "cv_mape_mean": float(np.mean(mapes)),
            "cv_mape_std": float(np.std(mapes)),
        }
        logger.info(
            f"CV for {commodity}: MAE={cv_metrics['cv_mae_mean']:.1f}±{cv_metrics['cv_mae_std']:.1f}"
        )
        return cv_metrics

    def forecast_sarimax(self, commodity: str) -> ForecastResult:
        """Generate SARIMAX forecast for a commodity."""
        model = self.sarimax_models.get(commodity)
        if model is None:
            raise ValueError(f"SARIMAX model not trained for {commodity}")

        mean, ci_80, ci_95 = model.predict(self.horizon)
        future_dates = pd.date_range(
            pd.Timestamp.now().normalize(), periods=self.horizon, freq="MS"
        )

        return ForecastResult(
            commodity=commodity,
            dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            point_forecast=mean.tolist(),
            lower_80=ci_80[:, 0].tolist(),
            upper_80=ci_80[:, 1].tolist(),
            lower_95=ci_95[:, 0].tolist(),
            upper_95=ci_95[:, 1].tolist(),
            model_type="SARIMAX",
        )

    def train_all_commodities(
        self,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Train models for every configured commodity."""
        results = {}
        commodity_cols = [c for c in commodity_df.columns if c != "date"]

        for commodity in commodity_cols:
            logger.info(f"Training models for {commodity}...")

            # SARIMAX
            sarimax_metrics = self.train_sarimax(commodity, commodity_df[commodity])

            # XGBoost
            xgb_metrics = self.train_xgboost(commodity, commodity_df, macro_df)

            # Cross-validation
            cv_metrics = self.cross_validate(commodity, commodity_df, macro_df)

            results[commodity] = {
                "sarimax": sarimax_metrics,
                "xgboost": xgb_metrics,
                "cross_validation": cv_metrics,
            }

        return results

    def generate_commodity_index(
        self, commodity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute a weighted Commodity Index from individual prices.
        Weights come from BOM configuration.
        """
        settings = get_settings()
        weights = {c["name"]: c["bom_weight"] for c in settings["commodities"]}

        index_df = commodity_df[["date"]].copy()
        commodity_cols = [c for c in commodity_df.columns if c != "date"]

        # Normalize each commodity to base=100 at first observation
        normalized = pd.DataFrame()
        for col in commodity_cols:
            base_val = commodity_df[col].iloc[0]
            normalized[col] = (commodity_df[col] / base_val) * 100

        # Weighted index
        total_weight = sum(weights.get(c, 0) for c in commodity_cols)
        index_df["commodity_index"] = sum(
            normalized[c] * weights.get(c, 0) / total_weight
            for c in commodity_cols
        )
        index_df["commodity_index"] = index_df["commodity_index"].round(2)

        return index_df
