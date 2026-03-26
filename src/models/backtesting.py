"""
Commodity Forecast Backtesting Framework

Rigorous walk-forward out-of-sample backtesting for commodity price forecasts.
Implements two strategies:

  1. Expanding-Window (anchored): Training set grows from initial window forward.
     Good for data-scarce commodities; shows how models improve with more history.

  2. Rolling-Window: Fixed-length training window slides forward.
     Tests whether model stays fresh and avoids stale patterns.

Key metrics per commodity, per fold:
  - MAE, RMSE, MAPE — magnitude of error
  - Directional Accuracy — % of times direction of move is correct (binary)
  - Hit Rate (10%) — % of predictions within ±10% of actual
  - Bias — systematic over/under-estimation

Results are saved to data/processed/backtest_results.parquet and
data/processed/backtest_summary.parquet for dashboarding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

from src.config import get_project_root, get_settings
from src.data.feature_engineering import prepare_commodity_features


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Result of a single backtest fold."""
    commodity: str
    model_type: str
    strategy: str
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    dates: list[str]
    actuals: list[float]
    predictions: list[float]
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    hit_rate_10pct: float
    bias: float  # mean(pred - actual) — positive = overestimate


@dataclass
class BacktestReport:
    """Aggregated backtesting report across all folds."""
    commodity: str
    model_type: str
    strategy: str
    n_folds: int
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_mape: float
    std_mape: float
    mean_directional_accuracy: float
    mean_hit_rate_10pct: float
    mean_bias: float
    fold_results: list[FoldResult] = field(default_factory=list)
    training_time_sec: float = 0.0

    def to_dict(self) -> dict:
        """Flat dictionary for storing in parquet."""
        return {
            "commodity": self.commodity,
            "model_type": self.model_type,
            "strategy": self.strategy,
            "n_folds": self.n_folds,
            "mean_mae": self.mean_mae,
            "std_mae": self.std_mae,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_mape": self.mean_mape,
            "std_mape": self.std_mape,
            "mean_directional_accuracy": self.mean_directional_accuracy,
            "mean_hit_rate_10pct": self.mean_hit_rate_10pct,
            "mean_bias": self.mean_bias,
            "training_time_sec": self.training_time_sec,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core backtester
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward backtester for commodity price forecasting models.

    Supports two strategies:
      - "expanding": initial training window grows fold by fold
      - "rolling": fixed-size training window slides forward

    The model is retrained from scratch on each fold's training data and
    evaluated on the held-out test period. This eliminates look-ahead bias.

    Usage:
        backtester = WalkForwardBacktester()
        report = backtester.run(commodity_df, macro_df, commodity="Lithium")
        backtester.save_results([report])
    """

    XGB_PARAMS = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
        "verbosity": 0,
    }

    def __init__(
        self,
        strategy: Literal["expanding", "rolling"] = "expanding",
        initial_train_months: int = 24,
        test_months_per_fold: int = 3,
        rolling_window_months: int = 36,
        n_folds: int | None = None,
        lags: list[int] | None = None,
    ):
        self.strategy = strategy
        self.initial_train = initial_train_months
        self.test_per_fold = test_months_per_fold
        self.rolling_window = rolling_window_months
        self.n_folds = n_folds
        self.settings = get_settings()
        self.lags = lags or self.settings["forecast"].get("feature_lag_months", [1, 3, 6, 12])

    def _get_fold_indices(self, n: int) -> list[tuple[range, range]]:
        """Generate (train_indices, test_indices) pairs for each fold."""
        folds = []
        start = self.initial_train
        while start + self.test_per_fold <= n:
            if self.strategy == "expanding":
                train_range = range(0, start)
            else:
                train_start = max(0, start - self.rolling_window)
                train_range = range(train_start, start)

            test_end = min(start + self.test_per_fold, n)
            test_range = range(start, test_end)

            if len(train_range) >= 12 and len(test_range) >= 1:
                folds.append((train_range, test_range))

            start += self.test_per_fold

            if self.n_folds is not None and len(folds) >= self.n_folds:
                break

        return folds

    def _compute_fold_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """Compute all fold-level metrics."""
        mae = float(mean_absolute_error(actuals, predictions))
        rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
        mape = float(mean_absolute_percentage_error(actuals, predictions) * 100)
        bias = float(np.mean(predictions - actuals))

        # Directional accuracy: does prediction go the same direction as actual?
        if len(actuals) > 1:
            actual_dirs = np.sign(np.diff(actuals))
            pred_dirs = np.sign(np.diff(predictions))
            dir_acc = float(np.mean(actual_dirs == pred_dirs) * 100)
        else:
            dir_acc = float("nan")

        # Hit rate: % of predictions within ±10% of actual
        pct_errors = np.abs((predictions - actuals) / (np.abs(actuals) + 1e-9))
        hit_rate = float(np.mean(pct_errors <= 0.10) * 100)

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "hit_rate_10pct": hit_rate,
            "bias": bias,
        }

    def _train_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> XGBRegressor:
        """Train a fresh XGBoost model on a fold's training data."""
        model = XGBRegressor(**self.XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        return model

    def run(
        self,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        commodity: str,
        model_type: Literal["xgboost"] = "xgboost",
    ) -> BacktestReport:
        """
        Run full walk-forward backtest for one commodity.

        Args:
            commodity_df: DataFrame with DatetimeIndex or 'date' column + commodity columns
            macro_df: Optional macro context for feature engineering
            commodity: Name of the commodity column to forecast
            model_type: Model algorithm to use (only "xgboost" for now)

        Returns:
            BacktestReport with per-fold and aggregate metrics
        """
        t0 = time.time()
        logger.info(f"Starting {self.strategy} backtest for {commodity} ({model_type})")

        # Ensure datetime index
        df = commodity_df.copy()
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Reset index to have date column for prepare_commodity_features
        df_with_date = df.reset_index().rename(columns={"index": "date"})
        if "date" not in df_with_date.columns and df.index.name:
            df_with_date = df.reset_index().rename(columns={df.index.name: "date"})

        # Build macro_df if None: use empty DataFrame with date column
        if macro_df is None:
            effective_macro = pd.DataFrame({"date": df_with_date["date"]})
        else:
            effective_macro = macro_df

        if commodity not in df.columns:
            raise ValueError(f"Commodity '{commodity}' not found in DataFrame. "
                             f"Available: {df.columns.tolist()}")

        # Build feature matrix (uses full data for feature col consistency)
        feat_df = prepare_commodity_features(df_with_date, effective_macro, commodity, lags=self.lags)

        feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]
        X_all = feat_df[feature_cols].values
        y_all = feat_df["target"].values
        # Preserve dates for fold metadata — convert to numpy array for positional indexing
        if "date" in feat_df.columns:
            dates_all = pd.to_datetime(feat_df["date"]).values
        elif isinstance(feat_df.index, pd.DatetimeIndex):
            dates_all = feat_df.index.values
        else:
            # Fall back to the known dates from df_with_date aligned to feat_df length
            dates_all = pd.to_datetime(
                df_with_date["date"].iloc[-len(feat_df):].reset_index(drop=True)
            ).values

        if len(X_all) < self.initial_train + self.test_per_fold:
            logger.warning(
                f"Skipping {commodity}: not enough data "
                f"({len(X_all)} < {self.initial_train + self.test_per_fold})"
            )
            return BacktestReport(
                commodity=commodity, model_type=model_type, strategy=self.strategy,
                n_folds=0, mean_mae=float("nan"), std_mae=float("nan"),
                mean_rmse=float("nan"), std_rmse=float("nan"),
                mean_mape=float("nan"), std_mape=float("nan"),
                mean_directional_accuracy=float("nan"),
                mean_hit_rate_10pct=float("nan"), mean_bias=float("nan"),
            )

        folds = self._get_fold_indices(len(X_all))
        fold_results = []

        for fold_id, (train_idx, test_idx) in enumerate(folds):
            X_train = X_all[list(train_idx)]
            y_train = y_all[list(train_idx)]
            X_test = X_all[list(test_idx)]
            y_test = y_all[list(test_idx)]
            dates_test = dates_all[list(test_idx)]

            model = self._train_xgb(X_train, y_train)
            preds = model.predict(X_test)

            metrics = self._compute_fold_metrics(y_test, preds)

            train_dates = dates_all[list(train_idx)]
            # Convert numpy datetime64 to date strings
            _fmt = lambda d: str(pd.Timestamp(d).date())
            fold_result = FoldResult(
                commodity=commodity,
                model_type=model_type,
                strategy=self.strategy,
                fold_id=fold_id,
                train_start=_fmt(train_dates[0]),
                train_end=_fmt(train_dates[-1]),
                test_start=_fmt(dates_test[0]),
                test_end=_fmt(dates_test[-1]),
                dates=[_fmt(d) for d in dates_test],
                actuals=y_test.tolist(),
                predictions=preds.tolist(),
                **metrics,
            )
            fold_results.append(fold_result)

        # Aggregate across folds
        maes = [f.mae for f in fold_results if not np.isnan(f.mae)]
        rmses = [f.rmse for f in fold_results if not np.isnan(f.rmse)]
        mapes = [f.mape for f in fold_results if not np.isnan(f.mape)]
        dir_accs = [f.directional_accuracy for f in fold_results
                    if not np.isnan(f.directional_accuracy)]
        hit_rates = [f.hit_rate_10pct for f in fold_results
                     if not np.isnan(f.hit_rate_10pct)]
        biases = [f.bias for f in fold_results if not np.isnan(f.bias)]

        report = BacktestReport(
            commodity=commodity,
            model_type=model_type,
            strategy=self.strategy,
            n_folds=len(fold_results),
            mean_mae=float(np.mean(maes)) if maes else float("nan"),
            std_mae=float(np.std(maes)) if maes else float("nan"),
            mean_rmse=float(np.mean(rmses)) if rmses else float("nan"),
            std_rmse=float(np.std(rmses)) if rmses else float("nan"),
            mean_mape=float(np.mean(mapes)) if mapes else float("nan"),
            std_mape=float(np.std(mapes)) if mapes else float("nan"),
            mean_directional_accuracy=float(np.mean(dir_accs)) if dir_accs else float("nan"),
            mean_hit_rate_10pct=float(np.mean(hit_rates)) if hit_rates else float("nan"),
            mean_bias=float(np.mean(biases)) if biases else float("nan"),
            fold_results=fold_results,
            training_time_sec=time.time() - t0,
        )

        logger.info(
            f"Backtest complete: {commodity} | {len(fold_results)} folds | "
            f"MAPE={report.mean_mape:.1f}% | Dir.Acc={report.mean_directional_accuracy:.0f}%"
        )
        return report

    def run_all_commodities(
        self,
        commodity_df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        commodities: list[str] | None = None,
    ) -> dict[str, BacktestReport]:
        """Run backtesting for all available/configured commodities."""
        df = commodity_df.copy()
        if "date" in df.columns:
            available_cols = [c for c in df.columns if c != "date"]
        elif isinstance(df.index, pd.DatetimeIndex):
            available_cols = df.columns.tolist()
        else:
            available_cols = df.columns.tolist()

        if commodities is None:
            settings = get_settings()
            configured = [c["name"] for c in settings["commodities"]]
            commodities = [c for c in configured if c in available_cols]

        if not commodities:
            commodities = available_cols

        reports: dict[str, BacktestReport] = {}
        for commodity in commodities:
            try:
                report = self.run(commodity_df, macro_df, commodity)
                reports[commodity] = report
            except Exception as e:
                logger.error(f"Backtest failed for {commodity}: {e}")

        logger.info(f"Completed backtesting for {len(reports)} commodities")
        return reports

    def save_results(
        self, reports: dict[str, BacktestReport] | list[BacktestReport]
    ) -> None:
        """Save backtesting results to parquet files."""
        root = get_project_root()
        output_dir = root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize input
        if isinstance(reports, dict):
            report_list = list(reports.values())
        else:
            report_list = reports

        # ── Summary (one row per commodity) ───────────────────────────────────
        summary_rows = [r.to_dict() for r in report_list]
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            summary_path = output_dir / "backtest_summary.parquet"
            summary_df.write_parquet(summary_path)
            logger.info(f"Saved backtest summary to {summary_path}")

        # ── Fold-level detail (all predictions vs actuals) ────────────────────
        detail_rows = []
        for report in report_list:
            for fold in report.fold_results:
                for i, (date, actual, pred) in enumerate(
                    zip(fold.dates, fold.actuals, fold.predictions)
                ):
                    detail_rows.append({
                        "commodity": fold.commodity,
                        "model_type": fold.model_type,
                        "strategy": fold.strategy,
                        "fold_id": fold.fold_id,
                        "train_start": fold.train_start,
                        "train_end": fold.train_end,
                        "test_date": date,
                        "actual": actual,
                        "prediction": pred,
                        "error": pred - actual,
                        "pct_error": (pred - actual) / (abs(actual) + 1e-9) * 100,
                        "abs_pct_error": abs(pred - actual) / (abs(actual) + 1e-9) * 100,
                    })

        if detail_rows:
            detail_df = pl.DataFrame(detail_rows)
            detail_path = output_dir / "backtest_results.parquet"
            detail_df.write_parquet(detail_path)
            logger.info(f"Saved backtest detail ({len(detail_rows)} rows) to {detail_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_commodity_backtesting(
    commodity_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    commodities: list[str] | None = None,
) -> dict[str, BacktestReport]:
    """
    One-call backtesting shortcut. Runs both expanding and rolling strategies
    for each commodity and saves combined results.

    Args:
        commodity_df: pandas DataFrame with DatetimeIndex or 'date' col + commodity cols
        macro_df: optional macro context (FRED, market indices)
        commodities: list of commodity names to test. Defaults to configured ones.

    Returns:
        dict mapping "{commodity}_{strategy}" -> BacktestReport
    """
    results: dict[str, BacktestReport] = {}

    for strategy in ("expanding", "rolling"):
        backtester = WalkForwardBacktester(
            strategy=strategy,
            initial_train_months=24,
            test_months_per_fold=3,
            rolling_window_months=36,
        )
        strategy_reports = backtester.run_all_commodities(
            commodity_df, macro_df, commodities
        )
        for commodity, report in strategy_reports.items():
            key = f"{commodity}_{strategy}"
            results[key] = report
        backtester.save_results(strategy_reports)

    return results


def load_backtest_results() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load saved backtest results from disk.

    Returns:
        (summary_df, detail_df) — Polars DataFrames
    """
    root = get_project_root()
    output_dir = root / "data" / "processed"

    summary_path = output_dir / "backtest_summary.parquet"
    detail_path = output_dir / "backtest_results.parquet"

    summary_df = pl.read_parquet(summary_path) if summary_path.exists() else pl.DataFrame()
    detail_df = pl.read_parquet(detail_path) if detail_path.exists() else pl.DataFrame()

    return summary_df, detail_df
