"""
Backtesting Dashboard — Commodity Forecast Model Validation

Allows analysts to:
  1. Run walk-forward backtests on the fly (or load saved results)
  2. Inspect per-fold actual vs predicted charts
  3. Compare model accuracy across commodities and strategies
  4. Identify systematic biases and areas for improvement
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.helpers import format_pct, load_parquet


def render():
    st.title("Commodity Forecast Backtesting")
    st.markdown(
        "**Walk-forward out-of-sample validation** — no look-ahead bias. "
        "Models are retrained from scratch on each fold's training window."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Run Backtest", "Results Summary", "Fold Detail"])

    with tab1:
        _render_run_backtest()
    with tab2:
        _render_results_summary()
    with tab3:
        _render_fold_detail()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Run Backtest
# ─────────────────────────────────────────────────────────────────────────────

def _render_run_backtest():
    """Interactive backtest runner."""
    st.subheader("Configure & Run Backtest")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Backtest Parameters**")
        strategy = st.radio(
            "Validation Strategy",
            ["Expanding Window", "Rolling Window"],
            help=(
                "**Expanding**: training set grows each fold — tests absolute improvement.\n\n"
                "**Rolling**: fixed-size window slides forward — tests model freshness."
            ),
        )
        initial_train = st.slider(
            "Initial Training Window (months)", 18, 48, 24,
            help="Minimum months of data before the first out-of-sample test."
        )
        test_per_fold = st.slider(
            "Test Period per Fold (months)", 1, 6, 3,
            help="How many months to forecast out-of-sample in each fold."
        )
        rolling_window = st.slider(
            "Rolling Window (months)", 24, 60, 36,
            help="Only relevant for 'Rolling Window' strategy."
        )
        n_folds_limit = st.number_input(
            "Max Folds (0 = all)", 0, 20, 0,
            help="Limit number of folds to speed up testing. 0 = run all available."
        )

        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            _execute_backtest(
                strategy="expanding" if strategy == "Expanding Window" else "rolling",
                initial_train=initial_train,
                test_per_fold=test_per_fold,
                rolling_window=rolling_window,
                n_folds=int(n_folds_limit) if n_folds_limit > 0 else None,
            )
        else:
            st.info(
                "Configure parameters on the left, then click **Run Backtest**.\n\n"
                "The backtest will:\n"
                "1. Load real commodity price data (yfinance)\n"
                "2. Build rich time-series features (RSI, momentum, rolling z-scores)\n"
                "3. Train XGBoost on expanding/rolling training windows\n"
                "4. Measure out-of-sample prediction accuracy per fold\n"
                "5. Save detailed results to `data/processed/`"
            )


def _execute_backtest(
    strategy: str,
    initial_train: int,
    test_per_fold: int,
    rolling_window: int,
    n_folds: int | None,
):
    """Run the actual walk-forward backtest and display results inline."""
    from src.models.backtesting import WalkForwardBacktester, run_commodity_backtesting

    commodity_df = load_parquet("market_commodities")
    if commodity_df is None:
        commodity_df = load_parquet("commodity_prices")

    if commodity_df is None:
        st.error("No commodity data found. Run `python scripts/fetch_data.py` or `python scripts/generate_data.py` first.")
        return

    with st.spinner("Running walk-forward backtest... this may take 1-2 minutes."):
        try:
            backtester = WalkForwardBacktester(
                strategy=strategy,
                initial_train_months=initial_train,
                test_months_per_fold=test_per_fold,
                rolling_window_months=rolling_window,
                n_folds=n_folds,
            )
            pandas_df = commodity_df.to_pandas()

            # Normalize date column / index
            import pandas as pd
            if "date" in pandas_df.columns:
                pandas_df = pandas_df.set_index(pd.to_datetime(pandas_df["date"])).drop(columns=["date"])
            pandas_df.index = pd.to_datetime(pandas_df.index)

            reports = backtester.run_all_commodities(pandas_df, macro_df=None)
            backtester.save_results(reports)

            st.success(f"Backtest complete! {len(reports)} commodities tested.")

            # Show quick summary table
            rows = []
            for commodity, report in reports.items():
                rows.append({
                    "Commodity": commodity,
                    "Folds": report.n_folds,
                    "MAPE": f"{report.mean_mape:.1f}%",
                    "RMSE": f"{report.mean_rmse:.1f}",
                    "Dir. Accuracy": f"{report.mean_directional_accuracy:.0f}%",
                    "Hit Rate (10%)": f"{report.mean_hit_rate_10pct:.0f}%",
                    "Bias": f"{report.mean_bias:+.1f}",
                })

            import polars as pl
            st.dataframe(
                pl.DataFrame(rows).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )

            # Highlight best/worst
            mapes = {c: r.mean_mape for c, r in reports.items() if not np.isnan(r.mean_mape)}
            if mapes:
                best = min(mapes, key=mapes.get)
                worst = max(mapes, key=mapes.get)
                col_a, col_b = st.columns(2)
                col_a.success(f"Best forecast: **{best}** (MAPE = {mapes[best]:.1f}%)")
                col_b.warning(f"Hardest to forecast: **{worst}** (MAPE = {mapes[worst]:.1f}%)")

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Results Summary (load from parquet)
# ─────────────────────────────────────────────────────────────────────────────

def _render_results_summary():
    """Load and display saved backtest summary."""
    st.subheader("Backtest Results Summary")

    try:
        from src.models.backtesting import load_backtest_results
        summary_df, detail_df = load_backtest_results()
    except Exception as e:
        st.warning(f"Could not load results: {e}")
        summary_df = None
        detail_df = None

    if summary_df is None or len(summary_df) == 0:
        st.info("No saved backtest results. Run a backtest first in the **Run Backtest** tab.")
        return

    summary_pd = summary_df.to_pandas()

    # ── Accuracy Bar Chart ────────────────────────────────────────────────────
    st.markdown("#### Forecast Accuracy by Commodity (MAPE)")
    fig_mape = px.bar(
        summary_pd.sort_values("mean_mape"),
        x="commodity",
        y="mean_mape",
        error_y="std_mape",
        color="strategy",
        barmode="group",
        labels={"mean_mape": "Mean MAPE (%)", "commodity": "Commodity"},
        template="plotly_dark",
        color_discrete_sequence=["#00d4aa", "#ff6b6b"],
    )
    fig_mape.update_layout(height=350)
    st.plotly_chart(fig_mape, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # ── Directional Accuracy ──────────────────────────────────────────────
        st.markdown("#### Directional Accuracy (%)")
        st.markdown("*(% of time model predicts correct price direction)*")
        fig_dir = px.bar(
            summary_pd.sort_values("mean_directional_accuracy", ascending=False),
            x="commodity",
            y="mean_directional_accuracy",
            color="strategy",
            barmode="group",
            labels={"mean_directional_accuracy": "Directional Accuracy (%)"},
            template="plotly_dark",
            color_discrete_sequence=["#007aff", "#ff9f0a"],
        )
        fig_dir.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% (random baseline)")
        fig_dir.update_layout(height=300)
        st.plotly_chart(fig_dir, use_container_width=True)

    with col_right:
        # ── Hit Rate ─────────────────────────────────────────────────────────
        st.markdown("#### Hit Rate — ±10% of Actual")
        st.markdown("*(% of predictions within ±10% of true price)*")
        fig_hit = px.bar(
            summary_pd.sort_values("mean_hit_rate_10pct", ascending=False),
            x="commodity",
            y="mean_hit_rate_10pct",
            color="strategy",
            barmode="group",
            labels={"mean_hit_rate_10pct": "Hit Rate (%)"},
            template="plotly_dark",
            color_discrete_sequence=["#30d158", "#ff6b6b"],
        )
        fig_hit.update_layout(height=300)
        st.plotly_chart(fig_hit, use_container_width=True)

    # ── Detailed Table ────────────────────────────────────────────────────────
    st.markdown("#### Full Results Table")
    display_cols = [
        "commodity", "strategy", "n_folds",
        "mean_mape", "std_mape",
        "mean_rmse", "mean_directional_accuracy",
        "mean_hit_rate_10pct", "mean_bias",
        "training_time_sec",
    ]
    available_cols = [c for c in display_cols if c in summary_pd.columns]
    st.dataframe(
        summary_pd[available_cols].round(2),
        use_container_width=True,
        hide_index=True,
    )

    # ── Bias Analysis ─────────────────────────────────────────────────────────
    if "mean_bias" in summary_pd.columns:
        st.markdown("#### Systematic Bias Analysis")
        st.markdown(
            "Positive bias = model consistently **overestimates** price. "
            "Negative = consistently **underestimates**."
        )
        fig_bias = px.bar(
            summary_pd,
            x="commodity",
            y="mean_bias",
            color="strategy",
            barmode="group",
            labels={"mean_bias": "Mean Bias ($)"},
            template="plotly_dark",
            color_discrete_sequence=["#ff9f0a", "#007aff"],
        )
        fig_bias.add_hline(y=0, line_color="white", line_dash="dash")
        fig_bias.update_layout(height=300)
        st.plotly_chart(fig_bias, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Fold Detail
# ─────────────────────────────────────────────────────────────────────────────

def _render_fold_detail():
    """Drill down into per-fold actual vs. predicted data."""
    st.subheader("Fold-Level Actual vs. Predicted")

    try:
        from src.models.backtesting import load_backtest_results
        _, detail_df = load_backtest_results()
    except Exception as e:
        st.warning(f"Could not load detail data: {e}")
        detail_df = None

    if detail_df is None or len(detail_df) == 0:
        st.info("No saved backtest results. Run a backtest first in the **Run Backtest** tab.")
        return

    detail_pd = detail_df.to_pandas()

    commodities = sorted(detail_pd["commodity"].unique().tolist())
    strategies = sorted(detail_pd["strategy"].unique().tolist())

    sel_commodity = st.selectbox("Select Commodity", commodities)
    sel_strategy = st.selectbox("Select Strategy", strategies)

    filtered = detail_pd[
        (detail_pd["commodity"] == sel_commodity) &
        (detail_pd["strategy"] == sel_strategy)
    ].copy()

    if filtered.empty:
        st.warning("No data for this combination.")
        return

    filtered = filtered.sort_values("test_date")

    # ── Actual vs Predicted over time ──────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["test_date"],
        y=filtered["actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#00d4aa", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=filtered["test_date"],
        y=filtered["prediction"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#ff9f0a", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title=f"{sel_commodity} — Actual vs Predicted ({sel_strategy})",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # ── Percentage Error over time ────────────────────────────────────
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=filtered["test_date"],
            y=filtered["pct_error"],
            name="% Error",
            marker_color=[
                "#ff6b6b" if abs(e) > 10 else "#00d4aa"
                for e in filtered["pct_error"]
            ],
        ))
        fig_err.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="+10%")
        fig_err.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10%")
        fig_err.add_hline(y=0, line_color="white")
        fig_err.update_layout(
            title="Percentage Error by Period",
            yaxis_title="% Error",
            height=300,
            template="plotly_dark",
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with col_b:
        # ── Error distribution ────────────────────────────────────────────
        fig_hist = px.histogram(
            filtered,
            x="pct_error",
            nbins=30,
            title="Error Distribution",
            labels={"pct_error": "% Error"},
            template="plotly_dark",
            color_discrete_sequence=["#007aff"],
        )
        fig_hist.add_vline(x=0, line_color="white", line_dash="dash")
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Fold-level Metrics Table ──────────────────────────────────────────────
    st.markdown("#### Fold-Level Metrics")
    fold_agg = filtered.groupby("fold_id").agg(
        n_obs=("actual", "count"),
        mae=("error", lambda x: np.mean(np.abs(x))),
        mape=("abs_pct_error", "mean"),
        bias=("error", "mean"),
    ).reset_index()
    fold_agg.columns = ["Fold", "Obs", "MAE", "MAPE (%)", "Bias"]
    fold_agg = fold_agg.round(2)
    st.dataframe(fold_agg, use_container_width=True, hide_index=True)
