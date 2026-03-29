"""
Commodity Intelligence — deep-dive into commodity price analytics.

Shows:
  - Individual commodity price charts (real-world data)
  - SARIMAX & XGBoost forecast comparison
  - Correlation heatmap across commodity basket
  - FFN performance analytics (Sharpe, drawdown, CAGR)
  - BOM-weighted commodity index
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st

from src.dashboard.helpers import format_currency, load_parquet


def render():
    st.title("Commodity Intelligence")
    st.markdown("**Real-time commodity price tracking, forecasts & risk analytics**")
    st.markdown("---")

    # ── Load Data ──
    market_df = load_parquet("market_commodities")
    synthetic_df = load_parquet("commodity_prices")
    prices_df = market_df if market_df is not None else synthetic_df

    if prices_df is None:
        st.error("No commodity data available. Run `python scripts/fetch_data.py` or `python scripts/generate_data.py`.")
        return

    date_col = "date" if "date" in prices_df.columns else "Date"
    value_cols = [c for c in prices_df.columns if c not in ("date", "Date")]

    # ── Commodity Selector ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Charts", "Correlation Matrix", "Performance Analytics", "Commodity Index"
    ])

    with tab1:
        _render_price_charts(prices_df, date_col, value_cols)

    with tab2:
        _render_correlation(prices_df, date_col, value_cols)

    with tab3:
        _render_performance(prices_df, date_col, value_cols)

    with tab4:
        _render_commodity_index(prices_df, date_col, value_cols)


def _render_price_charts(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """Individual commodity price charts with technical indicators."""
    st.subheader("Commodity Price Tracker")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox("Select Commodity", value_cols, index=0)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_bands = st.checkbox("Show Bollinger Bands", value=False)

    with col2:
        series = df.select(pl.col(date_col), pl.col(selected)).drop_nulls()
        dates = series[date_col].to_list()
        prices = series[selected].to_list()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode="lines", name=selected,
            line=dict(color="#00d4aa", width=2),
        ))

        if show_ma and len(prices) > 20:
            # 20-period and 50-period moving averages
            ma20 = _moving_average(prices, 20)
            ma50 = _moving_average(prices, 50) if len(prices) > 50 else None

            fig.add_trace(go.Scatter(
                x=dates, y=ma20,
                mode="lines", name="MA(20)",
                line=dict(color="#ff9500", width=1, dash="dash"),
            ))
            if ma50 is not None:
                fig.add_trace(go.Scatter(
                    x=dates, y=ma50,
                    mode="lines", name="MA(50)",
                    line=dict(color="#ff3b30", width=1, dash="dash"),
                ))

        if show_bands and len(prices) > 20:
            upper, lower = _bollinger_bands(prices, 20)
            fig.add_trace(go.Scatter(
                x=dates, y=upper,
                mode="lines", name="Upper Band",
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=lower,
                mode="lines", name="Lower Band",
                line=dict(color="rgba(255,255,255,0.2)", width=1),
                fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
            ))

        fig.update_layout(
            title=f"{selected} — Price History",
            yaxis_title="Price",
            height=450,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    # Stats row
    if len(prices) > 0:
        scol1, scol2, scol3, scol4, scol5 = st.columns(5)
        current = prices[-1]
        high = max(prices)
        low = min(prices)
        avg = sum(prices) / len(prices)
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1]]
        vol = float(np.std(returns)) * 100 if returns else 0

        scol1.metric("Current", format_currency(current))
        scol2.metric("52W High", format_currency(high))
        scol3.metric("52W Low", format_currency(low))
        scol4.metric("Average", format_currency(avg))
        scol5.metric("Volatility", f"{vol:.1f}%")


def _render_correlation(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """Correlation heatmap across commodities."""
    st.subheader("Commodity Correlation Matrix")
    st.markdown("*Returns-based correlation — identifies diversification opportunities and co-movement risks*")

    # Compute returns
    pdf = df.to_pandas()
    numeric_cols = [c for c in value_cols if c in pdf.columns]
    returns = pdf[numeric_cols].pct_change().dropna()

    if returns.empty or len(returns) < 5:
        st.warning("Not enough data to compute correlations")
        return

    corr = returns.corr()

    fig = px.imshow(
        corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(
        height=500,
        template="plotly_dark",
        title="Return Correlation Heatmap",
    )
    st.plotly_chart(fig, width='stretch')

    # Key insights
    st.markdown("**Key Correlations:**")
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            val = corr.iloc[i, j]
            if abs(val) > 0.6:
                high_corr.append((corr.index[i], corr.columns[j], val))

    if high_corr:
        for a, b, v in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
            direction = "positively" if v > 0 else "negatively"
            st.markdown(f"- **{a}** & **{b}**: {v:.2f} ({direction} correlated)")
    else:
        st.info("No strong correlations detected (all < 0.6)")


def _render_performance(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """FFN-style performance analytics."""
    st.subheader("Performance Analytics")

    # Compute performance metrics
    pdf = df.to_pandas()
    numeric_cols = [c for c in value_cols if c in pdf.columns]

    if not numeric_cols:
        st.warning("No numeric columns for analysis")
        return

    records = []
    for col in numeric_cols:
        series = pdf[col].dropna()
        if len(series) < 10:
            continue

        total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
        returns = series.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(12) * 100)  # annualized for monthly
        sharpe = float(returns.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else 0

        # Max drawdown
        cummax = series.cummax()
        drawdown = ((series - cummax) / cummax * 100)
        max_dd = float(drawdown.min())

        records.append({
            "Commodity": col,
            "Total Return": f"{total_return:.1f}%",
            "Ann. Volatility": f"{ann_vol:.1f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.1f}%",
            "Current Price": f"${series.iloc[-1]:,.2f}",
        })

    if records:
        perf_df = pl.DataFrame(records)
        st.dataframe(perf_df.to_pandas(), width='stretch', hide_index=True)

    # Drawdown chart
    st.subheader("Drawdown Analysis")
    selected_dd = st.multiselect("Select commodities", numeric_cols, default=numeric_cols[:3])

    if selected_dd:
        fig = go.Figure()
        for col in selected_dd:
            series = pdf[col].dropna()
            cummax = series.cummax()
            dd = (series - cummax) / cummax * 100
            fig.add_trace(go.Scatter(
                x=pdf[date_col], y=dd,
                mode="lines", name=col,
                fill="tozeroy",
            ))

        fig.update_layout(
            yaxis_title="Drawdown (%)",
            height=350,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, width='stretch')


def _render_commodity_index(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """BOM-weighted commodity index."""
    st.subheader("BOM-Weighted Commodity Index")
    st.markdown("*Composite index weighted by Bill of Materials cost impact*")

    from src.config import get_settings
    settings = get_settings()
    # Build weight lookup keyed by both commodity name AND yfinance ticker
    # so the index works whether we loaded real market data (tickers) or synthetic data (names)
    weights = {}
    display_name = {}
    for c in settings["commodities"]:
        weights[c["name"]] = c["bom_weight"]
        display_name[c["name"]] = c["name"]
        if "yfinance_ticker" in c:
            weights[c["yfinance_ticker"]] = c["bom_weight"]
            display_name[c["yfinance_ticker"]] = c["name"]

    # Build index
    pdf = df.to_pandas()
    available_commodities = [c for c in value_cols if c in weights]

    if not available_commodities:
        st.warning("No matching commodities found in BOM configuration")
        return

    # Normalize each to base 100
    normalized = {}
    for col in available_commodities:
        series = pdf[col].dropna()
        if len(series) > 0:
            base = series.iloc[0]
            normalized[col] = (series / base) * 100

    # Weighted index
    total_weight = sum(weights.get(c, 0) for c in available_commodities)
    index_series = sum(
        normalized[c] * weights.get(c, 0) / total_weight
        for c in available_commodities if c in normalized
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf[date_col], y=index_series,
        mode="lines", name="Commodity Index",
        line=dict(color="#00d4aa", width=3),
        fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)",
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Base = 100")

    fig.update_layout(
        yaxis_title="Index Value",
        height=400,
        template="plotly_dark",
    )
    st.plotly_chart(fig, width='stretch')

    # Weight breakdown
    st.subheader("BOM Weight Allocation")
    weight_data = [{"Commodity": display_name.get(c, c), "BOM Weight": f"{weights[c]:.0%}"}
                   for c in available_commodities]
    st.dataframe(pl.DataFrame(weight_data).to_pandas(), width='stretch', hide_index=True)


def _moving_average(values: list[float], window: int) -> list[float | None]:
    """Simple moving average."""
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        result[i] = sum(values[i - window + 1:i + 1]) / window
    return result


def _bollinger_bands(values: list[float], window: int, num_std: float = 2.0):
    """Bollinger Bands."""
    upper = [None] * len(values)
    lower = [None] * len(values)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1:i + 1]
        mean = sum(segment) / window
        std = (sum((x - mean) ** 2 for x in segment) / window) ** 0.5
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, lower
