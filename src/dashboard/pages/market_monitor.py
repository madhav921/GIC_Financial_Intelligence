"""
Market Monitor — real-time market data, crypto, FX, and macro signals.

Shows:
  - Live market indices (S&P 500, VIX, Oil, Gold)
  - FX rate tracker
  - Crypto market overview
  - FRED macroeconomic indicators
  - Market regime indicator
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.helpers import format_currency, load_parquet


def render():
    st.title("Market Monitor")
    st.markdown("**Real-time market data from Yahoo Finance, CCXT & FRED**")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Market Indices", "FX Tracker", "Crypto Markets", "Macro Indicators"
    ])

    with tab1:
        _render_market_indices()
    with tab2:
        _render_fx_tracker()
    with tab3:
        _render_crypto()
    with tab4:
        _render_macro()


def _render_market_indices():
    """Market indices overview."""
    st.subheader("Market Indices")

    market_df = load_parquet("market_indices")
    if market_df is None:
        st.warning("No market index data. Run `python scripts/fetch_data.py` to fetch real-time data.")
        return

    date_col = "date" if "date" in market_df.columns else "Date"
    value_cols = [c for c in market_df.columns if c not in ("date", "Date")]

    # Latest values
    cols = st.columns(min(len(value_cols), 6))
    for i, col_name in enumerate(value_cols[:6]):
        series = market_df[col_name].drop_nulls().to_list()
        if len(series) >= 2:
            current = series[-1]
            prev = series[-2]
            delta = (current - prev) / prev * 100 if prev else 0
            with cols[i % len(cols)]:
                st.metric(col_name, format_currency(current), f"{delta:+.2f}%")

    # Chart
    selected = st.multiselect("Select Indices", value_cols, default=value_cols[:3])
    if selected:
        fig = go.Figure()
        for col in selected:
            series = market_df[col].to_list()
            base = series[0] if series[0] else 1
            normalized = [(v / base * 100 if v else None) for v in series]
            fig.add_trace(go.Scatter(
                x=market_df[date_col].to_list(),
                y=normalized,
                mode="lines", name=col,
                line=dict(width=2),
            ))

        fig.update_layout(
            yaxis_title="Normalized (Base = 100)",
            height=450,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width='stretch')


def _render_fx_tracker():
    """FX rate visualization."""
    st.subheader("Foreign Exchange Rates")

    fx_df = load_parquet("fx_rates")
    if fx_df is None:
        st.warning("No FX data. Run `python scripts/fetch_data.py`.")
        return

    date_col = "date" if "date" in fx_df.columns else "Date"
    fx_cols = [c for c in fx_df.columns if c not in ("date", "Date")]

    # Current rates
    cols = st.columns(min(len(fx_cols), 4))
    for i, pair in enumerate(fx_cols[:4]):
        series = fx_df[pair].drop_nulls().to_list()
        if len(series) >= 2:
            current = series[-1]
            prev = series[-2]
            delta = (current - prev) / prev * 100 if prev else 0
            with cols[i]:
                st.metric(pair.replace("_", "/"), f"{current:.4f}", f"{delta:+.2f}%")

    # Chart
    fig = go.Figure()
    for pair in fx_cols:
        fig.add_trace(go.Scatter(
            x=fx_df[date_col].to_list(),
            y=fx_df[pair].to_list(),
            mode="lines", name=pair.replace("_", "/"),
            line=dict(width=2),
        ))

    fig.update_layout(
        yaxis_title="Exchange Rate",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width='stretch')


def _render_crypto():
    """Crypto market overview."""
    st.subheader("Crypto Market Overview")
    st.markdown("*Data sourced from Binance via CCXT*")

    crypto_df = load_parquet("crypto_prices")
    if crypto_df is None:
        st.warning(
            "No crypto data available. Run `python scripts/fetch_data.py`. "
            "CCXT connects to Binance for real-time crypto data."
        )
        return

    date_col = "date" if "date" in crypto_df.columns else "Date"
    crypto_cols = [c for c in crypto_df.columns if c not in ("date", "Date")]

    # Current prices
    cols = st.columns(min(len(crypto_cols), 6))
    for i, coin in enumerate(crypto_cols[:6]):
        series = crypto_df[coin].drop_nulls().to_list()
        if len(series) >= 2:
            current = series[-1]
            prev = series[-2]
            delta = (current - prev) / prev * 100 if prev else 0
            with cols[i % len(cols)]:
                st.metric(coin, format_currency(current), f"{delta:+.2f}%")

    # Price chart
    selected_crypto = st.multiselect("Select Assets", crypto_cols, default=crypto_cols[:3])
    if selected_crypto:
        fig = go.Figure()
        for coin in selected_crypto:
            series = crypto_df[coin].to_list()
            base = series[0] if series[0] else 1
            normalized = [(v / base * 100 if v else None) for v in series]
            fig.add_trace(go.Scatter(
                x=crypto_df[date_col].to_list(),
                y=normalized,
                mode="lines", name=coin,
                line=dict(width=2),
            ))

        fig.update_layout(
            yaxis_title="Normalized (Base = 100)",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig, width='stretch')


def _render_macro():
    """FRED macroeconomic indicators."""
    st.subheader("Macroeconomic Indicators (FRED)")

    fred_df = load_parquet("fred_macro")
    if fred_df is None:
        st.warning(
            "No FRED data available. Set `FRED_API_KEY` in `.env` and run `python scripts/fetch_data.py`.\n\n"
            "Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

        # Show synthetic macro data instead
        macro_df = load_parquet("macro_indicators")
        if macro_df is not None:
            st.info("Showing synthetic macro data as fallback")
            fred_df = macro_df
        else:
            return

    date_col = "date" if "date" in fred_df.columns else "Date"
    indicator_cols = [c for c in fred_df.columns if c not in ("date", "Date")]

    if not indicator_cols:
        st.warning("No indicator columns found")
        return

    selected_indicators = st.multiselect(
        "Select Indicators",
        indicator_cols,
        default=indicator_cols[:4],
    )

    if selected_indicators:
        n_charts = len(selected_indicators)
        n_cols = min(n_charts, 2)
        n_rows = (n_charts + n_cols - 1) // n_cols

        for row in range(n_rows):
            chart_cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                idx = row * n_cols + col_idx
                if idx >= n_charts:
                    break
                indicator = selected_indicators[idx]
                with chart_cols[col_idx]:
                    series = fred_df[indicator].drop_nulls().to_list()
                    dates = fred_df[date_col].to_list()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates, y=series,
                        mode="lines",
                        line=dict(color="#00d4aa", width=2),
                        fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)",
                    ))
                    fig.update_layout(
                        title=indicator,
                        height=250,
                        template="plotly_dark",
                        margin=dict(l=40, r=20, t=40, b=30),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, width='stretch')
