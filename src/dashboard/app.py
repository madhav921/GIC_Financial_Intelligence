"""
GIC Plan-to-Perform: Executive Dashboard

Multi-page Streamlit application for CFO-level financial intelligence.

Pages:
  1. Executive Summary — KPIs, alerts, market snapshot
  2. Commodity Intelligence — real-time prices, forecasts, correlations
  3. Financial P&L — revenue, COGS, margins, scenario impact
  4. Scenario Simulation — Monte Carlo, what-if analysis
  5. Market Monitor — live market data, crypto, FX
  6. Data Explorer — raw data inspection, quality metrics
"""

import streamlit as st

st.set_page_config(
    page_title="GIC Plan-to-Perform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.sidebar.title("GIC Plan-to-Perform")
    st.sidebar.markdown("### AI-Powered Financial Intelligence")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Summary",
            "Commodity Intelligence",
            "Financial P&L",
            "Scenario Simulation",
            "Market Monitor",
            "Backtesting",
            "Data Explorer",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**v0.4.0** | Real-World Data Pipeline\n\n"
        "Sources: Yahoo Finance, FRED, CCXT (Binance)"
    )

    # Show data source status
    from src.dashboard.helpers import detect_data_source
    src = detect_data_source("commodity_prices")
    if src == "real":
        st.sidebar.success("Data: Real-world (Yahoo Finance)")
    else:
        st.sidebar.warning("Data: Synthetic — run `fetch_data.py`")

    if page == "Executive Summary":
        from src.dashboard.pages import executive_summary
        executive_summary.render()
    elif page == "Commodity Intelligence":
        from src.dashboard.pages import commodity_intelligence
        commodity_intelligence.render()
    elif page == "Financial P&L":
        from src.dashboard.pages import financial_pnl
        financial_pnl.render()
    elif page == "Scenario Simulation":
        from src.dashboard.pages import scenario_simulation
        scenario_simulation.render()
    elif page == "Market Monitor":
        from src.dashboard.pages import market_monitor
        market_monitor.render()
    elif page == "Backtesting":
        from src.dashboard.pages import backtesting
        backtesting.render()
    elif page == "Data Explorer":
        from src.dashboard.pages import data_explorer
        data_explorer.render()


if __name__ == "__main__":
    main()
