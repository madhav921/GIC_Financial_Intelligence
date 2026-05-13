"""
Data Source Protocols — plug-and-play abstraction layer.

All business logic is written against these interfaces.
Swap data sources by changing config/settings.yaml — zero code change required.

Implementations:
  MarketDataSource  ← YFinanceMarketSource | ParquetMarketSource | SyntheticMarketSource
  OperationalDataSource ← SyntheticOperationalSource | ParquetOperationalSource
                        | SAPOperationalSource | SalesforceOperationalSource
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataSource(Protocol):
    """
    Commodity prices, macro indicators, FX rates.
    Implementations: YFinanceMarketSource, ParquetMarketSource.
    """

    def get_commodity_prices(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns monthly commodity prices.
        Columns: date + one column per commodity (e.g. Steel, Lithium, Copper …).
        """
        ...

    def get_macro_indicators(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns monthly macro indicators.
        Columns: date, gdp_growth_pct, interest_rate_pct, usd_gbp, usd_eur,
                 oil_price_usd, manufacturing_pmi, cpi_index, dxy_index, …
        """
        ...

    def get_fx_rates(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns monthly FX rates.
        Columns: date, usd_gbp, usd_eur, usd_jpy, usd_cny.
        """
        ...


@runtime_checkable
class OperationalDataSource(Protocol):
    """
    JLR internal business data — sales, production, COGS breakdown, BOM, inventory.
    Currently synthetic; plugs into SAP S/4HANA or Salesforce with zero code change.
    """

    def get_sales(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns monthly vehicle sales by segment.
        Columns: date, segment, volume, avg_price_usd, incentive_pct, region.
        """
        ...

    def get_production(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns monthly production & inventory data by segment.
        Columns: date, segment, production_units, sales_units,
                 ending_inventory, capacity_utilization_pct.
        """
        ...

    def get_cogs_detail(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns detailed COGS breakdown by segment and commodity.
        Columns: date, segment, commodity, cost_usd, volume.
        """
        ...

    def get_bom(self) -> pd.DataFrame:
        """
        Returns Bill of Materials: commodity cost weight by segment.
        Columns: segment, commodity, cost_per_unit_usd, bom_weight_pct.
        """
        ...

    def get_inventory(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Returns inventory valuation by segment.
        Columns: date, segment, ending_inventory, inventory_value_usd.
        """
        ...
