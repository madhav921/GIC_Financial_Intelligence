"""
ERP / SAP connector placeholder.

PLACEHOLDER: This module will connect to SAP S/4HANA or equivalent ERP
to pull production orders, BOM costs, inventory levels, and GL data.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


class ERPConnector:
    """Interface for SAP/ERP system data extraction."""

    def __init__(self, host: str = "", port: int = 443):
        self.host = host
        self.port = port
        logger.info(f"ERPConnector initialised (host={host}, port={port}) — PLACEHOLDER")

    def fetch_production_orders(self, plant: str, period: str) -> pd.DataFrame:
        """
        Fetch production orders from SAP PP module.

        TODO: Use SAP RFC/BAPI via `pyrfc` or SAP OData APIs.
        Relevant tables: AUFK (Order Master), AFKO (Order Header),
                         AFPO (Order Item), JEST (Status)
        """
        raise NotImplementedError("SAP production order connector not implemented.")

    def fetch_bom_costs(self, material: str) -> pd.DataFrame:
        """
        Fetch Bill of Materials with cost roll-up from SAP.

        TODO: Use transaction CK13N or BAPI_MATERIAL_BOM_GROUP.
        """
        raise NotImplementedError("SAP BOM connector not implemented.")

    def fetch_gl_actuals(self, company_code: str, fiscal_year: int) -> pd.DataFrame:
        """
        Fetch GL account actuals for financial reconciliation.

        TODO: Use SAP FI tables: BKPF, BSEG, or CDS views.
        """
        raise NotImplementedError("SAP GL connector not implemented.")

    def fetch_inventory_levels(self, plant: str) -> pd.DataFrame:
        """
        Fetch current stock levels by material and storage location.

        TODO: Use SAP MM tables: MARD, MARC, MBEW.
        """
        raise NotImplementedError("SAP inventory connector not implemented.")
