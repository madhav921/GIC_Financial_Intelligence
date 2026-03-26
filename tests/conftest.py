"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_generator import (
    generate_bom_data,
    generate_commodity_prices,
    generate_macro_indicators,
    generate_production_inventory,
    generate_sales_data,
)


@pytest.fixture(scope="session")
def commodity_df():
    return generate_commodity_prices(periods=48, seed=99)


@pytest.fixture(scope="session")
def macro_df():
    return generate_macro_indicators(periods=48, seed=99)


@pytest.fixture(scope="session")
def sales_df():
    return generate_sales_data(periods=48, seed=99)


@pytest.fixture(scope="session")
def production_df():
    return generate_production_inventory(periods=48, seed=99)


@pytest.fixture(scope="session")
def bom_df():
    return generate_bom_data()
