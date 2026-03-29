"""
Data Explorer — inspect raw data, quality metrics, and dataset catalog.

Shows:
  - Available datasets catalog (Parquet, CSV, External)
  - Data previews with column stats
  - Data quality metrics (nulls, duplicates, date coverage)
  - Export capabilities
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from src.dashboard.helpers import load_parquet


_ROOT = Path(__file__).resolve().parents[3]


def render():
    st.title("Data Explorer")
    st.markdown("**Browse, inspect, and validate all datasets in the pipeline**")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Dataset Catalog", "Data Preview", "Quality Report"])

    with tab1:
        _render_catalog()
    with tab2:
        _render_preview()
    with tab3:
        _render_quality()


def _render_catalog():
    """List all available datasets."""
    st.subheader("Dataset Catalog")

    sources = {
        "External (Real-World)": _ROOT / "data" / "external",
        "Parquet (Cached)": _ROOT / "data" / "parquet",
        "Synthetic (Generated)": _ROOT / "data" / "synthetic",
        "Raw (Uploaded)": _ROOT / "data" / "raw",
    }

    for label, path in sources.items():
        if not path.exists():
            continue

        files = list(path.glob("*.*"))
        if not files:
            continue

        st.markdown(f"### {label}")
        records = []
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            records.append({
                "File": f.name,
                "Format": f.suffix.upper().lstrip("."),
                "Size": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB",
            })

        if records:
            st.dataframe(
                __import__("pandas").DataFrame(records),
                width='stretch',
                hide_index=True,
            )


def _render_preview():
    """Preview dataset contents."""
    st.subheader("Data Preview")

    # Collect all available datasets
    all_datasets = []
    for d in ["data/external", "data/parquet", "data/synthetic"]:
        dir_path = _ROOT / d
        if dir_path.exists():
            for f in dir_path.glob("*.*"):
                if f.suffix in (".parquet", ".csv"):
                    all_datasets.append(f.stem)

    all_datasets = sorted(set(all_datasets))

    if not all_datasets:
        st.warning("No datasets found. Generate data first.")
        return

    selected = st.selectbox("Select Dataset", all_datasets)

    if selected:
        df = load_parquet(selected)
        if df is not None:
            st.markdown(f"**Shape:** {df.height} rows × {df.width} columns")
            st.markdown(f"**Columns:** {', '.join(df.columns)}")

            # Show dtypes
            dtype_info = {col: str(df[col].dtype) for col in df.columns}
            st.markdown(f"**Data Types:** {dtype_info}")

            # Preview
            n_rows = st.slider("Rows to display", 5, 100, 20)
            st.dataframe(df.head(n_rows).to_pandas(), width='stretch', hide_index=True)

            # Column statistics
            if st.checkbox("Show Column Statistics"):
                desc = df.describe().to_pandas()
                st.dataframe(desc, width='stretch')


def _render_quality():
    """Data quality report."""
    st.subheader("Data Quality Report")

    all_datasets = []
    for d in ["data/external", "data/parquet", "data/synthetic"]:
        dir_path = _ROOT / d
        if dir_path.exists():
            for f in dir_path.glob("*.*"):
                if f.suffix in (".parquet", ".csv"):
                    all_datasets.append(f.stem)

    all_datasets = sorted(set(all_datasets))

    quality_records = []
    for name in all_datasets:
        df = load_parquet(name)
        if df is None:
            continue

        null_count = sum(df[col].null_count() for col in df.columns)
        total_cells = df.height * df.width
        null_pct = null_count / total_cells * 100 if total_cells else 0

        # Check for date column and range
        date_range = "N/A"
        if "date" in df.columns:
            try:
                dates = df["date"].drop_nulls()
                if dates.len() > 0:
                    date_range = f"{dates.min()} → {dates.max()}"
            except Exception:
                pass

        quality_records.append({
            "Dataset": name,
            "Rows": df.height,
            "Columns": df.width,
            "Null Cells": null_count,
            "Null %": f"{null_pct:.1f}%",
            "Date Range": date_range,
        })

    if quality_records:
        st.dataframe(
            __import__("pandas").DataFrame(quality_records),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No datasets available for quality analysis")
