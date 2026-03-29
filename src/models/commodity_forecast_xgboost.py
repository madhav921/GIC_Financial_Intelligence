import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import yfinance as yf
from src.models.commodity_forecast import CommodityForecastModel
import matplotlib.pyplot as plt

# --- 1. Fetch Real Market Data ---
# Automotive Commodities (Tickers): 
# HG=F (Copper), ALI=F (Aluminum), CL=F (Crude Oil - for energy/plastic)
tickers = {
    "copper": "HG=F",
    "aluminum": "ALI=F",
    "energy": "CL=F"
}

print("Fetching real market data from Yahoo Finance...")
raw_data = yf.download(list(tickers.values()), start="2020-01-01", end="2026-01-01")

# --- 2. Clean & Resample for your Model ---
# Your model expects a column named after the commodity (e.g., 'copper') 
# and a 'date' column or index.
# yfinance >= 0.2 uses 'Close' instead of 'Adj Close'
price_col = 'Close' if 'Close' in raw_data.columns.get_level_values(0) else 'Adj Close'
df_real = raw_data[price_col].rename(columns={v: k for k, v in tickers.items()})

# Resample to Monthly Start (MS) as your model handles monthly granularity
df_real = df_real.resample("MS").last().ffill().dropna()
df_real.reset_index(inplace=True)
df_real.rename(columns={"Date": "date"}, inplace=True)

print(f"Data ready. Shape: {df_real.shape}")
print(df_real.head())

# # --- 3. Instantiate and Train your Model ---
model = CommodityForecastModel()

# # We'll test 'copper' because it's highly volatile and great for XGBoost
# target_commodity = "copper"

# print(f"\nTraining XGBoost on REAL {target_commodity} data...")
# # Note: Since you're local, ensure your src.config and src.data imports are reachable
# try:
#     model.train_xgboost(target_commodity, df_real, macro_df=None)
    
#     # --- 4. Generate Forecast ---
#     result = model.forecast_xgboost(target_commodity, df_real, macro_df=None)

#     # --- 5. Inspect Results ---
#     print("\n--- Forecast Results ---")
#     print(f"Model Type: {result.model_type}")
#     print(f"Horizon: {len(result.dates)} months")
#     print(f"First 3 Predicted Prices: {result.point_forecast[:3]}")
    
#     # --- 6. Quick Visualization ---
#     forecast_dates = pd.to_datetime(result.dates)  # convert strings to datetime for proper axis
#     plt.figure(figsize=(12, 5))
#     plt.plot(forecast_dates, result.point_forecast, label="Point Forecast", color='blue', linewidth=2)
#     plt.fill_between(forecast_dates, result.lower_80, result.upper_80, color='blue', alpha=0.2, label="80% CI")
#     plt.fill_between(forecast_dates, result.lower_95, result.upper_95, color='blue', alpha=0.1, label="95% CI")
#     plt.title(f"{target_commodity.capitalize()} Price Forecast (XGBoost) — {len(result.dates)}-month horizon")
#     plt.xlabel("Date")
#     plt.ylabel("Price (USD)")
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # --- 7. Validation summary ---
#     print("\n--- Validation Summary ---")
#     print(f"Forecast dates: {result.dates[0]} → {result.dates[-1]}")
#     print(f"Min / Max forecast: {min(result.point_forecast):.2f} / {max(result.point_forecast):.2f}")
#     print(f"80% CI width at month 1 : {result.upper_80[0] - result.lower_80[0]:.2f}")
#     print(f"80% CI width at month 12: {result.upper_80[-1] - result.lower_80[-1]:.2f}")
#     if result.feature_importance:
#         print("\nTop 5 features:")
#         for feat, imp in list(result.feature_importance.items())[:5]:
#             print(f"  {feat:<30s} {imp:.4f}")

# except Exception as e:
#     print(f"Error during training/forecast: {e}")
#     traceback.print_exc()

# --- 3. Split Data for Validation ---
# Training: 2020-2024 | Validation: 2025
df_train = df_real[df_real['date'] < "2025-01-01"].copy()
df_actual_2025 = df_real[(df_real['date'] >= "2025-01-01") & (df_real['date'] < "2026-01-01")].copy()

target_commodity = "copper"

try:
    # --- 4. Train on Historical Data (Pre-2025) ---
    print(f"\nTraining model on historical data (2020-2024)...")
    model.train_xgboost(target_commodity, df_train, macro_df=None)
   
    # --- 5. Generate Forecast for 2025 ---
    # Horizon should match the length of our validation set (12 months)
    model.horizon = 12
    result = model.forecast_xgboost(target_commodity, df_train, macro_df=None)

    # --- 6. Create Comparison Table ---
    # result.dates are strings 'YYYY-MM-DD', align them with actuals
    comparison_df = pd.DataFrame({
        "Date": result.dates,
        "Forecasted": result.point_forecast,
        "Actual": df_actual_2025[target_commodity].values[:12]
    })
   
    # Calculate Error Metrics
    comparison_df["Abs_Error"] = (comparison_df["Forecasted"] - comparison_df["Actual"]).abs()
    comparison_df["Error_Percent"] = (comparison_df["Abs_Error"] / comparison_df["Actual"]) * 100
   
    print("\n--- 2025 Validation Table ---")
    print(comparison_df.to_string(index=False))
   
    avg_mape = comparison_df["Error_Percent"].mean()
    print(f"\nAverage Forecast Error (MAPE) for 2025: {avg_mape:.2f}%")

    # --- 7. Advanced Visualization ---
    plt.figure(figsize=(12, 6))
   
    # Plot Historical Training Data
    plt.plot(df_train['date'], df_train[target_commodity], label="Historical (Train)", color='gray', alpha=0.5)
   
    # Plot Actual 2025 Data
    plt.plot(df_actual_2025['date'], df_actual_2025[target_commodity], label="Actual 2025", color='black', linewidth=2)
   
    # Plot Forecasted 2025 Data
    forecast_dates = pd.to_datetime(result.dates)
    plt.plot(forecast_dates, result.point_forecast, label="XGBoost Forecast", color='red', linestyle='--')
   
    # Confidence Intervals
    plt.fill_between(forecast_dates, result.lower_95, result.upper_95, color='red', alpha=0.1, label="95% CI")
   
    plt.title(f"Model Validation: {target_commodity.capitalize()} (2025 Forecast vs Actual)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except Exception as e:
    print(f"Validation Error: {e}")
    traceback.print_exc()