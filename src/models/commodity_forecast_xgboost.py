import pandas as pd
import numpy as np
from commodity_forecast import CommodityForecastModel

# --- 1. Prepare data: date index + commodity columns ---
dates = pd.date_range("2022-01-01", periods=36, freq="MS")
df = pd.DataFrame({
    "date": dates,
    "steel":    np.random.uniform(700, 900, 36),
    "aluminum": np.random.uniform(2000, 2600, 36),
    "copper":   np.random.uniform(8000, 10000, 36),
})

# --- 2. Instantiate model ---
model = CommodityForecastModel()

# --- 3. Train XGBoost for one commodity (no macro needed) ---
model.train_xgboost("steel", df, macro_df=None)

# --- 4. Forecast ---
result = model.forecast_xgboost("steel", df, macro_df=None)

# --- 5. Inspect output ---
print(result.dates)           # list of YYYY-MM-DD strings
print(result.point_forecast)  # list of floats
print(result.lower_95)        # confidence band lower
print(result.upper_95)        # confidence band upper