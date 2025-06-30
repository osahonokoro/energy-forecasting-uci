import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import holidays

# === STEP 1: Load Dataset ===
df = pd.read_csv(r"C:\Users\Otala\Desktop\Osahon\Project 2025\Benedict\Data_for_UCI_named.csv")

# === STEP 2: Simulate Timestamp Column ===
start_time = datetime(2025, 1, 1, 0, 0)  # arbitrary start time
df['ds'] = [start_time + timedelta(minutes=i) for i in range(len(df))]

# === STEP 3: Select Target Variable ===
# Using 'stab' (stability index) as forecasting target
df = df[['ds', 'stab']].rename(columns={'stab': 'y'})

# Optional: Resample to hourly granularity (averaging every 60 minutes)
df = df.set_index('ds').resample('H').mean().reset_index()
df.dropna(inplace=True)

# === STEP 4: Nigerian Holidays ===
ng_holidays = holidays.CountryHoliday('NG', years=[2025])
holiday_df = pd.DataFrame({
    'holiday': 'nigeria_holiday',
    'ds': pd.to_datetime(list(ng_holidays.keys()))
})

# === STEP 5: Train/Test Split ===
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# === STEP 6: Train Prophet Model ===
model = Prophet(holidays=holiday_df, daily_seasonality=True, weekly_seasonality=True)
model.fit(train_df)

# === STEP 7: Forecasting ===
future = model.make_future_dataframe(periods=len(test_df), freq='H')
forecast = model.predict(future)
forecast_df = forecast[['ds', 'yhat']].tail(len(test_df)).reset_index(drop=True)

# === STEP 8: Evaluation ===
actual = test_df['y'].values
predicted = forecast_df['yhat'].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = mean_absolute_percentage_error(actual, predicted) * 100

print(f"\nPerformance Metrics:")
print(f"MAE:  {mae:.3f} kWh")
print(f"RMSE: {rmse:.3f} kWh")
print(f"MAPE: {mape:.2f}%")

# === STEP 9: Visualization ===
plt.figure(figsize=(14, 5))
plt.plot(test_df['ds'], actual, label='Actual (stab)', color='black')
plt.plot(test_df['ds'], predicted, label='Prophet Forecast', color='blue')
plt.title("Figure 4.1 – Actual vs. Forecasted Stability Index (Prophet Model)")
plt.xlabel("Date")
plt.ylabel("Stability Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
