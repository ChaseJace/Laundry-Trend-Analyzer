import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans


file_path = "Processed/laundrybook.xlsx"
df = pd.read_excel(file_path)

month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
df['Month'] = df['Month'].map(month_mapping)

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')

df = df.dropna(subset=['Date'])

daily_transactions = df.groupby('Date').size().reset_index(name='Count')

# ---------------------- Prophet Model for Forecasting ----------------------
daily_transactions.rename(columns={'Date': 'ds', 'Count': 'y'}, inplace=True)

split_date = daily_transactions['ds'].max() - pd.Timedelta(days=30)
train_data = daily_transactions[daily_transactions['ds'] <= split_date]
test_data = daily_transactions[daily_transactions['ds'] > split_date]

prophet_model = Prophet()
prophet_model.fit(train_data)

future_test = test_data[['ds']]
forecast_test = prophet_model.predict(future_test)

predicted_values = forecast_test['yhat'].values
actual_values = test_data['y'].values
mae_prophet = mean_absolute_error(actual_values, predicted_values)

print(f"Prophet MAE: {mae_prophet:.2f}")

future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

prophet_model.plot(forecast)
plt.title("Forecast of Laundry Transactions Per Day (Prophet)")
plt.xlabel("Date")
plt.ylabel("Number of Transactions")
plt.show()

# ----------------------  Random Forest for Prediction ----------------------
daily_transactions['ds_ordinal'] = daily_transactions['ds'].map(lambda x: x.toordinal())
X = daily_transactions[['ds_ordinal']]
y = daily_transactions['y']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

future_dates = future[['ds']]
future_dates['ds_ordinal'] = future_dates['ds'].map(lambda x: x.toordinal())
rf_predictions = rf_model.predict(future_dates[['ds_ordinal']])

y_pred = rf_model.predict(X)
mae = mean_absolute_error(y, y_pred)
print(f"Random Forest MAE: {mae:.2f}")

rf_forecast_df = pd.DataFrame({'Date': future['ds'], 'RF_Predictions': rf_predictions})
print(rf_forecast_df.tail(10))

# ---------------------- K-Means Clustering ----------------------
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
daily_transactions['Cluster'] = kmeans.fit_predict(daily_transactions[['y']])

print(daily_transactions[['ds', 'y', 'Cluster']].head(10))

plt.scatter(daily_transactions['ds'], daily_transactions['y'], c=daily_transactions['Cluster'], cmap='viridis')
plt.title("K-Means Clustering of Transactions")
plt.xlabel("Date")
plt.ylabel("Number of Transactions")
plt.colorbar(label="Cluster")
plt.show()
