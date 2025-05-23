import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Cryptocurrency Forecast Dashboard (Prophet | ARIMA | SARIMA | LSTM)")

# Load data
st.subheader("Loading BTC-USD data...")
df = yf.download("BTC-USD", start="2021-01-01", end="2024-12-31")
df = df[["Close"]].dropna()
df = df.rename(columns={"Close": "y"})
df["ds"] = df.index
df.reset_index(drop=True, inplace=True)

# Prophet
with st.spinner("Training Prophet model..."):
    prophet_df = df[["ds", "y"]].copy()
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(prophet_df)
    future = prophet.make_future_dataframe(periods=30)
    forecast_prophet = prophet.predict(future)

# ARIMA
with st.spinner("Training ARIMA model..."):
    arima_model = ARIMA(df["y"], order=(5,1,0))
    arima_fit = arima_model.fit()
    forecast_arima = arima_fit.forecast(steps=30)

# SARIMA
with st.spinner("Training SARIMA model..."):
    sarima_model = SARIMAX(df["y"], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit()
    forecast_sarima = sarima_fit.forecast(steps=30)

# LSTM
with st.spinner("Training LSTM model..."):
    sequence_length = 60
    data = df[["y"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    x_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    test_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    lstm_predictions = []

    for _ in range(30):
        next_price = model.predict(test_input, verbose=0)[0][0]
        lstm_predictions.append(next_price)
        next_seq = np.append(test_input[0][1:], [[next_price]], axis=0)
        test_input = next_seq.reshape(1, sequence_length, 1)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
    lstm_forecast_dates = df["ds"].iloc[-1] + pd.to_timedelta(np.arange(1, 31), unit='D')

# Final Plot
st.subheader("Forecast Comparison (Next 30 Days)")
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(df["ds"], df["y"], label="Actual", color="black")
ax.plot(forecast_prophet["ds"], forecast_prophet["yhat"], label="Prophet", linestyle="--")
ax.plot(df["ds"].iloc[-1] + pd.to_timedelta(range(1,31), unit="D"), forecast_arima, label="ARIMA")
ax.plot(df["ds"].iloc[-1] + pd.to_timedelta(range(1,31), unit="D"), forecast_sarima, label="SARIMA")
ax.plot(lstm_forecast_dates, lstm_predictions, label="LSTM")
ax.set_title("BTC Forecast: Prophet vs ARIMA vs SARIMA vs LSTM")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.grid(True)
ax.legend()
st.pyplot(fig)
