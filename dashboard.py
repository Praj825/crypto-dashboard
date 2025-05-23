import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

st.set_page_config(page_title="Crypto Forecast", layout="wide")
st.title("📈 Cryptocurrency Forecast Dashboard (Prophet | ARIMA | SARIMA | LSTM)")

@st.cache_data
def load_data():
    data = yf.download("BTC-USD", start="2020-01-01")
    df = data.reset_index()
    df = df[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    return df

df = load_data()
st.subheader("📊 Latest BTC-USD Data")
st.dataframe(df.tail())

# Prophet Forecast
st.subheader("🔮 Prophet Forecast")
prophet_df = df[["ds", "y"]].copy()
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual"))
fig1.add_trace(go.Scatter(x=forecast_prophet["ds"], y=forecast_prophet["yhat"], name="Forecast"))
fig1.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig1)

# ARIMA Forecast
st.subheader("📉 ARIMA Forecast")
model_arima = ARIMA(df["y"], order=(5, 1, 0))
fit_arima = model_arima.fit()
forecast_arima = fit_arima.forecast(steps=30)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=df["y"], name="Actual"))
fig2.add_trace(go.Scatter(y=forecast_arima, name="Forecast"))
fig2.update_layout(title="ARIMA Forecast", xaxis_title="Index", yaxis_title="Price (USD)")
st.plotly_chart(fig2)

# SARIMA Forecast
st.subheader("🌀 SARIMA Forecast")
model_sarima = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fit_sarima = model_sarima.fit(disp=False)
forecast_sarima = fit_sarima.forecast(steps=30)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=df["y"], name="Actual"))
fig3.add_trace(go.Scatter(y=forecast_sarima, name="Forecast"))
fig3.update_layout(title="SARIMA Forecast", xaxis_title="Index", yaxis_title="Price (USD)")
st.plotly_chart(fig3)

# LSTM Forecast
st.subheader("🤖 LSTM Forecast")

scaler = MinMaxScaler()
scaled_y = scaler.fit_transform(df["y"].values.reshape(-1, 1))

X, y_lstm = [], []
for i in range(60, len(scaled_y)):
    X.append(scaled_y[i-60:i])
    y_lstm.append(scaled_y[i])

X, y_lstm = np.array(X), np.array(y_lstm)
X = X.reshape((X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X, y_lstm, epochs=5, batch_size=32, verbose=0)

inputs = scaled_y[-60:]
lstm_preds = []
for _ in range(30):
    input_reshaped = inputs.reshape(1, 60, 1)
    pred = model_lstm.predict(input_reshaped, verbose=0)
    lstm_preds.append(pred[0, 0])
    inputs = np.append(inputs[1:], pred)

forecast_lstm = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))

fig4 = go.Figure()
fig4.add_trace(go.Scatter(y=df["y"], name="Actual"))
fig4.add_trace(go.Scatter(y=forecast_lstm.flatten(), name="Forecast"))
fig4.update_layout(title="LSTM Forecast", xaxis_title="Index", yaxis_title="Price (USD)")
st.plotly_chart(fig4)

st.success("✅ All forecasts generated successfully!")
