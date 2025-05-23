import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

st.title("📈 Cryptocurrency Forecast Dashboard")

@st.cache_data
def load_data():
    data = yf.download("BTC-USD", start="2020-01-01")
    if "Close" not in data.columns:
        st.error("❌ 'Close' column not found in data from yFinance.")
        st.stop()
    df = data.reset_index()[["Date", "Close"]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    return df

# Load data
df = load_data()
st.subheader("Latest BTC-USD Data")
st.dataframe(df.tail())

# Prophet Forecast
if st.button("Run Prophet Forecast"):
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast = model_prophet.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.update_layout(title="Prophet 30-day Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# ARIMA Forecast
if st.button("Run ARIMA Forecast"):
    model_arima = ARIMA(df['y'], order=(5, 1, 0))
    fit_arima = model_arima.fit()
    forecast_arima = fit_arima.forecast(steps=30)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df['y'], name='Actual'))
    fig2.add_trace(go.Scatter(y=forecast_arima, name='Forecast'))
    fig2.update_layout(title="ARIMA 30-day Forecast", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig2)

# SARIMA Forecast
if st.button("Run SARIMA Forecast"):
    model_sarima = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit_sarima = model_sarima.fit(disp=False)
    forecast_sarima = fit_sarima.forecast(steps=30)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=df['y'], name='Actual'))
    fig3.add_trace(go.Scatter(y=forecast_sarima, name='Forecast'))
    fig3.update_layout(title="SARIMA 30-day Forecast", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig3)

# LSTM Forecast (lightweight)
if st.button("Run LSTM Forecast"):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))

    X, y_lstm = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y_lstm.append(scaled_data[i, 0])
    X, y_lstm = np.array(X), np.array(y_lstm)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y_lstm, epochs=3, batch_size=32, verbose=0)

    inputs = scaled_data[-60:]
    lstm_predictions = []
    for _ in range(30):
        pred = model.predict(inputs.reshape(1, 60, 1), verbose=0)[0][0]
        lstm_predictions.append(pred)
        inputs = np.append(inputs[1:], pred).reshape(60, 1)

    forecast_lstm = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=df['y'], name='Actual'))
    fig4.add_trace(go.Scatter(y=forecast_lstm.flatten(), name='Forecast'))
    fig4.update_layout(title="LSTM 30-day Forecast", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig4)
