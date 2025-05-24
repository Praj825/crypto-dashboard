import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")
st.title("📈 Cryptocurrency Forecast Dashboard")

# Sidebar Controls
st.sidebar.header("Controls")
selected_coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "ADA-USD"])
forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)

@st.cache_data

def load_data(symbol):
    data = yf.download(symbol, start="2020-01-01")
    if "Close" not in data.columns:
        st.error("❌ 'Close' column not found in data from yFinance.")
        st.stop()
    df = data.reset_index()[["Date", "Close"]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    return df

def print_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    st.write(f"**MAE:** {mae:.2f} | **MSE:** {mse:.2f} | **RMSE:** {rmse:.2f}")

# Load data
df = load_data(selected_coin)
st.subheader(f"Latest {selected_coin} Data")
st.dataframe(df.tail())

# Prophet Forecast
if st.button("Run Prophet Forecast"):
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df)
    future = model_prophet.make_future_dataframe(periods=forecast_days)
    forecast = model_prophet.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    print_metrics(df['y'][-forecast_days:], forecast['yhat'][-forecast_days:])

# ARIMA Forecast
if st.button("Run ARIMA Forecast"):
    model_arima = ARIMA(df['y'], order=(5, 1, 0))
    fit_arima = model_arima.fit()
    forecast_arima = fit_arima.forecast(steps=forecast_days)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df['y'], name='Actual'))
    fig2.add_trace(go.Scatter(y=forecast_arima, name='Forecast'))
    fig2.update_layout(title="ARIMA Forecast", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig2)

    print_metrics(df['y'][-forecast_days:], forecast_arima)

# SARIMA Forecast
if st.button("Run SARIMA Forecast"):
    model_sarima = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit_sarima = model_sarima.fit(disp=False)
    forecast_sarima = fit_sarima.forecast(steps=forecast_days)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=df['y'], name='Actual'))
    fig3.add_trace(go.Scatter(y=forecast_sarima, name='Forecast'))
    fig3.update_layout(title="SARIMA Forecast", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig3)

    print_metrics(df['y'][-forecast_days:], forecast_sarima)

# LSTM Forecast
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
    for _ in range(forecast_days):
        pred = model.predict(inputs.reshape(1, 60, 1), verbose=0)[0][0]
        lstm_predictions.append(pred)
        inputs = np.append(inputs[1:], pred).reshape(60, 1)

    forecast_lstm = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=df['y'], name='Actual'))
    fig4.add_trace(go.Scatter(y=forecast_lstm.flatten(), name='Forecast'))
    fig4.update_layout(title="LSTM Forecast", xaxis_title="Index", yaxis_title="Price")
    st.plotly_chart(fig4)

    print_metrics(df['y'][-forecast_days:], forecast_lstm.flatten())
