import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

st.title("📈 Cryptocurrency Forecast Dashboard (Prophet | ARIMA | SARIMA | LSTM)")

@st.cache_data

def load_data():
    df = yf.download("BTC-USD", start="2020-01-01")
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds', 'y'])
    return df

with st.spinner("Loading BTC-USD data..."):
    df = load_data()

st.success("Data loaded successfully!")

# Forecast with Prophet
st.header("Prophet Forecast")
prophet_df = df[['ds', 'y']].copy()
prophet = Prophet(daily_seasonality=True)
prophet.fit(prophet_df)
future = prophet.make_future_dataframe(periods=30)
forecast_prophet = prophet.predict(future)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
fig1.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], mode='lines', name='Forecast'))
fig1.update_layout(title="Prophet 30-day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig1, use_container_width=True)

# Forecast with ARIMA
st.header("ARIMA Forecast")
df_arima = df.set_index("ds")['y']
model_arima = ARIMA(df_arima, order=(5, 1, 0))
results_arima = model_arima.fit()
forecast_arima = results_arima.forecast(steps=30)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
future_dates = pd.date_range(df['ds'].iloc[-1], periods=31, freq='D')[1:]
fig2.add_trace(go.Scatter(x=future_dates, y=forecast_arima, mode='lines', name='Forecast'))
fig2.update_layout(title="ARIMA 30-day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig2, use_container_width=True)

# Forecast with SARIMA
st.header("SARIMA Forecast")
model_sarima = SARIMAX(df_arima, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_sarima = model_sarima.fit(disp=False)
forecast_sarima = results_sarima.forecast(steps=30)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
fig3.add_trace(go.Scatter(x=future_dates, y=forecast_sarima, mode='lines', name='Forecast'))
fig3.update_layout(title="SARIMA 30-day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig3, use_container_width=True)

# Forecast with LSTM
st.header("LSTM Forecast")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_arima.values.reshape(-1, 1))

train_size = int(len(data_scaled) * 0.80)
train, test = data_scaled[:train_size], data_scaled[train_size:]

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)

inputs = data_scaled[-look_back:].reshape(1, look_back, 1)
forecast_lstm_scaled = []
for _ in range(30):
    pred = model_lstm.predict(inputs)[0][0]
    forecast_lstm_scaled.append(pred)
    inputs = np.append(inputs[:, 1:, :], [[[pred]]], axis=1)
forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm_scaled).reshape(-1, 1)).flatten()

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
fig4.add_trace(go.Scatter(x=future_dates, y=forecast_lstm, mode='lines', name='Forecast'))
fig4.update_layout(title="LSTM 30-day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig4, use_container_width=True)
