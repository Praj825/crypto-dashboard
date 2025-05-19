import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_BASE_URL = "https://8eaa-35-189-181-155.ngrok-free.app"  # Replace this if ngrok URL changes

st.title("📈 Cryptocurrency Price Forecast Dashboard")

def get_latest_price():
    try:
        response = requests.get(f"{API_BASE_URL}/price")
        return response.json().get("latest_price", None)
    except:
        return None

def get_forecast():
    try:
        response = requests.get(f"{API_BASE_URL}/predict")
        data = response.json()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

def get_actual_vs_predicted():
    try:
        response = requests.get(f"{API_BASE_URL}/actual_vs_predicted")
        data = response.json()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

latest_price = get_latest_price()
if latest_price:
    st.metric("Latest BTC Price (USD)", f"${latest_price:,.2f}")
else:
    st.warning("Could not fetch latest price.")

forecast_df = get_forecast()
if not forecast_df.empty:
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title='30-Day BTC Price Forecast', xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig)
else:
    st.warning("Could not fetch forecast data.")

actual_pred_df = get_actual_vs_predicted()
if not actual_pred_df.empty:
    actual_pred_df['ds'] = pd.to_datetime(actual_pred_df['ds'])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=actual_pred_df['ds'], y=actual_pred_df['y'], mode='lines+markers', name='Actual'))
    fig2.add_trace(go.Scatter(x=actual_pred_df['ds'], y=actual_pred_df['yhat'], mode='lines+markers', name='Predicted'))
    fig2.update_layout(title='Actual vs Predicted BTC Prices', xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig2)
else:
    st.warning("Could not fetch actual vs predicted data.")
