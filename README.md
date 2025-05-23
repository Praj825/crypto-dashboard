# 📊 Cryptocurrency Forecast Dashboard

This project forecasts Bitcoin prices using four time series models:
- Facebook Prophet
- ARIMA
- SARIMA
- LSTM (deep learning)

An interactive dashboard is built with **Streamlit** to display predictions.

---

## 🔧 Tech Stack
- Python, Pandas, NumPy
- YFinance API for data
- Prophet, Statsmodels, TensorFlow, Keras
- Streamlit, Plotly for visualization

---

## 📁 Project Structure
- `dashboard.py` – Streamlit dashboard app
- `TSA for crypto.ipynb` – Model development notebook (not needed for deployment)
- `requirements.txt` – Dependencies for Streamlit Cloud

---

## 🚀 How to Run

### 🔹 On Streamlit Cloud
1. Fork the repo
2. Deploy on [Streamlit](https://share.streamlit.io)
3. Set the app file as: `dashboard.py`

### 🔹 Locally
```bash
pip install -r requirements.txt
streamlit run dashboard.py
