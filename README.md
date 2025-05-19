# 📈 Cryptocurrency Price Forecast Dashboard

This is an internship project that forecasts Bitcoin prices and visualizes them in an interactive dashboard using **Streamlit**.

---

## 🚀 Live App Link

🔗 [Click here to view the deployed dashboard](https://crypto-dashboard-4wqx5xfzmn9az8fnbwstir.streamlit.app)

---

## 📊 Features

- 📥 Downloads BTC-USD price data using `yfinance`
- 🔮 Trains a **Prophet** model for time series forecasting
- 📅 Predicts the next 30 days of prices
- 📉 Shows forecast and actual-vs-predicted plots
- 🌐 Hosted live using Streamlit Cloud

---

## 🧰 Tech Stack

| Tool       | Role                          |
|------------|-------------------------------|
| Python     | Programming language          |
| yfinance   | Fetch historical price data   |
| Prophet    | Forecasting model             |
| FastAPI    | REST API for backend          |
| Ngrok      | Public tunneling for FastAPI  |
| Streamlit  | Frontend web dashboard        |

---

## 📁 Files in this Project

| File             | Purpose                                  |
|------------------|------------------------------------------|
| `dashboard.py`   | Streamlit code to build the dashboard    |
| `requirements.txt` | Lists required Python packages         |
| `README.md`      | Describes the project and setup info     |

---

## 📦 How to Run This Project Locally

> *Only needed if you're cloning the repo — not for the deployed version.*

### 1. Clone the repository
```bash
git clone https://github.com/Praj825/crypto-dashboard.git
cd crypto-dashboard
