# 📈 Stock Price Forecasting using LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

🔗 Live Demo

👉 https://stockmaertprediction.vercel.app/
---

## 🚀 Project Overview

This project implements an **end-to-end stock price forecasting system** using **Long Short-Term Memory (LSTM)** neural networks.  
It covers the full machine learning workflow:

**EDA → Model Training → Evaluation → Forecasting → Visualization → Deployment**

The model predicts the **next 30 business days** of stock prices and is deployable via **Streamlit** or integrable with **low-code / no-code frontends** such as Retool, Lovable.dev, or Bubble.

---

## 🎯 Objectives

- Perform time-series **Exploratory Data Analysis (EDA)**
- Analyze **trend, volatility, seasonality**
- Build an **LSTM-based forecasting model**
- Evaluate using **MAE, RMSE, MAPE**
- Generate **date-wise future forecasts**
- Visualize historical and predicted prices
- Enable deployment-ready architecture

---

## 🗂️ Dataset

| Attribute | Description |
|---------|-------------|
| Source | Yahoo Finance |
| Stock Example | POWERGRID.NS |
| Time Range | 2000 – 2026 |
| Frequency | Daily (Business Days) |
| Target Variable | Closing Price |

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- 📈 Closing Price Trend (long-term behavior)
- 📉 Rolling Mean & Rolling Standard Deviation (trend + volatility)
- 📊 Daily Returns Analysis (risk & volatility clustering)
- 📦 Distribution of Returns (heavy tails, non-normality)
- 🔁 Moving Averages (technical trend insight)
- 🔄 Seasonal Decomposition (trend, seasonality, residuals)
- 📐 ACF & PACF plots (temporal dependency)
- 🧪 ADF Test (confirmed non-stationarity)

> These insights justify the use of LSTM, which does **not require stationarity**.

---

## 🧠 Methodology

1. Data collection using `yfinance`
2. Scaling with `MinMaxScaler`
3. Creation of 60-day rolling sequences
4. Training a stacked LSTM model with dropout
5. Model evaluation on unseen test data
6. Recursive multi-step forecasting
7. Visualization and deployment

---

## 🧱 Model Architecture

Input (60 days)
↓
LSTM (50 units, return_sequences=True)
↓
Dropout (0.2)
↓
LSTM (50 units)
↓
Dropout (0.2)
↓
Dense (1)

---

## 📊 Model Evaluation

| Metric | Result |
|------|--------|
| MAE | ~ ₹4 |
| MAPE | ~ 1.99% |
| RMSE | Low relative to price range |

A ~2% MAPE indicates strong performance for financial time-series forecasting.

---

## 🔮 Forecasting

- Generated **next 30 business days** predictions
- Date-wise forecasting
- Historical vs future visualization
- Exported predictions as CSV

---

## 🌐 Deployment

### Streamlit App Features
- Interactive EDA
- Visual forecast chart
- Date-wise prediction
- CSV download option

The backend can also be exposed as an API for integration with:
- Retool
- Lovable.dev
- Bubble.io

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- scikit-learn
- statsmodels
- Streamlit
- Yahoo Finance API

---

## 📂 Project Structure

├── app.py
├── powergrid_lstm_model.h5
├── requirements.txt
├── notebooks/
│ └── Stock_market.ipynb
├── data/
└── README.md


---

## ⚠️ Limitations

- Uses only historical prices
- No news or sentiment data
- Forecasts are probabilistic, not guaranteed

---

## 🚀 Future Enhancements

- Add RSI, MACD, and volume features
- Compare LSTM with SARIMAX
- Hybrid statistical + deep learning model
- Confidence intervals for forecasts
- REST API for no-code frontends

---


