import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from statsmodels.tsa.seasonal import seasonal_decompose

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📈 Stock Price Forecasting using LSTM")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Input Settings")

stock = st.sidebar.text_input("Stock Symbol", "POWERGRID.NS")
start_date = st.sidebar.date_input("Start Date", dt.date(2000, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

run_btn = st.sidebar.button("Run Forecast")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("powergrid_lstm_model.h5")

model = load_lstm_model()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(stock, start, end):
    df = yf.download(stock, start, end)
    return df[['Close']]

df = load_data(stock, start_date, end_date)

# --------------------------------------------------
# EDA SECTION
# --------------------------------------------------
st.header("🔍 Exploratory Data Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Start Date", df.index.min().date())
col3.metric("End Date", df.index.max().date())

# Closing price trend
st.subheader("📈 Closing Price Trend")
st.line_chart(df['Close'])

# Rolling statistics
st.subheader("📉 Rolling Mean & Volatility")
df['Rolling_Mean'] = df['Close'].rolling(30).mean()
df['Rolling_STD'] = df['Close'].rolling(30).std()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['Close'], label="Close", alpha=0.6)
ax.plot(df['Rolling_Mean'], label="Rolling Mean")
ax.plot(df['Rolling_STD'], label="Rolling Std")
ax.legend()
ax.grid()
st.pyplot(fig)

# Daily returns
st.subheader("📊 Daily Returns")
df['Daily_Return'] = df['Close'].pct_change()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['Daily_Return'])
ax.axhline(0, color='black')
ax.grid()
st.pyplot(fig)

# Returns distribution
st.subheader("📦 Distribution of Daily Returns")
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(df['Daily_Return'].dropna(), bins=50)
ax.grid()
st.pyplot(fig)

# --------------------------------------------------
# FORECAST SECTION
# --------------------------------------------------
if run_btn:
    st.header("🔮 LSTM Forecast – Next 30 Business Days")

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']])

    last_60_days = scaled_data[-60:]
    future_input = last_60_days.reshape(1,60,1)

    future_prices = []

    for _ in range(30):
        next_price = model.predict(future_input, verbose=0)
        future_prices.append(next_price[0][0])

        future_input = np.append(
            future_input[:,1:,:],
            next_price.reshape(1,1,1),
            axis=1
        )

    future_prices = scaler.inverse_transform(
        np.array(future_prices).reshape(-1,1)
    )

    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=30,
        freq='B'
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close_Price": future_prices.flatten()
    }).set_index("Date")

    # --------------------------------------------------
    # VISUAL FORECAST
    # --------------------------------------------------
    st.subheader("📈 Visual Forecast")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index[-120:], df['Close'].tail(120), label="Historical")
    ax.plot(forecast_df.index, forecast_df['Predicted_Close_Price'],
            linestyle="--", marker="o", label="Forecast")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # --------------------------------------------------
    # DATE-WISE PREDICTION
    # --------------------------------------------------
    st.subheader("📅 Predict for a Specific Date")

    selected_date = st.date_input(
        "Choose a date",
        min_value=future_dates.min().date(),
        max_value=future_dates.max().date()
    )

    selected_date = pd.to_datetime(selected_date)

    if selected_date in forecast_df.index:
        price = forecast_df.loc[selected_date, 'Predicted_Close_Price']
        st.success(f"Predicted price on {selected_date.date()} : ₹{price:.2f}")
    else:
        st.error("Date not in forecast range")

    # --------------------------------------------------
    # FORECAST TABLE & DOWNLOAD
    # --------------------------------------------------
    st.subheader("📊 Forecast Table")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="stock_forecast_30_days.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("⚠️ *Educational purpose only. Not financial advice.*")
