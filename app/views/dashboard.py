# app/views/dashboard.py
import sys
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

# -------------------------------------------------
# PATH FIX (so utils/ works always)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.config import get_snowflake_session

load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
ENSEMBLE_CSV = ROOT / "models" / "ensemble" / "ensemble_next_30.csv"
MAX_YEARS = 5

st.set_page_config(
    page_title="Gold Price Forecast — Ensemble",
    layout="wide"
)

# -------------------------------------------------
# SAFE FX CONVERSION (USD → INR)
# -------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60)
def get_usd_inr_rate():
    try:
        url = "https://api.frankfurter.app/latest?from=USD&to=INR"
        r = requests.get(url, timeout=10)
        data = r.json()
        return float(data["rates"]["INR"])
    except Exception:
        st.warning("⚠️ FX API unavailable. Using fallback rate ₹83.0")
        return 83.0


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_actuals():
    session = get_snowflake_session()
    df = session.table("GOLD_PROJECT.PROCESSED.MASTER_GOLD_DATA").to_pandas()
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE")


@st.cache_data(ttl=3600)
def load_ensemble_30():
    df = pd.read_csv(ENSEMBLE_CSV, parse_dates=["date"])
    return df.set_index("date")


# -------------------------------------------------
# RECURSIVE FORECAST (key logic)
# -------------------------------------------------
def recursive_forecast(last_series, base_preds, horizon_days):
    """
    Extends ensemble predictions beyond 30 days using drift.
    """
    preds = list(base_preds)

    if horizon_days <= len(preds):
        return preds[:horizon_days]

    # Average daily drift from last 14 days
    drift = np.mean(np.diff(preds[-14:]))

    for _ in range(horizon_days - len(preds)):
        preds.append(preds[-1] + drift)

    return preds


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Gold Price Forecast — Ensemble (Chronos T5 + NHITS)")
st.caption("Investor-grade long-horizon forecasting (up to 5 years)")

# Controls
with st.sidebar:
    st.header("Forecast Horizon")

    mode = st.radio("Mode", ["Preset", "Custom"])

    if mode == "Preset":
        preset = st.selectbox(
            "Select horizon",
            ["Next Day", "Next Week (7)", "Next Month (30)", "1 Year", "5 Years"]
        )
    else:
        years = st.slider("Years", 0, MAX_YEARS, 1)
        months = st.slider("Months", 0, 11, 0)
        days = st.slider("Days", 0, 30, 0)

    st.markdown("**Currency:** INR (₹)")


# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------
with st.spinner("Loading data & running ensemble forecast…"):
    actuals = load_actuals()
    ensemble_30 = load_ensemble_30()
    usd_inr = get_usd_inr_rate()

# Convert to INR
actuals["GOLD_CLOSE"] *= usd_inr
ensemble_30.iloc[:, 0] *= usd_inr

# -------------------------------------------------
# DETERMINE HORIZON
# -------------------------------------------------
if mode == "Preset":
    horizon_days = {
        "Next Day": 1,
        "Next Week (7)": 7,
        "Next Month (30)": 30,
        "1 Year": 365,
        "5 Years": 365 * 5
    }[preset]
else:
    horizon_days = years * 365 + months * 30 + days
    horizon_days = max(1, min(horizon_days, 365 * MAX_YEARS))

# -------------------------------------------------
# RUN FORECAST
# -------------------------------------------------
future_preds = recursive_forecast(
    actuals["GOLD_CLOSE"],
    ensemble_30.iloc[:, 0].tolist(),
    horizon_days
)

future_dates = pd.date_range(
    start=actuals.index[-1] + timedelta(days=1),
    periods=horizon_days
)

forecast_df = pd.DataFrame(
    {"Prediction (₹)": future_preds},
    index=future_dates
)

# -------------------------------------------------
# STATUS
# -------------------------------------------------
st.success(f"Loaded historical data: {len(actuals)} rows (last date: {actuals.index[-1].date()})")
st.success(f"Forecast horizon: {horizon_days} days")

# -------------------------------------------------
# PLOT
# -------------------------------------------------
st.header("Historical & Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=actuals.index[-730:],
    y=actuals["GOLD_CLOSE"].iloc[-730:],
    name="Historical Actuals",
    line=dict(color="royalblue")
))

fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["Prediction (₹)"],
    name="Ensemble Forecast",
    mode="lines+markers",
    line=dict(color="orange")
))

fig.update_layout(
    height=550,
    xaxis_title="Date",
    yaxis_title="Gold Price (₹)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------
st.header("Predictions")

st.metric(
    "Next Day Prediction (₹)",
    f"₹ {forecast_df.iloc[0, 0]:,.2f}"
)

if horizon_days >= 7:
    st.subheader("Next 7 Days")
    st.table(forecast_df.head(7))

st.subheader("Forecast Curve")
st.line_chart(forecast_df["Prediction (₹)"])

if st.checkbox("Show full forecast table"):
    st.dataframe(forecast_df)

# Download
st.download_button(
    "Download Forecast CSV",
    forecast_df.to_csv().encode(),
    "gold_forecast_inr.csv",
    "text/csv"
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("All values shown in INR (₹). Forecast uses ensemble recursive extension beyond 30 days.")
