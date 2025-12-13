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
# PATH FIX
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.config import get_snowflake_session

load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MAX_YEARS = 5

st.set_page_config(
    page_title="Gold Price Forecast — Ensemble",
    layout="wide"
)

# -------------------------------------------------
# FX RATE (USD → INR)
# -------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60)
def get_usd_inr_rate():
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=INR",
            timeout=10
        )
        return float(r.json()["rates"]["INR"])
    except Exception:
        st.warning("⚠️ FX API unavailable. Using fallback INR rate.")
        return 83.0


# -------------------------------------------------
# LOAD DATA FROM SNOWFLAKE (READ-ONLY)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_actuals():
    session = get_snowflake_session()
    df = session.table(
        "GOLD_PROJECT.PROCESSED.MASTER_GOLD_DATA"
    ).to_pandas()
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE")


@st.cache_data(ttl=3600)
def load_ensemble_forecast():
    session = get_snowflake_session()
    df = session.table(
        "GOLD_PROJECT.PROCESSED.ENSEMBLE_FORECAST"
    ).to_pandas()
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE")


# -------------------------------------------------
# SAFE RECURSIVE EXTENSION (LONG HORIZON)
# -------------------------------------------------
def recursive_forecast(base_preds, horizon_days, fallback_value):
    """
    Production-safe recursive forecast.

    - If ensemble predictions are empty:
        → fallback to last known gold price
    - If only few predictions exist:
        → zero/low drift extension
    """

    # ---- Case 1: No ensemble data ----
    if base_preds is None or len(base_preds) == 0:
        st.warning(
            "⚠️ Ensemble forecast not available yet. "
            "Using last known gold price as flat forecast."
        )
        return [fallback_value] * horizon_days

    preds = list(base_preds)

    # ---- Case 2: Enough predictions ----
    if horizon_days <= len(preds):
        return preds[:horizon_days]

    # ---- Case 3: Extend recursively ----
    if len(preds) < 2:
        drift = 0.0
    else:
        drift = np.mean(np.diff(preds[-min(14, len(preds)):]))
        if np.isnan(drift):
            drift = 0.0

    for _ in range(horizon_days - len(preds)):
        preds.append(preds[-1] + drift)

    return preds


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Gold Price Forecast — Ensemble")
st.caption("Chronos T5 + NHITS • Investor-grade forecasting")

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


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
with st.spinner("Loading data…"):
    actuals = load_actuals()
    ensemble = load_ensemble_forecast()
    usd_inr = get_usd_inr_rate()

# Convert to INR
actuals["GOLD_CLOSE"] *= usd_inr

if "ENSEMBLE_PRED" in ensemble.columns:
    ensemble["ENSEMBLE_PRED"] *= usd_inr
else:
    ensemble["ENSEMBLE_PRED"] = []


# -------------------------------------------------
# HORIZON
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
    horizon_days = max(
        1,
        min(years * 365 + months * 30 + days, 365 * MAX_YEARS)
    )


# -------------------------------------------------
# FORECAST
# -------------------------------------------------
future_preds = recursive_forecast(
    ensemble["ENSEMBLE_PRED"].tolist(),
    horizon_days,
    fallback_value=actuals["GOLD_CLOSE"].iloc[-1]
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
# PLOT
# -------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=actuals.index[-730:],
    y=actuals["GOLD_CLOSE"].iloc[-730:],
    name="Historical",
    line=dict(color="royalblue")
))

fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["Prediction (₹)"],
    name="Ensemble Forecast",
    line=dict(color="orange")
))

fig.update_layout(
    height=550,
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Gold Price (₹)"
)

st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
st.metric(
    "Next Day Prediction (₹)",
    f"₹ {forecast_df.iloc[0, 0]:,.2f}"
)

st.line_chart(forecast_df["Prediction (₹)"])

st.download_button(
    "Download Forecast CSV",
    forecast_df.to_csv().encode(),
    "gold_forecast.csv",
    "text/csv"
)

st.caption(
    "Dashboard is read-only. "
    "Models (Chronos T5 + NHITS) run outside the UI via scheduled pipelines."
)
