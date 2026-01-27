import sys
import sqlite3
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from dotenv import load_dotenv

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MAX_YEARS = 5
DB_PATH = ROOT / "database" / "gold_data.db"

st.set_page_config(
    page_title="Gold Price Forecast — Ensemble",
    layout="wide"
)

# -------------------------------------------------
# CONVERSION CONSTANTS
# -------------------------------------------------
TROY_OUNCE_TO_GRAMS = 31.1035  # 1 troy ounce = 31.1035 grams
DEFAULT_WEIGHT_GRAMS = 8.0

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
# CONVERSION FUNCTIONS
# -------------------------------------------------
def convert_usd_per_oz_to_inr_per_gram(usd_per_oz, usd_inr_rate, weight_grams=1.0):
    """
    Convert USD per troy ounce to INR per custom weight in grams
    """
    # USD per troy ounce → USD per gram
    usd_per_gram = usd_per_oz / TROY_OUNCE_TO_GRAMS
    
    # USD per gram → INR per gram
    inr_per_gram = usd_per_gram * usd_inr_rate
    
    # INR for specified weight
    return inr_per_gram * weight_grams

# -------------------------------------------------
# LOAD DATA FROM SQLITE (READ-ONLY)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_actuals():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT
            date,
            gold_close AS GOLD_CLOSE
        FROM features
        ORDER BY date
        """,
        conn
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


@st.cache_data(ttl=3600)
def load_ensemble_forecast():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT
            date,
            ensemble_pred AS ENSEMBLE_PRED
        FROM ensemble_forecast
        ORDER BY date
        """,
        conn
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


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

    if base_preds is None or len(base_preds) == 0:
        st.warning(
            "⚠️ Ensemble forecast not available yet. "
            "Using last known gold price as flat forecast."
        )
        return [fallback_value] * horizon_days

    preds = list(base_preds)

    if horizon_days <= len(preds):
        return preds[:horizon_days]

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
st.caption("Chronos-T5 + N-HiTS • Investor-grade forecasting")

with st.sidebar:
    st.header("Configuration")
    
    # Weight Customization
    st.subheader("Gold Weight")
    weight_grams = st.number_input(
        "Weight (grams)",
        min_value=0.1,
        max_value=1000.0,
        value=DEFAULT_WEIGHT_GRAMS,
        step=0.1,
        help="Enter the gold weight in grams to calculate the corresponding price"
    )
    st.caption(f"Displaying prices for **{weight_grams} grams** of gold")
    
    st.divider()
    
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

# Convert actuals to INR for specified weight
actuals["GOLD_CLOSE_CONVERTED"] = actuals["GOLD_CLOSE"].apply(
    lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
)

if "ENSEMBLE_PRED" in ensemble.columns:
    ensemble["ENSEMBLE_PRED_CONVERTED"] = ensemble["ENSEMBLE_PRED"].apply(
        lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
    )
else:
    ensemble["ENSEMBLE_PRED_CONVERTED"] = pd.Series([], dtype=float)


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
# Get base predictions (USD per ounce)
fallback_usd_per_oz = actuals["GOLD_CLOSE"].iloc[-1]
future_preds_usd_per_oz = recursive_forecast(
    ensemble["ENSEMBLE_PRED"].tolist() if "ENSEMBLE_PRED" in ensemble.columns else [],
    horizon_days,
    fallback_value=fallback_usd_per_oz
)

# Convert to INR for specified weight
future_preds_inr = [
    convert_usd_per_oz_to_inr_per_gram(pred, usd_inr, weight_grams)
    for pred in future_preds_usd_per_oz
]

future_dates = pd.date_range(
    start=actuals.index[-1] + timedelta(days=1),
    periods=horizon_days
)

forecast_df = pd.DataFrame(
    {
        "Prediction (USD/oz)": future_preds_usd_per_oz,
        f"Prediction (₹ for {weight_grams}g)": future_preds_inr
    },
    index=future_dates
)


# -------------------------------------------------
# MAIN PLOT - Historical + Forecast
# -------------------------------------------------
fig_main = go.Figure()

# Historical data
fig_main.add_trace(go.Scatter(
    x=actuals.index[-730:],
    y=actuals["GOLD_CLOSE_CONVERTED"].iloc[-730:],
    name=f"Historical ({weight_grams}g)",
    line=dict(color="royalblue", width=2)
))

# Forecast data
fig_main.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df[f"Prediction (₹ for {weight_grams}g)"],
    name=f"Ensemble Forecast ({weight_grams}g)",
    line=dict(color="orange", width=2, dash="dash")
))

fig_main.update_layout(
    height=550,
    template="plotly_dark",
    title=f"Gold Price Forecast for {weight_grams} grams",
    xaxis_title="Date",
    yaxis_title=f"Gold Price (₹ for {weight_grams}g)",
    hovermode="x unified",
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

st.plotly_chart(fig_main, use_container_width=True)


# -------------------------------------------------
# OUTPUT METRICS
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric(
        f"Next Day Prediction ({weight_grams}g)",
        f"₹ {forecast_df.iloc[0][f'Prediction (₹ for {weight_grams}g)']:,.2f}"
    )

with col2:
    st.metric(
        "Gold Price (USD/oz)",
        f"${forecast_df.iloc[0]['Prediction (USD/oz)']:,.2f}"
    )

# -------------------------------------------------
# FORECAST TREND CHART
# -------------------------------------------------
if horizon_days > 1:
    st.subheader(f"Forecast Trend ({weight_grams} grams)")
    
    # Create a separate chart for forecast trend only
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df[f"Prediction (₹ for {weight_grams}g)"],
        mode='lines+markers',
        name=f'Forecast ({weight_grams}g)',
        line=dict(color='#FFA500', width=3),
        marker=dict(size=6, color='#FFA500'),
        hovertemplate='%{x|%b %d, %Y}<br>₹ %{y:,.2f}<extra></extra>'
    ))
    
    # Add starting point (last historical value)
    fig_trend.add_trace(go.Scatter(
        x=[actuals.index[-1]],
        y=[actuals["GOLD_CLOSE_CONVERTED"].iloc[-1]],
        mode='markers',
        name='Last Historical',
        marker=dict(size=10, color='royalblue', symbol='circle'),
        hovertemplate='%{x|%b %d, %Y}<br>₹ %{y:,.2f}<extra></extra>'
    ))
    
    fig_trend.update_layout(
        height=400,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title=f"Price (₹ for {weight_grams}g)",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Display forecast summary statistics
    with st.expander("Forecast Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_start = forecast_df.iloc[0][f'Prediction (₹ for {weight_grams}g)']
            forecast_end = forecast_df.iloc[-1][f'Prediction (₹ for {weight_grams}g)']
            percent_change = ((forecast_end - forecast_start) / forecast_start) * 100
            
            st.metric(
                "Total Change",
                f"₹ {(forecast_end - forecast_start):,.2f}",
                f"{percent_change:.2f}%"
            )
        
        with col2:
            avg_price = forecast_df[f'Prediction (₹ for {weight_grams}g)'].mean()
            st.metric("Average Price", f"₹ {avg_price:,.2f}")
        
        with col3:
            max_price = forecast_df[f'Prediction (₹ for {weight_grams}g)'].max()
            min_price = forecast_df[f'Prediction (₹ for {weight_grams}g)'].min()
            st.metric("Price Range", f"₹ {min_price:,.2f} - ₹ {max_price:,.2f}")

# -------------------------------------------------
# ADDITIONAL INFORMATION
# -------------------------------------------------
with st.expander("📊 Price Information & Conversion Details"):
    st.info(f"""
    **Price Conversion Details:**
    
    - **Current USD/INR Exchange Rate**: ₹ {usd_inr:.2f}
    - **Gold Unit**: Traded per troy ounce (31.1035 grams) in USD
    - **Selected Weight**: {weight_grams} grams
    
    **Conversion Formula:**
    1. USD per ounce → USD per gram: `USD/oz ÷ 31.1035`
    2. USD per gram → INR per gram: `USD/gram × {usd_inr:.2f}`
    3. INR for {weight_grams}g: `INR/gram × {weight_grams}`
    
    **For {weight_grams} grams:**
    - Price per gram: ₹ {convert_usd_per_oz_to_inr_per_gram(future_preds_usd_per_oz[0], usd_inr, 1.0):,.2f}
    - Total for {weight_grams}g: ₹ {forecast_df.iloc[0][f'Prediction (₹ for {weight_grams}g)']:,.2f}
    """)

# -------------------------------------------------
# DOWNLOAD FUNCTIONALITY
# -------------------------------------------------
csv_data = forecast_df.reset_index().rename(columns={"index": "date"})
csv_data["weight_grams"] = weight_grams
csv_data["usd_inr_rate"] = usd_inr
csv_data["price_per_gram_inr"] = csv_data["Prediction (USD/oz)"].apply(
    lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, 1.0)
)

st.download_button(
    f"📥 Download Forecast CSV ({weight_grams}g)",
    csv_data.to_csv(index=False).encode(),
    f"gold_forecast_{weight_grams}g_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    "text/csv",
    help="Download the forecast data with conversion details"
)
