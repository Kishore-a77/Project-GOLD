"""
Gold Price Forecast Dashboard
Reads predictions from Supabase and displays them in Streamlit.
"""

import sys
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from supabase import create_client
import os

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# SUPABASE SETUP
# -------------------------------------------------
# Streamlit Cloud uses st.secrets, local uses environment variables
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    # Fallback for local development
    from dotenv import load_dotenv
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Supabase credentials not found!")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MAX_YEARS = 5

st.set_page_config(
    page_title="Gold Price Forecast — Ensemble",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL STYLES
# -------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
TROY_OUNCE_TO_GRAMS = 31.1035
DEFAULT_WEIGHT_GRAMS = 8.0

# -------------------------------------------------
# FX RATE
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
# CONVERSION
# -------------------------------------------------
def convert_usd_per_oz_to_inr_per_gram(usd_per_oz, usd_inr_rate, weight_grams=1.0):
    usd_per_gram = usd_per_oz / TROY_OUNCE_TO_GRAMS
    inr_per_gram = usd_per_gram * usd_inr_rate
    return inr_per_gram * weight_grams

# -------------------------------------------------
# SUPABASE LOADERS (REPLACED SQLAlchemy with Supabase client)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_actuals():
    """Fetch historical gold prices from Supabase."""
    response = supabase.table('gold_prices').select('date, close').order('date').execute()
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"close": "GOLD_CLOSE"})
    return df.set_index("date")

@st.cache_data(ttl=3600)
def load_ensemble_forecast(horizon='30d'):
    """Fetch ensemble predictions from Supabase."""
    response = (supabase.table('predictions')
                .select('date, ensemble_pred')
                .eq('horizon', horizon)
                .order('date')
                .execute())
    df = pd.DataFrame(response.data)
    if df.empty:
        return pd.DataFrame(columns=["date", "ENSEMBLE_PRED"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"ensemble_pred": "ENSEMBLE_PRED"})
    return df.set_index("date")

# -------------------------------------------------
# RECURSIVE FORECAST
# -------------------------------------------------
def recursive_forecast(base_preds, horizon_days, fallback_value, historical_prices=None):
    if base_preds is None or len(base_preds) == 0:
        st.warning("⚠️ Ensemble forecast unavailable. Using flat forecast.")
        return [fallback_value] * horizon_days

    preds = list(base_preds)
    if horizon_days <= len(preds):
        return preds[:horizon_days]

    if historical_prices is not None and len(historical_prices) > 30:
        hist_prices = np.array(historical_prices)
        log_returns = np.log(hist_prices[1:] / hist_prices[:-1])
        mu = np.mean(log_returns)
        sigma = np.std(log_returns)
    else:
        if len(preds) < 2:
            mu = 0.0
            sigma = 0.0
        else:
            diffs = np.diff(preds[-min(14, len(preds)):])
            mu = np.mean(diffs) / preds[-1] if preds[-1] != 0 else 0.0
            sigma = np.std(diffs) / preds[-1] if preds[-1] != 0 else 0.0

    if np.isnan(mu): mu = 0.0
    if np.isnan(sigma): sigma = 0.0

    np.random.seed(42)
    for _ in range(horizon_days - len(preds)):
        Z = np.random.normal(0, 1)
        next_val = preds[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        preds.append(next_val)

    return preds

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🪙 Gold Price Forecast — Ensemble")
st.caption("Chronos-T5 + N-HiTS • Investor-grade forecasting")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.subheader("Gold Weight")
    weight_grams = st.number_input(
        "Weight (grams)",
        min_value=0.1,
        max_value=1000.0,
        value=DEFAULT_WEIGHT_GRAMS,
        step=0.1
    )
    st.caption(f"Displaying prices for {weight_grams} grams of gold")
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
with st.spinner("Loading data from Supabase..."):
    actuals = load_actuals()
    usd_inr = get_usd_inr_rate()

# Map preset to horizon key
horizon_map = {
    "Next Day": "1d",
    "Next Week (7)": "7d",
    "Next Month (30)": "30d",
    "1 Year": "1y",
    "5 Years": "5y"
}

if mode == "Preset":
    horizon_key = horizon_map[preset]
    horizon_days = {"1d": 1, "7d": 7, "30d": 30, "1y": 365, "5y": 365 * 5}[horizon_key]
else:
    horizon_days = max(1, min(years * 365 + months * 30 + days, 365 * MAX_YEARS))
    horizon_key = "30d"  # Default to 30d predictions for custom

ensemble = load_ensemble_forecast(horizon_key)

# -------------------------------------------------
# CONVERT ACTUALS
# -------------------------------------------------
actuals["GOLD_CLOSE_CONVERTED"] = actuals["GOLD_CLOSE"].apply(
    lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
)

# -------------------------------------------------
# CONVERT FORECASTS
# -------------------------------------------------
if "ENSEMBLE_PRED" in ensemble.columns and not ensemble.empty:
    ensemble["ENSEMBLE_PRED_CONVERTED"] = ensemble["ENSEMBLE_PRED"].apply(
        lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
    )
else:
    ensemble["ENSEMBLE_PRED_CONVERTED"] = pd.Series([], dtype=float)

# -------------------------------------------------
# FORECAST
# -------------------------------------------------
fallback_usd_per_oz = actuals["GOLD_CLOSE"].iloc[-1]

future_preds_usd_per_oz = recursive_forecast(
    ensemble["ENSEMBLE_PRED"].tolist() if "ENSEMBLE_PRED" in ensemble.columns else [],
    horizon_days,
    fallback_value=fallback_usd_per_oz,
    historical_prices=actuals["GOLD_CLOSE"].tolist()
)

future_preds_inr = [
    convert_usd_per_oz_to_inr_per_gram(pred, usd_inr, weight_grams)
    for pred in future_preds_usd_per_oz
]

future_dates = pd.date_range(
    start=actuals.index[-1] + timedelta(days=1),
    periods=horizon_days,
    freq="D"
)

forecast_df = pd.DataFrame(
    {"forecast_price": future_preds_inr},
    index=future_dates
)
forecast_df = forecast_df.dropna().reset_index().rename(columns={"index": "date"})

# -------------------------------------------------
# HISTORICAL DATAFRAME
# -------------------------------------------------
hist_df = actuals.reset_index().rename(
    columns={"date": "date", "GOLD_CLOSE_CONVERTED": "price"}
)
hist_df = hist_df.tail(730)

# -------------------------------------------------
# MAIN CHART
# -------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=hist_df["date"],
        y=hist_df["price"],
        mode="lines",
        name=f"Historical ({weight_grams}g)",
        line=dict(width=3, color="#4C78FF")
    )
)

if len(forecast_df) == 1:
    bridge_x = [hist_df["date"].iloc[-1], forecast_df["date"].iloc[0]]
    bridge_y = [hist_df["price"].iloc[-1], forecast_df["forecast_price"].iloc[0]]
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="Next Day Forecast",
            line=dict(color="#FFB000", width=3),
            marker=dict(size=10)
        )
    )
else:
    extended_dates = pd.concat([pd.Series([hist_df["date"].iloc[-1]]), forecast_df["date"]])
    extended_prices = pd.concat([pd.Series([hist_df["price"].iloc[-1]]), forecast_df["forecast_price"]])
    fig.add_trace(
        go.Scatter(
            x=extended_dates,
            y=extended_prices,
            mode="lines+markers" if len(forecast_df) <= 90 else "lines",
            name=f"Forecast ({weight_grams}g)",
            line=dict(width=3, color="#FFB000"),
            marker=dict(size=7) if len(forecast_df) <= 90 else None,
            connectgaps=True
        )
    )

fig.update_layout(
    height=500,
    template="plotly_dark",
    xaxis=dict(rangeslider=dict(visible=False)),
    xaxis_title="Date",
    yaxis_title=f"Gold Price (₹ for {weight_grams}g)",
    hovermode="x unified",
    legend=dict(font=dict(size=16)),
    title=dict(text=f"Gold Price Forecast for {weight_grams} grams", font=dict(size=34))
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# NEXT DAY METRICS
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if not forecast_df.empty:
        st.metric(f"Next Day Prediction ({weight_grams}g)", f"₹ {forecast_df.iloc[0]['forecast_price']:,.2f}")
    else:
        st.metric(f"Next Day Prediction ({weight_grams}g)", "N/A")

with col2:
    st.metric("USD → INR Rate", f"₹ {usd_inr:.2f}")

# -------------------------------------------------
# FORECAST TREND
# -------------------------------------------------
if horizon_days > 1 and not forecast_df.empty:
    st.subheader(f"Forecast Trend ({weight_grams} grams)")
    fig_trend = go.Figure()

    bridge_x = [hist_df["date"].iloc[-1], forecast_df["date"].iloc[0]]
    bridge_y = [hist_df["price"].iloc[-1], forecast_df["forecast_price"].iloc[0]]

    fig_trend.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines",
            name="Transition",
            line=dict(color="#FFB000", width=2, dash="dot"),
            showlegend=False
        )
    )

    fig_trend.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_price"],
            mode="lines+markers" if len(forecast_df) <= 90 else "lines",
            name=f"Forecast ({weight_grams}g)",
            line=dict(color="#FFB000", width=4),
            marker=dict(size=7) if len(forecast_df) <= 90 else None
        )
    )

    fig_trend.update_layout(
        height=350,
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis_title="Date",
        yaxis_title=f"Price (₹ for {weight_grams}g)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # -------------------------------------------------
    # STATS
    # -------------------------------------------------
    with st.expander("Forecast Statistics"):
        col1, col2, col3 = st.columns(3)
        forecast_start = forecast_df["forecast_price"].iloc[0]
        forecast_end = forecast_df["forecast_price"].iloc[-1]
        total_change = forecast_end - forecast_start
        percent_change = (total_change / forecast_start) * 100
        avg_price = forecast_df["forecast_price"].mean()
        max_price = forecast_df["forecast_price"].max()
        min_price = forecast_df["forecast_price"].min()

        with col1:
            st.markdown("### Total Change")
            st.markdown(f"<h2>₹ {total_change:,.2f}</h2>", unsafe_allow_html=True)
            st.success(f"{percent_change:.2f}%")

        with col2:
            st.markdown("### Average Price")
            st.markdown(f"<h2>₹ {avg_price:,.2f}</h2>", unsafe_allow_html=True)

        with col3:
            st.markdown("### Price Range")
            st.markdown(f"<h2>₹ {min_price:,.2f}</h2><p>to</p><h2>₹ {max_price:,.2f}</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# EXTRA INFO
# -------------------------------------------------
with st.expander("📊 Price Information & Conversion Details"):
    st.info(f"""
    Current USD/INR Rate: ₹ {usd_inr:.2f}
    Gold is traded internationally in USD per troy ounce.
    Selected Weight: {weight_grams} grams
    """)

# -------------------------------------------------
# DOWNLOAD
# -------------------------------------------------
if not forecast_df.empty:
    csv_data = forecast_df.copy()
    csv_data["weight_grams"] = weight_grams
    csv_data["usd_inr_rate"] = usd_inr
    st.download_button(
        f"📥 Download Forecast CSV ({weight_grams}g)",
        csv_data.to_csv(index=False).encode(),
        f"gold_forecast_{weight_grams}g.csv",
        "text/csv"
    )