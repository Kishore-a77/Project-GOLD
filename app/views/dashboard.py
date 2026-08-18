"""
Gold Price Forecast Dashboard
Reads predictions from Supabase and displays them in Streamlit.
This is a CONSUMER dashboard - it does NOT run models, training, or pipelines.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

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
from services.fx_service import fetch_usd_inr_rate, FALLBACK_USD_INR

@st.cache_data(ttl=6 * 60 * 60)
def get_usd_inr_rate():
    try:
        return fetch_usd_inr_rate()
    except requests.RequestException:
        st.warning("⚠️ FX API unavailable. Using fallback INR rate.")
        return FALLBACK_USD_INR

# -------------------------------------------------
# CONVERSION
# -------------------------------------------------
def convert_usd_per_oz_to_inr_per_gram(usd_per_oz, usd_inr_rate, weight_grams=1.0):
    usd_per_gram = usd_per_oz / TROY_OUNCE_TO_GRAMS
    inr_per_gram = usd_per_gram * usd_inr_rate
    return inr_per_gram * weight_grams

# -------------------------------------------------
# SUPABASE LOADERS
# -------------------------------------------------

@st.cache_data(ttl=3600)
def load_actuals():
    """Fetch historical gold prices from Supabase."""
    try:
        response = supabase.table('gold_prices').select('date, close').order('date').execute()
        df = pd.DataFrame(response.data)
        if df.empty:
            return pd.DataFrame(columns=["date", "GOLD_CLOSE"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"close": "GOLD_CLOSE"})
        return df.set_index("date")
    except Exception as e:
        st.error(f"❌ Error loading gold_prices: {str(e)}")
        return pd.DataFrame(columns=["date", "GOLD_CLOSE"])


@st.cache_data(ttl=3600)
def load_ensemble_forecast(horizon='30d'):
    """Fetch ensemble predictions from Supabase."""
    try:
        response = (supabase.table('predictions')
                    .select('date, ensemble_pred, chronos_pred, nhits_pred, model_version')
                    .eq('horizon', horizon)
                    .order('date')
                    .execute())
        df = pd.DataFrame(response.data)
        if df.empty:
            return pd.DataFrame(columns=["date", "ENSEMBLE_PRED", "CHRONOS_PRED", "NHITS_PRED", "MODEL_VERSION"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"ensemble_pred": "ENSEMBLE_PRED"})
        return df.set_index("date")
    except Exception as e:
        st.error(f"❌ Error loading predictions: {str(e)}")
        return pd.DataFrame(columns=["date", "ENSEMBLE_PRED", "CHRONOS_PRED", "NHITS_PRED", "MODEL_VERSION"])


@st.cache_data(ttl=600)
def load_latest_pipeline_run():
    """Load the latest pipeline run (regardless of status)."""
    try:
        response = supabase.table('pipeline_runs')\
            .select('*')\
            .order('started_at', descending=True)\
            .limit(1)\
            .execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['started_at'] = pd.to_datetime(df['started_at'])
            df['finished_at'] = pd.to_datetime(df['finished_at'])
            return df.iloc[0]
    except Exception as e:
        st.error(f"❌ Error loading pipeline status: {str(e)}")
    return None


@st.cache_data(ttl=3600)
def load_last_successful_pipeline_run():
    """Load the last successful pipeline run."""
    try:
        response = supabase.table('pipeline_runs')\
            .select('*')\
            .eq('status', 'success')\
            .order('started_at', descending=True)\
            .limit(1)\
            .execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['started_at'] = pd.to_datetime(df['started_at'])
            df['finished_at'] = pd.to_datetime(df['finished_at'])
            return df.iloc[0]
    except Exception as e:
        st.error(f"❌ Error loading pipeline status: {str(e)}")
    return None


@st.cache_data(ttl=3600)
def load_latest_gold_price_date():
    """Get the most recent date available in gold_prices."""
    try:
        response = supabase.table('gold_prices')\
            .select('date')\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        if response.data:
            return pd.to_datetime(response.data[0]['date'])
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def load_latest_prediction_date():
    """Get the most recent date available in predictions."""
    try:
        response = supabase.table('predictions')\
            .select('date')\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        if response.data:
            return pd.to_datetime(response.data[0]['date'])
    except Exception:
        pass
    return None


def load_model_metadata():
    """Load the active model metadata from Supabase."""
    try:
        response = supabase.table('model_metadata')\
            .select('*')\
            .eq('model_name', 'nhits')\
            .eq('is_active', True)\
            .single()\
            .execute()
        return response.data
    except Exception:
        return None

# -------------------------------------------------
# RECURSIVE FORECAST
# -------------------------------------------------
def recursive_forecast(base_preds, horizon_days, fallback_value, historical_prices=None):
    if base_preds is None or len(base_preds) == 0:
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
    latest_run = load_latest_pipeline_run()
    last_successful_run = load_last_successful_pipeline_run()
    model_metadata = load_model_metadata()
    latest_gold_date = load_latest_gold_price_date()
    latest_pred_date = load_latest_prediction_date()

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
# SYSTEM STATUS
# -------------------------------------------------
status_cols = st.columns(4)

with status_cols[0]:
    data_updated = latest_gold_date.strftime('%Y-%m-%d') if latest_gold_date is not None else 'N/A'
    st.metric("Data Last Updated", data_updated)

with status_cols[1]:
    pred_generated = latest_pred_date.strftime('%Y-%m-%d') if latest_pred_date is not None else 'N/A'
    st.metric("Predictions Last Generated", pred_generated)

with status_cols[2]:
    if latest_run is not None:
        status = latest_run.get('status', 'unknown')
        if status == 'success':
            st.metric("Pipeline Status", "✅ Success", delta_color="normal")
        elif status == 'failed':
            st.metric("Pipeline Status", "❌ Failed", delta_color="inverse")
        else:
            st.metric("Pipeline Status", status.title())
    else:
        st.metric("Pipeline Status", "No runs")

with status_cols[3]:
    model_version = None
    if latest_run is not None:
        model_version = latest_run.get('model_version')
    if not model_version and model_metadata:
        model_version = model_metadata.get('version')
    st.metric("Model Version", model_version or "N/A")

# Stale data warning if latest pipeline failed
if latest_run is not None and latest_run.get('status') == 'failed':
    st.error("⚠️ **Latest pipeline run FAILED.** Predictions and data may be stale. Showing last available data.")
    if last_successful_run is not None:
        last_success_finished = last_successful_run.get('finished_at')
        st.warning(f"Last successful run: {last_success_finished.strftime('%Y-%m-%d %H:%M UTC') if last_success_finished else 'N/A'}")

st.divider()

# -------------------------------------------------
# GUARD: Check if we have data
# -------------------------------------------------
if actuals.empty:
    st.error("❌ No gold price data available. Please run the daily pipeline to populate the database, then reload this page.")
    st.stop()

# Check if we have predictions for the selected horizon
if ensemble.empty and horizon_key in ['1d', '7d', '30d']:
    st.warning(f"⚠️ No predictions found for horizon '{horizon_key}'. The latest pipeline may not have generated forecasts for this horizon.")
    # Don't stop - show historical data and use recursive forecast as fallback

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
    ensemble["ENSEMBLE_PRED"].tolist() if "ENSEMBLE_PRED" in ensemble.columns and not ensemble.empty else [],
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
        if latest_run is not None and latest_run.get('status') == 'failed':
            st.metric(f"Next Day Prediction ({weight_grams}g)", "N/A - Pipeline failed")
        else:
            st.metric(f"Next Day Prediction ({weight_grams}g)", "N/A")

with col2:
    st.metric("USD → INR Rate", f"₹ {usd_inr:.2f}")

# -------------------------------------------------
# NEXT WEEK TABLE
# -------------------------------------------------
if horizon_days >= 7 and not forecast_df.empty:
    st.subheader(f"Next 7 Days Forecast ({weight_grams} grams)")
    week_df = forecast_df.head(7).copy()
    week_df["date"] = week_df["date"].dt.strftime("%Y-%m-%d")
    week_df["forecast_price"] = week_df["forecast_price"].apply(lambda x: f"₹ {x:,.2f}")
    st.dataframe(week_df, use_container_width=True, hide_index=True)

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
        percent_change = (total_change / forecast_start) * 100 if forecast_start != 0 else 0
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
# NEXT MONTH TABLE (if 30d horizon selected)
# -------------------------------------------------
if horizon_key == '30d' and not forecast_df.empty:
    st.subheader(f"Next 30 Days Forecast ({weight_grams} grams)")
    month_df = forecast_df.head(30).copy()
    month_df["date"] = month_df["date"].dt.strftime("%Y-%m-%d")
    month_df["forecast_price"] = month_df["forecast_price"].apply(lambda x: f"₹ {x:,.2f}")
    st.dataframe(month_df, use_container_width=True, hide_index=True)

# -------------------------------------------------
# EXTRA INFO
# -------------------------------------------------
with st.expander("📊 Price Information & Conversion Details"):
    st.info(f"""
    Current USD/INR Rate: ₹ {usd_inr:.2f}
    Gold is traded internationally in USD per troy ounce.
    Selected Weight: {weight_grams} grams
    """)

# Model info expander
with st.expander("🤖 Model & Pipeline Info"):
    if latest_run is not None:
        st.write(f"**Latest pipeline run:** {latest_run.get('started_at').strftime('%Y-%m-%d %H:%M UTC') if latest_run.get('started_at') else 'N/A'}")
        status = latest_run.get('status', 'unknown')
        if status == 'success':
            st.write(f"**Status:** ✅ Success")
            st.write(f"**Model version:** {latest_run.get('model_version', 'N/A')}")
        elif status == 'failed':
            st.write(f"**Status:** ❌ Failed")
            if latest_run.get('error'):
                st.write(f"**Error:** {str(latest_run.get('error'))[:200]}")
        else:
            st.write(f"**Status:** {status}")
    else:
        st.write("No pipeline run data available.")
    
    if last_successful_run is not None and latest_run is not None and latest_run.get('status') == 'failed':
        st.write(f"**Last successful run:** {last_successful_run.get('finished_at').strftime('%Y-%m-%d %H:%M UTC') if last_successful_run.get('finished_at') else 'N/A'}")
        st.write(f"**Last successful model version:** {last_successful_run.get('model_version', 'N/A')}")
    
    if model_metadata:
        st.write(f"**Active model:** {model_metadata.get('model_name', 'N/A')}")
        st.write(f"**Model version:** {model_metadata.get('version', 'N/A')}")
        if model_metadata.get('mae') is not None:
            st.write(f"**Metrics:** MAE={model_metadata.get('mae', 0):.4f}, RMSE={model_metadata.get('rmse', 0):.4f}, MAPE={model_metadata.get('mape', 0):.4f}%")
    else:
        st.write("No model metadata available.")

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