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
import logging

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
logger = logging.getLogger("gold_dashboard")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MAX_YEARS = 5
PAGE_SIZE = 1000  # PostgREST default page size

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

HORIZON_MAPPING = {
    "Next Day": "1d",
    "7 Days": "7d",
    "30 Days": "30d",
    "90 Days": "90d",
    "180 Days": "180d",
    "1 Year": "365d",
}

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
# SUPABASE LOADERS WITH PAGINATION
# -------------------------------------------------
def _fetch_all_rows(table, select_cols, order_col, filters=None, page_size=PAGE_SIZE):
    """
    Fetch ALL rows from a Supabase table using pagination.
    
    Args:
        table: Table name (string)
        select_cols: Columns to select (string, comma-separated)
        order_col: Column to order by (string)
        filters: Dict of filter conditions {col: value} for .eq() filters
        page_size: Number of rows per page
    
    Returns:
        List of all records (dicts)
    """
    all_rows = []
    start = 0
    while True:
        end = start + page_size - 1
        query = supabase.table(table).select(select_cols).order(order_col).range(start, end)
        
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        
        response = query.execute()
        page = response.data or []
        all_rows.extend(page)
        
        if len(page) < page_size:
            break
        start += page_size
    
    return all_rows


def _as_naive_datetime(values):
    """Normalize Supabase date/timestamp values for reliable comparisons."""
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(converted, pd.DatetimeIndex):
        return converted.tz_localize(None)
    return converted.dt.tz_localize(None)


@st.cache_data(ttl=300)
def load_actuals():
    """Fetch ALL historical gold prices from Supabase with pagination."""
    try:
        rows = _fetch_all_rows('gold_prices', 'date, close', 'date')
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "GOLD_CLOSE"])
        df["date"] = _as_naive_datetime(df["date"])
        df = df.rename(columns={"close": "GOLD_CLOSE"})
        df["GOLD_CLOSE"] = pd.to_numeric(df["GOLD_CLOSE"], errors="coerce")
        df = (df.dropna(subset=["date", "GOLD_CLOSE"])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True))
        return df.set_index("date")
    except Exception as e:
        st.error(f"❌ Error loading gold_prices: {str(e)}")
        return pd.DataFrame(columns=["date", "GOLD_CLOSE"])


@st.cache_data(ttl=300)
def load_ensemble_forecast(horizon='30d', latest_historical_date=None):
    """Fetch ensemble predictions from Supabase with pagination."""
    try:
        rows = _fetch_all_rows('predictions', 'date, ensemble_pred, chronos_pred, nhits_pred, model_version', 'date', filters={'horizon': horizon})
        df = pd.DataFrame(rows)
        if df.empty:
            empty = pd.DataFrame(columns=["date", "ensemble_pred", "chronos_pred", "nhits_pred", "model_version"])
            return empty, None
        df["date"] = _as_naive_datetime(df["date"])
        # Keep the database schema names throughout the dashboard. Supabase
        # returns these fields in lowercase; renaming only one field caused
        # Keep all three prediction fields in the database's lowercase form.
        for column in ["ensemble_pred", "chronos_pred", "nhits_pred"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = (df.dropna(subset=["date", "ensemble_pred"])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True))

        # A prediction row is only usable for this dashboard when it follows
        # the currently loaded historical data. This excludes old forecast
        # batches when the predictions table contains multiple daily runs.
        if latest_historical_date is not None:
            cutoff = pd.Timestamp(latest_historical_date)
            df = df[df["date"] > cutoff].reset_index(drop=True)
        logger.info(
            "Prediction load: database horizon=%s rows=%d columns=%s",
            horizon, len(df), list(df.columns)
        )
        return df.set_index("date"), None
    except Exception as e:
        st.error(f"❌ Error loading predictions: {str(e)}")
        logger.exception("Prediction query failed for horizon %s", horizon)
        empty = pd.DataFrame(columns=["date", "ensemble_pred", "chronos_pred", "nhits_pred", "model_version"])
        return empty, str(e)


@st.cache_data(ttl=60)
def load_prediction_inventory():
    """Return all prediction metadata for diagnostics and horizon availability."""
    columns = "date, horizon, chronos_pred, nhits_pred, ensemble_pred, model_version"
    try:
        rows = _fetch_all_rows("predictions", columns, "date")
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=[name.strip() for name in columns.split(",")]), None
        df["date"] = _as_naive_datetime(df["date"])
        for column in ["chronos_pred", "nhits_pred", "ensemble_pred"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df.sort_values(["horizon", "date"]).reset_index(drop=True), None
    except Exception as e:
        logger.exception("Prediction inventory query failed")
        return pd.DataFrame(columns=[name.strip() for name in columns.split(",")]), str(e)


@st.cache_data(ttl=60)
def load_latest_pipeline_run():
    """Load the latest pipeline run (regardless of status)."""
    try:
        response = supabase.table('pipeline_runs')\
            .select('*')\
            .order('started_at', desc=True)\
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


@st.cache_data(ttl=60)
def load_last_successful_pipeline_run():
    """Load the last successful pipeline run."""
    try:
        response = supabase.table('pipeline_runs')\
            .select('*')\
            .eq('status', 'success')\
            .order('started_at', desc=True)\
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def load_latest_prediction_date(latest_historical_date=None):
    """Get the newest usable prediction date after the actuals cutoff."""
    try:
        rows = _fetch_all_rows(
            'predictions', 'date', 'date',
            filters=None,
        )
        dates = _as_naive_datetime(pd.Series([row.get('date') for row in rows]))
        dates = dates.dropna()
        if latest_historical_date is not None:
            dates = dates[dates > pd.Timestamp(latest_historical_date)]
        if not dates.empty:
            return dates.max()
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
            ["Next Day", "7 Days", "30 Days", "90 Days", "180 Days", "1 Year"]
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
    latest_gold_date = actuals.index.max() if not actuals.empty else None

# Map preset to horizon key
if mode == "Preset":
    horizon_key = HORIZON_MAPPING[preset]
    horizon_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "180d": 180, "365d": 365}[horizon_key]
else:
    horizon_days = max(1, min(years * 365 + months * 30 + days, 365 * MAX_YEARS))
    horizon_key = "30d"  # Default to 30d predictions for custom

ensemble, prediction_query_error = load_ensemble_forecast(horizon_key, latest_gold_date)
prediction_inventory, inventory_error = load_prediction_inventory()
next_day_ensemble, next_day_query_error = load_ensemble_forecast("1d", latest_gold_date)
latest_pred_date = load_latest_prediction_date(latest_gold_date)

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
if not prediction_query_error and ensemble.empty and horizon_key in ['1d', '7d', '30d', '90d', '180d', '365d']:
    st.warning(
        f"{horizon_days}-day forecast is not currently available. "
        "Run the forecasting pipeline after enabling this horizon."
    )
elif not prediction_query_error and len(ensemble) < horizon_days and horizon_days > 30:
    st.warning(
        f"{horizon_days}-day forecast incomplete: {len(ensemble)}/{horizon_days} points available. "
        "Missing values were not padded or forward-filled."
    )

# -------------------------------------------------
# CONVERT ACTUALS
# -------------------------------------------------
actuals["GOLD_CLOSE_CONVERTED"] = actuals["GOLD_CLOSE"].apply(
    lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
)

# -------------------------------------------------
# CONVERT FORECASTS
# -------------------------------------------------
if "ensemble_pred" in ensemble.columns and not ensemble.empty:
    ensemble["ensemble_pred_converted"] = ensemble["ensemble_pred"].apply(
        lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
    )
else:
    ensemble["ensemble_pred_converted"] = pd.Series([], dtype=float)

# -------------------------------------------------
# FORECAST
# -------------------------------------------------
latest_historical_date = actuals.index.max()
forecast_available = not prediction_query_error and len(ensemble) > 0
if forecast_available or horizon_days <= 30:
    fallback_usd_per_oz = actuals["GOLD_CLOSE"].iloc[-1]
    if horizon_days > 30:
        # Long horizons use only rows actually returned by the model/database.
        # Missing rows remain missing; they are never padded or extrapolated.
        future_preds_usd_per_oz = ensemble["ensemble_pred"].tolist()
    else:
        future_preds_usd_per_oz = recursive_forecast(
            ensemble["ensemble_pred"].tolist() if "ensemble_pred" in ensemble.columns and not ensemble.empty else [],
            horizon_days,
            fallback_value=fallback_usd_per_oz,
            historical_prices=actuals["GOLD_CLOSE"].tolist()
        )
    future_preds_inr = [
        convert_usd_per_oz_to_inr_per_gram(pred, usd_inr, weight_grams)
        for pred in future_preds_usd_per_oz
    ]
    if horizon_days > 30 and len(ensemble):
        future_dates = pd.DatetimeIndex(ensemble.index)
    elif len(ensemble) >= horizon_days:
        future_dates = pd.DatetimeIndex(ensemble.index[:horizon_days])
    else:
        future_dates = pd.date_range(
            start=latest_historical_date + timedelta(days=1),
            periods=horizon_days,
            freq="D"
        )
    if len(future_dates) and future_dates[0] <= latest_historical_date:
        st.error("Forecast date alignment problem: the first forecast is not after the latest historical date.")
        forecast_df = pd.DataFrame(columns=["date", "forecast_price"])
    else:
        forecast_df = pd.DataFrame(
            {"forecast_price": future_preds_inr}, index=future_dates
        ).reset_index().rename(columns={"index": "date"})
else:
    forecast_df = pd.DataFrame(columns=["date", "forecast_price"])

forecast_df = forecast_df.sort_values("date").reset_index(drop=True)
forecast_date_duplicates = int(forecast_df["date"].duplicated().sum()) if not forecast_df.empty else 0
forecast_nan_count = int(forecast_df["forecast_price"].isna().sum()) if not forecast_df.empty else 0
forecast_invalid_count = 0
if not forecast_df.empty:
    forecast_invalid_count = int((~np.isfinite(forecast_df["forecast_price"].astype(float))).sum())
if forecast_date_duplicates or forecast_nan_count or forecast_invalid_count:
    st.error(
        "Invalid forecast data received; the forecast trace was omitted. "
        f"duplicate_dates={forecast_date_duplicates}, nan={forecast_nan_count}, "
        f"non_finite={forecast_invalid_count}"
    )
    forecast_df = pd.DataFrame(columns=["date", "forecast_price"])
expected_forecast_length = len(future_dates) if "future_dates" in locals() else 0
if len(forecast_df) != expected_forecast_length:
    st.error("Forecast date/value length mismatch; the forecast trace was omitted.")
    forecast_df = pd.DataFrame(columns=["date", "forecast_price"])

forecast_line_mode = "lines+markers" if 0 < len(forecast_df) <= 30 else "lines"
selected_ui_horizon = preset if mode == "Preset" else "Custom"
available_horizons = sorted(prediction_inventory["horizon"].dropna().astype(str).unique().tolist()) if not prediction_inventory.empty else []
selected_inventory = (
    prediction_inventory[prediction_inventory["horizon"] == horizon_key]
    if not prediction_inventory.empty and "horizon" in prediction_inventory.columns
    else pd.DataFrame()
)

forecast_std = float(forecast_df["forecast_price"].std(ddof=0)) if len(forecast_df) else 0.0
historical_std = float(actuals["GOLD_CLOSE_CONVERTED"].std(ddof=0))
if len(forecast_df) >= 2 and horizon_days >= 90 and historical_std > 0 and forecast_std / historical_std < 1e-3:
    st.warning(
        f"The {horizon_days}-day forecast has near-zero variance. "
        "This may indicate model collapse; it was not padded or artificially varied."
    )

# -------------------------------------------------
# HISTORICAL DATAFRAME
# -------------------------------------------------
hist_df = actuals.reset_index().rename(
    columns={"date": "date", "GOLD_CLOSE_CONVERTED": "price"}
)

with st.expander("Data Diagnostics"):
    st.write(f"Historical rows loaded: {len(hist_df):,}")
    st.write(f"Historical first date: {hist_df['date'].iloc[0].date()}")
    st.write(f"Historical latest date: {latest_historical_date.date()}")
    st.write(f"Prediction rows loaded: {len(ensemble):,}")
    st.write(f"Prediction latest date: {ensemble.index.max().date() if not ensemble.empty else 'N/A'}")
    st.write(f"Available horizons: {available_horizons or 'None'}")
    st.write(f"Selected UI horizon: {selected_ui_horizon}")
    st.write(f"Selected backend horizon: {horizon_key}")
    st.write(f"Requested forecast points: {horizon_days}")
    st.write(f"Available forecast points: {len(ensemble):,}")
    st.write(f"Prediction range: {selected_inventory['date'].min() if not selected_inventory.empty else 'N/A'} to {selected_inventory['date'].max() if not selected_inventory.empty else 'N/A'}")
    st.write(f"Chronos valid values: {int(ensemble['chronos_pred'].notna().sum()) if 'chronos_pred' in ensemble else 0}")
    st.write(f"N-HiTS valid values: {int(ensemble['nhits_pred'].notna().sum()) if 'nhits_pred' in ensemble else 0}")
    st.write(f"Ensemble valid values: {int(ensemble['ensemble_pred'].notna().sum()) if 'ensemble_pred' in ensemble else 0}")
    st.write(f"Database horizon: {horizon_key}")
    st.write(f"Forecast rows: {len(forecast_df):,}")
    st.write(f"Forecast first date: {forecast_df['date'].iloc[0].date() if not forecast_df.empty else 'N/A'}")
    st.write(f"Forecast last date: {forecast_df['date'].iloc[-1].date() if not forecast_df.empty else 'N/A'}")
    st.write(f"Forecast minimum: {forecast_df['forecast_price'].min() if not forecast_df.empty else 'N/A'}")
    st.write(f"Forecast maximum: {forecast_df['forecast_price'].max() if not forecast_df.empty else 'N/A'}")
    st.write(f"Forecast unique values: {forecast_df['forecast_price'].nunique() if not forecast_df.empty else 0}")
    st.write(f"Forecast NaN count: {forecast_nan_count}")
    st.write(f"Forecast date duplicates: {forecast_date_duplicates}")
    st.write(f"Forecast line mode: {forecast_line_mode}")
    st.write(f"Selected gold weight: {weight_grams:g}g")
    st.write(f"Selected horizon: {horizon_key} ({horizon_days} days)")
    st.write(f"Latest historical price: ₹ {hist_df['price'].iloc[-1]:,.2f}")
    st.write(f"First forecast price: ₹ {forecast_df['forecast_price'].iloc[0]:,.2f}" if not forecast_df.empty else "First forecast price: N/A")

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
            mode=forecast_line_mode,
            name=f"Forecast ({weight_grams}g)",
            line=dict(width=3, color="#FFB000"),
            marker=dict(size=6) if forecast_line_mode == "lines+markers" else None,
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
    if not next_day_query_error and not next_day_ensemble.empty:
        next_day_usd = float(next_day_ensemble.iloc[0]["ensemble_pred"])
        next_day_inr = convert_usd_per_oz_to_inr_per_gram(next_day_usd, usd_inr, weight_grams)
        st.metric(f"Next Day Prediction ({weight_grams}g)", f"₹ {next_day_inr:,.2f}")
    else:
        if next_day_query_error:
            st.metric(f"Next Day Prediction ({weight_grams}g)", "N/A - Query error")
        elif latest_run is not None and latest_run.get('status') == 'failed':
            st.metric(f"Next Day Prediction ({weight_grams}g)", "N/A - Pipeline failed")
        else:
            st.metric(f"Next Day Prediction ({weight_grams}g)", "N/A - No prediction available")

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
            mode=forecast_line_mode,
            name=f"Forecast ({weight_grams}g)",
            line=dict(color="#FFB000", width=4),
            marker=dict(size=6) if forecast_line_mode == "lines+markers" else None
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
