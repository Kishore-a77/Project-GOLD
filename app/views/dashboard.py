"""
Gold Price Forecast Dashboard
Reads predictions from Supabase and displays them in Streamlit.
This is a CONSUMER dashboard - it does NOT retrain models or run heavy pipeline inference locally.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("gold_dashboard")

# -------------------------------------------------
# DATABASE & SERVICE INTEGRATION
# -------------------------------------------------
from app.db.supabase_client import (
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseAPIError,
)
from app.services.database_service import (
    fetch_actuals,
    fetch_ensemble_forecast,
    fetch_prediction_inventory,
    check_system_readiness,
    STATE_READY,
    STATE_STALE,
    STATE_MISSING,
    STATE_NETWORK_ERROR,
    STATE_CONFIG_ERROR,
    STATE_PIPELINE_RUNNING,
)
from app.services.refresh_service import (
    initiate_automatic_refresh,
    poll_for_fresh_data,
    is_pipeline_currently_running,
)
from services.fx_service import fetch_usd_inr_rate, FALLBACK_USD_INR

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MAX_YEARS = 5

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
@st.cache_data(ttl=6 * 60 * 60)
def get_usd_inr_rate():
    try:
        return fetch_usd_inr_rate()
    except Exception:
        return FALLBACK_USD_INR


# -------------------------------------------------
# CONVERSION
# -------------------------------------------------
def convert_usd_per_oz_to_inr_per_gram(usd_per_oz, usd_inr_rate, weight_grams=1.0):
    usd_per_gram = usd_per_oz / TROY_OUNCE_TO_GRAMS
    inr_per_gram = usd_per_gram * usd_inr_rate
    return inr_per_gram * weight_grams


# -------------------------------------------------
# CACHED DATA LOADERS (Failures are NOT cached)
# -------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_actuals_cached():
    """Load historical actuals from database service."""
    return fetch_actuals()


@st.cache_data(ttl=30, show_spinner=False)
def load_ensemble_forecast_cached(horizon: str, latest_date: pd.Timestamp):
    """Load predictions for selected horizon from database service."""
    return fetch_ensemble_forecast(horizon=horizon, latest_historical_date=latest_date)


@st.cache_data(ttl=30, show_spinner=False)
def load_prediction_inventory_cached():
    """Load prediction inventory metadata."""
    return fetch_prediction_inventory()


# -------------------------------------------------
# RECURSIVE FORECAST (Fallback / Extrapolation)
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


def render_dashboard():
    """Render the live dashboard inside Streamlit's script runtime."""

    st.set_page_config(
        page_title="Gold Price Forecast - Ensemble",
        layout="wide",
    )
    st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------
    # TITLE & HEADER
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


    # Map preset to horizon key
    if mode == "Preset":
        horizon_key = HORIZON_MAPPING[preset]
        horizon_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "180d": 180, "365d": 365}[horizon_key]
    else:
        horizon_days = max(1, min(years * 365 + months * 30 + days, 365 * MAX_YEARS))
        horizon_key = "30d"  # Default to 30d predictions for custom


    # -------------------------------------------------
    # STARTUP ORCHESTRATION & READINESS EVALUATION
    # -------------------------------------------------
    readiness = check_system_readiness()
    state = readiness["state"]
    latest_gold_date = readiness.get("latest_gold_date")

    # Keep health state visible without conflating prediction target dates
    # with the timestamp of the successful pipeline execution.
    health_labels = {
        STATE_READY: ("Data: Connected", "Predictions: Fresh", "Pipeline: Success"),
        STATE_STALE: ("Data: Connected", "Predictions: Stale", "Pipeline: Refreshing"),
        STATE_MISSING: ("Data: Connected", "Predictions: Missing", "Pipeline: Starting"),
        STATE_PIPELINE_RUNNING: ("Data: Connected", "Predictions: Updating", "Pipeline: Running"),
        STATE_NETWORK_ERROR: ("Data: Connection unavailable", "Predictions: Unavailable", "Pipeline: Waiting"),
        STATE_CONFIG_ERROR: ("Data: Configuration unavailable", "Predictions: Unavailable", "Pipeline: Not configured"),
    }
    data_health, prediction_health, pipeline_health = health_labels.get(
        state, ("Data: Unknown", "Predictions: Unknown", "Pipeline: Unknown")
    )
    health_time = readiness.get("pred_generated_date")
    health_time_text = (
        health_time.strftime("%Y-%m-%d %H:%M UTC")
        if hasattr(health_time, "strftime") else "N/A"
    )
    st.caption(
        f"{data_health}  ·  {prediction_health}  ·  {pipeline_health}"
        f"  ·  Last update: {health_time_text}"
    )

    # STATE E — Permanent configuration failure
    if state == STATE_CONFIG_ERROR:
        st.error("❌ Supabase configuration is missing or invalid. Check SUPABASE_URL and SUPABASE_KEY / SUPABASE_SERVICE_KEY.")
        st.info("Ensure the credentials are properly set in your local .env file or Streamlit Cloud Secrets.")
        st.stop()

    # STATE D — Temporary network failure
    if state == STATE_NETWORK_ERROR:
        st.warning("⚠️ Connecting to prediction service... Retrying connection automatically.")
        st.caption("A transient network delay or timeout occurred while reaching Supabase. Please wait a moment.")
        if st.button("🔄 Retry Connection Now"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    # STATE C — Connected + no data
    if state == STATE_MISSING:
        st.info("ℹ️ Initializing database & generating forecasts... Please wait while latest market data is retrieved.")
        with st.spinner("Automating daily pipeline execution in the background..."):
            triggered, trigger_msg = initiate_automatic_refresh()
            st.caption(f"Status: {trigger_msg}")
            # Bounded poll for new predictions
            fresh_readiness = poll_for_fresh_data(
                max_wait_seconds=25,
                check_interval=4.0,
                initial_gold_date=readiness.get("latest_gold_date"),
                initial_total_preds=readiness.get("total_predictions", 0),
                initial_pred_generated_date=readiness.get("pred_generated_date"),
            )
            if fresh_readiness.get("state") in (STATE_READY, STATE_STALE):
                readiness = fresh_readiness
                state = readiness["state"]
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("Forecast pipeline is currently running. Please reload this page in a minute once complete.")
                st.stop()

    # STATE B — Connected + stale data
    if state == STATE_STALE:
        st.info("ℹ️ Showing latest available forecasts. An automatic data update has been initiated in the background.")
        # Check if pipeline is running, or initiate if not
        is_running, run_reason = is_pipeline_currently_running()
        if not is_running:
            triggered, trigger_msg = initiate_automatic_refresh()
            st.caption(f"Refresh: {trigger_msg}")
        else:
            triggered = True
            st.caption(f"Refresh: {run_reason or 'pipeline already running'}")

        if triggered:
            fresh_readiness = poll_for_fresh_data(
                max_wait_seconds=25,
                check_interval=4.0,
                initial_gold_date=readiness.get("latest_gold_date"),
                initial_total_preds=readiness.get("total_predictions", 0),
                initial_pred_generated_date=readiness.get("pred_generated_date"),
            )
            if fresh_readiness.get("state") == STATE_READY:
                st.cache_data.clear()
                st.rerun()

    if state == STATE_PIPELINE_RUNNING:
        st.info("A pipeline execution is already running. Showing the latest available data without starting another run.")

    # -------------------------------------------------
    # DATA RETRIEVAL (Protected by centralized database service)
    # -------------------------------------------------
    try:
        with st.spinner("Loading forecasts from Supabase..."):
            actuals = load_actuals_cached()
            usd_inr = get_usd_inr_rate()
            if latest_gold_date is None and not actuals.empty:
                latest_gold_date = actuals.index.max()

            ensemble = load_ensemble_forecast_cached(horizon_key, latest_gold_date)
            next_day_ensemble = load_ensemble_forecast_cached("1d", latest_gold_date)
            prediction_inventory = load_prediction_inventory_cached()
    except (DatabaseConnectionError, DatabaseAPIError) as dbe:
        st.warning(f"⚠️ Connecting to prediction service: {dbe}")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Error loading data: {exc}")
        st.stop()

    # -------------------------------------------------
    # SYSTEM STATUS HEADER METRICS
    # -------------------------------------------------
    status_cols = st.columns(4)

    with status_cols[0]:
        data_updated = latest_gold_date.strftime('%Y-%m-%d') if latest_gold_date is not None else 'N/A'
        st.metric("Data Last Updated", data_updated)

    with status_cols[1]:
        # Correct mapping: read from last successful pipeline run, NOT forecast horizon date!
        pred_gen_dt = readiness.get("pred_generated_date")
        if pred_gen_dt is not None:
            if hasattr(pred_gen_dt, "strftime"):
                pred_generated = pred_gen_dt.strftime('%Y-%m-%d %H:%M UTC')
            else:
                pred_generated = str(pred_gen_dt)[:16]
        else:
            pred_generated = 'N/A'
        st.metric("Predictions Last Generated", pred_generated)

    with status_cols[2]:
        latest_run = readiness.get("latest_run")
        if latest_run is not None:
            run_status = latest_run.get('status', 'unknown')
            if run_status == 'success':
                st.metric("Pipeline Status", "✅ Success", delta_color="normal")
            elif run_status == 'failed':
                st.metric("Pipeline Status", "❌ Failed", delta_color="inverse")
            elif run_status in ('running', 'in_progress'):
                st.metric("Pipeline Status", "⏳ Running", delta_color="off")
            else:
                st.metric("Pipeline Status", run_status.title())
        else:
            st.metric("Pipeline Status", "No runs")

    with status_cols[3]:
        model_version = readiness.get("model_version") or "N/A"
        st.metric("Model Version", model_version)

    # Stale warning if latest pipeline failed
    if latest_run is not None and latest_run.get('status') == 'failed':
        st.error("⚠️ **Latest pipeline run FAILED.** Showing last available successful forecasts.")
        last_success = readiness.get("last_successful_run")
        if last_success is not None and last_success.get('finished_at'):
            st.warning(f"Last successful run: {last_success.get('finished_at').strftime('%Y-%m-%d %H:%M UTC')}")

    st.divider()

    # -------------------------------------------------
    # GUARD: Check if actuals genuinely exist
    # -------------------------------------------------
    if actuals.empty:
        st.warning("No gold price data available in Supabase.")
        st.stop()

    # -------------------------------------------------
    # PRICE CONVERSIONS
    # -------------------------------------------------
    actuals["GOLD_CLOSE_CONVERTED"] = actuals["GOLD_CLOSE"].apply(
        lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
    )

    if "ensemble_pred" in ensemble.columns and not ensemble.empty:
        ensemble["ensemble_pred_converted"] = ensemble["ensemble_pred"].apply(
            lambda x: convert_usd_per_oz_to_inr_per_gram(x, usd_inr, weight_grams)
        )
    else:
        ensemble["ensemble_pred_converted"] = pd.Series([], dtype=float)

    # -------------------------------------------------
    # FORECAST ASSEMBLY
    # -------------------------------------------------
    latest_historical_date = actuals.index.max()
    # Render only persisted ensemble rows. Never extrapolate or fill missing
    # horizons with synthetic values in the user-facing forecast.
    if len(ensemble) > 0 and "ensemble_pred" in ensemble.columns:
        future_preds_usd_per_oz = ensemble["ensemble_pred"].iloc[:horizon_days].tolist()
        future_preds_inr = [
            convert_usd_per_oz_to_inr_per_gram(pred, usd_inr, weight_grams)
            for pred in future_preds_usd_per_oz
        ]
        future_dates = pd.DatetimeIndex(ensemble.index[:len(future_preds_inr)])
        forecast_df = pd.DataFrame(
            {"forecast_price": future_preds_inr}, index=future_dates
        ).reset_index().rename(columns={"index": "date"})
    else:
        forecast_df = pd.DataFrame(columns=["date", "forecast_price"])

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    # -------------------------------------------------
    # KEY PRICE HIGHLIGHT CARDS
    # -------------------------------------------------
    summary_cols = st.columns(3)

    latest_actual_close_usd = float(actuals["GOLD_CLOSE"].iloc[-1])
    latest_actual_close_inr = convert_usd_per_oz_to_inr_per_gram(latest_actual_close_usd, usd_inr, weight_grams)

    with summary_cols[0]:
        st.metric(
            f"Latest Actual Close ({weight_grams:g}g)",
            f"₹ {latest_actual_close_inr:,.2f}",
            f"${latest_actual_close_usd:,.2f} / oz"
        )

    with summary_cols[1]:
        if not next_day_ensemble.empty:
            next_day_usd = float(next_day_ensemble.iloc[0]["ensemble_pred"])
            next_day_inr = convert_usd_per_oz_to_inr_per_gram(next_day_usd, usd_inr, weight_grams)
            diff_inr = next_day_inr - latest_actual_close_inr
            pct_change = (diff_inr / latest_actual_close_inr) * 100
            st.metric(
                f"Next Day Predicted ({weight_grams:g}g)",
                f"₹ {next_day_inr:,.2f}",
                f"{diff_inr:+,.2f} ({pct_change:+.2f}%)"
            )
        else:
            st.metric(f"Next Day Predicted ({weight_grams:g}g)", "N/A")

    with summary_cols[2]:
        st.metric("USD → INR Rate", f"₹ {usd_inr:.2f}")

    st.write("")

    # -------------------------------------------------
    # MAIN INTERACTIVE CHART
    # -------------------------------------------------
    hist_df = actuals.reset_index().rename(
        columns={"date": "date", "GOLD_CLOSE_CONVERTED": "price"}
    )

    forecast_line_mode = "lines+markers" if 0 < len(forecast_df) <= 30 else "lines"

    fig = go.Figure()

    # Historical Price Trace
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["price"],
            mode="lines",
            name=f"Historical ({weight_grams:g}g)",
            line=dict(width=3, color="#4C78FF")
        )
    )

    # Forecast Trace
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
    elif len(forecast_df) > 1:
        extended_dates = pd.concat([pd.Series([hist_df["date"].iloc[-1]]), forecast_df["date"]])
        extended_prices = pd.concat([pd.Series([hist_df["price"].iloc[-1]]), forecast_df["forecast_price"]])
        fig.add_trace(
            go.Scatter(
                x=extended_dates,
                y=extended_prices,
                mode=forecast_line_mode,
                name=f"Forecast ({weight_grams:g}g)",
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
        yaxis_title=f"Gold Price (₹ for {weight_grams:g}g)",
        hovermode="x unified",
        legend=dict(font=dict(size=14)),
        title=dict(text=f"Gold Price Trajectory & Forecast ({weight_grams:g} grams)", font=dict(size=24)),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # NEXT 7 DAYS FORECAST TABLE
    # -------------------------------------------------
    if horizon_days >= 7 and not forecast_df.empty:
        st.subheader(f"Next 7 Days Forecast ({weight_grams:g} grams)")
        week_df = forecast_df.head(7).copy()
        week_df["date"] = pd.to_datetime(week_df["date"]).dt.strftime("%Y-%m-%d")
        week_df["forecast_price"] = week_df["forecast_price"].apply(lambda x: f"₹ {x:,.2f}")
        week_df.columns = ["Target Date", "Forecast Price"]
        st.dataframe(week_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------
    # FORECAST TREND & SUMMARY STATISTICS
    # -------------------------------------------------
    if horizon_days > 1 and not forecast_df.empty:
        st.subheader(f"Forecast Horizon Dynamics ({weight_grams:g} grams)")
        with st.expander("📊 Forecast Horizon Statistics", expanded=True):
            stat_cols = st.columns(3)
            forecast_start = forecast_df["forecast_price"].iloc[0]
            forecast_end = forecast_df["forecast_price"].iloc[-1]
            total_change = forecast_end - forecast_start
            percent_change = (total_change / forecast_start) * 100 if forecast_start != 0 else 0
            avg_price = forecast_df["forecast_price"].mean()
            max_price = forecast_df["forecast_price"].max()
            min_price = forecast_df["forecast_price"].min()

            with stat_cols[0]:
                st.markdown("### Projected Change")
                st.markdown(f"<h2>₹ {total_change:+,.2f}</h2>", unsafe_allow_html=True)
                if percent_change >= 0:
                    st.success(f"▲ {percent_change:+.2f}%")
                else:
                    st.error(f"▼ {percent_change:+.2f}%")

            with stat_cols[1]:
                st.markdown("### Horizon Average")
                st.markdown(f"<h2>₹ {avg_price:,.2f}</h2>", unsafe_allow_html=True)

            with stat_cols[2]:
                st.markdown("### Projected Range")
                st.markdown(f"<h2>₹ {min_price:,.2f} – ₹ {max_price:,.2f}</h2>", unsafe_allow_html=True)

    # -------------------------------------------------
    # NEXT 30 DAYS TABLE (If 30d or longer horizon selected)
    # -------------------------------------------------
    if horizon_key in ('30d', '90d', '180d', '365d') and len(forecast_df) >= 30:
        with st.expander("📅 View 30-Day Forecast Schedule", expanded=False):
            month_df = forecast_df.head(30).copy()
            month_df["date"] = pd.to_datetime(month_df["date"]).dt.strftime("%Y-%m-%d")
            month_df["forecast_price"] = month_df["forecast_price"].apply(lambda x: f"₹ {x:,.2f}")
            month_df.columns = ["Target Date", "Forecast Price"]
            st.dataframe(month_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------
    # EXPANDERS: DIAGNOSTICS & METADATA
    # -------------------------------------------------
    with st.expander("🔍 System Diagnostics"):
        available_horizons = sorted(prediction_inventory["horizon"].dropna().astype(str).unique().tolist()) if not prediction_inventory.empty else []
        st.write(f"**Historical records:** {len(hist_df):,}")
        st.write(f"**Historical date range:** {hist_df['date'].iloc[0].date()} to {latest_historical_date.date()}")
        st.write(f"**Active predictions in database:** {readiness.get('total_predictions', 0):,}")
        st.write(f"**Available prediction horizons:** {available_horizons or 'None'}")
        st.write(f"**Selected horizon:** {horizon_key} ({horizon_days} days)")
        st.write(f"**Forecast points loaded:** {len(forecast_df):,}")
        st.write(f"**Latest Gold Close (USD):** ${latest_actual_close_usd:,.2f}")
        st.write(f"**Live FX Rate (USD→INR):** ₹ {usd_inr:.2f}")

    with st.expander("🤖 Model & Pipeline Architecture"):
        st.markdown("""
        **Ensemble Architecture:**
        - **Chronos-T5 Small**: Zero-shot probabilistic time-series foundation model pretrained on billions of data points.
        - **N-HiTS**: Neural Hierarchical Interpolation for Time Series with multi-rate signal sampling.
        - **Optimal Weighting**: Blended dynamically based on rolling validation performance.
        """)
        if latest_run is not None:
            st.write(f"**Last execution started:** {latest_run.get('started_at')}")
            st.write(f"**Last execution finished:** {latest_run.get('finished_at')}")
            st.write(f"**Execution status:** {latest_run.get('status')}")
            st.write(f"**Records processed:** {latest_run.get('records_processed')}")
            st.write(f"**Predictions generated:** {latest_run.get('predictions_generated')}")

    # -------------------------------------------------
    # DOWNLOAD FORECAST CSV
    # -------------------------------------------------
    if not forecast_df.empty:
        csv_data = forecast_df.copy()
        csv_data["weight_grams"] = weight_grams
        csv_data["usd_inr_rate"] = usd_inr
        st.download_button(
            f"📥 Download Forecast CSV ({weight_grams:g}g)",
            csv_data.to_csv(index=False).encode(),
            f"gold_forecast_{weight_grams:g}g.csv",
            "text/csv"
        )
