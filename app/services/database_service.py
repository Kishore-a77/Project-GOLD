"""
app/services/database_service.py

Unified database access service for the Streamlit dashboard.
Encapsulates:
1. Historical gold price retrieval with robust pagination.
2. Horizon prediction retrieval and inventory diagnostics.
3. Pipeline run status & model metadata tracking.
4. Readiness state classification (READY, STALE, MISSING, NETWORK_ERROR, CONFIG_ERROR).
5. Clean error handling so failures are not silently swallowed or falsely cached as empty data.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.db.supabase_client import (
    get_supabase_client,
    execute_with_retry,
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseAPIError,
    DatabaseError,
)

logger = logging.getLogger("database_service")

PAGE_SIZE = 1000

# System Readiness States
STATE_READY = "READY"
STATE_STALE = "STALE"
STATE_MISSING = "MISSING"
STATE_NETWORK_ERROR = "NETWORK_ERROR"
STATE_CONFIG_ERROR = "CONFIG_ERROR"
STATE_PIPELINE_RUNNING = "PIPELINE_RUNNING"

REQUIRED_FORECAST_HORIZONS = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
}


def _as_naive_datetime(values):
    """Normalize Supabase date/timestamp values to naive UTC datetimes."""
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(converted, pd.DatetimeIndex):
        return converted.tz_localize(None)
    return converted.dt.tz_localize(None)


def _fetch_all_rows_with_retry(
    table: str,
    select_cols: str,
    order_col: str,
    filters: Optional[Dict[str, Any]] = None,
    page_size: int = PAGE_SIZE,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetch all rows from a Supabase table using range pagination and bounded retries.
    """
    client = get_supabase_client()
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1

        def query_page(s=start, e=end):
            q = client.table(table).select(select_cols).order(order_col).range(s, e)
            if filters:
                for col, val in filters.items():
                    q = q.eq(col, val)
            return q.execute()

        response = execute_with_retry(
            query_page,
            max_retries=max_retries,
            operation_name=f"fetch {table} range {start}-{end}"
        )
        page = response.data or []
        all_rows.extend(page)

        if len(page) < page_size:
            break
        start += page_size

    return all_rows


def fetch_actuals() -> pd.DataFrame:
    """
    Fetch ALL historical gold prices from Supabase.

    Returns:
        DataFrame with DatetimeIndex and 'GOLD_CLOSE' column.

    Raises:
        DatabaseError on connection or config failure (never returns empty DataFrame on error).
    """
    rows = _fetch_all_rows_with_retry("gold_prices", "date, close", "date")
    if not rows:
        return pd.DataFrame(columns=["date", "GOLD_CLOSE"]).set_index("date")

    df = pd.DataFrame(rows)
    df["date"] = _as_naive_datetime(df["date"])
    df = df.rename(columns={"close": "GOLD_CLOSE"})
    df["GOLD_CLOSE"] = pd.to_numeric(df["GOLD_CLOSE"], errors="coerce")
    df = (
        df.dropna(subset=["date", "GOLD_CLOSE"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df.set_index("date")


def fetch_ensemble_forecast(
    horizon: str = "30d",
    latest_historical_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Fetch ensemble predictions for a given horizon.

    Filters predictions strictly after latest_historical_date.

    Returns:
        DataFrame indexed by forecast target date with columns:
        ['ensemble_pred', 'chronos_pred', 'nhits_pred', 'model_version']
    """
    rows = _fetch_all_rows_with_retry(
        "predictions",
        "date, ensemble_pred, chronos_pred, nhits_pred, model_version",
        "date",
        filters={"horizon": horizon},
    )
    if not rows:
        return pd.DataFrame(
            columns=["date", "ensemble_pred", "chronos_pred", "nhits_pred", "model_version"]
        ).set_index("date")

    df = pd.DataFrame(rows)
    df["date"] = _as_naive_datetime(df["date"])
    for col in ["ensemble_pred", "chronos_pred", "nhits_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["date", "ensemble_pred"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if latest_historical_date is not None:
        cutoff = pd.Timestamp(latest_historical_date)
        df = df[df["date"] > cutoff].reset_index(drop=True)

    return df.set_index("date")


def fetch_prediction_inventory() -> pd.DataFrame:
    """Return prediction inventory metadata across all horizons."""
    columns = "date, horizon, chronos_pred, nhits_pred, ensemble_pred, model_version"
    rows = _fetch_all_rows_with_retry("predictions", columns, "date")
    if not rows:
        return pd.DataFrame(columns=[c.strip() for c in columns.split(",")])

    df = pd.DataFrame(rows)
    df["date"] = _as_naive_datetime(df["date"])
    for col in ["chronos_pred", "nhits_pred", "ensemble_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["horizon", "date"]).reset_index(drop=True)


def fetch_pipeline_runs() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Fetch (latest_run, last_successful_run) from pipeline_runs.
    """
    client = get_supabase_client()

    def query_latest():
        return (
            client.table("pipeline_runs")
            .select("*")
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )

    def query_last_success():
        return (
            client.table("pipeline_runs")
            .select("*")
            .eq("status", "success")
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )

    resp_latest = execute_with_retry(query_latest, operation_name="query latest pipeline run")
    latest_run = resp_latest.data[0] if resp_latest.data else None

    resp_success = execute_with_retry(query_last_success, operation_name="query last successful pipeline run")
    last_success = resp_success.data[0] if resp_success.data else None

    # Parse timestamps
    for run in (latest_run, last_success):
        if run:
            for field in ["started_at", "finished_at"]:
                if run.get(field):
                    run[field] = pd.to_datetime(run[field])

    return latest_run, last_success


def fetch_model_metadata() -> Optional[Dict[str, Any]]:
    """Fetch active model metadata/metrics from Supabase."""
    client = get_supabase_client()

    for table_name in ["model_metadata", "model_metrics"]:
        try:
            resp = execute_with_retry(
                lambda: client.table(table_name).select("*").order("created_at", desc=True).limit(1).execute(),
                operation_name=f"query {table_name}",
                max_retries=1
            )
            if resp.data:
                return resp.data[0]
        except Exception:
            continue

    return None


def _readiness_payload(state: str, **values) -> Dict[str, Any]:
    """Return a stable readiness payload for every failure and success path."""
    payload = {
        "state": state,
        "error": None,
        "latest_gold_date": None,
        "latest_gold_close": None,
        "pred_generated_date": None,
        "latest_run": None,
        "last_successful_run": None,
        "model_version": None,
        "model_metadata": None,
        "total_predictions": 0,
        "horizons_available": [],
        "prediction_target_dates": {},
        "prediction_rows_valid": False,
    }
    payload.update(values)
    return payload


def _prediction_snapshot(rows: List[Dict[str, Any]], latest_gold_date) -> Dict[str, Any]:
    """Validate prediction targets independently from their generation timestamp."""
    if not rows or latest_gold_date is None:
        return {
            "total_predictions": 0,
            "horizons_available": [],
            "prediction_target_dates": {},
            "prediction_rows_valid": False,
            "prediction_model_versions": [],
        }

    df = pd.DataFrame(rows)
    df["date"] = _as_naive_datetime(df["date"])
    df["horizon"] = df["horizon"].astype(str)
    df["ensemble_pred"] = pd.to_numeric(df["ensemble_pred"], errors="coerce")
    cutoff = pd.Timestamp(latest_gold_date).normalize()
    df = df[
        df["date"].notna()
        & (df["date"] > cutoff)
        & np.isfinite(df["ensemble_pred"])
    ].copy()

    target_dates = {}
    horizons_available = []
    complete = True
    for horizon, length in REQUIRED_FORECAST_HORIZONS.items():
        dates = sorted(df.loc[df["horizon"] == horizon, "date"].drop_duplicates())
        target_dates[horizon] = dates
        expected = list(pd.date_range(cutoff + pd.Timedelta(days=1), periods=length, freq="D"))
        if len(dates) >= length and dates[:length] == expected:
            horizons_available.append(horizon)
        else:
            complete = False

    versions = sorted({str(v) for v in df["model_version"].dropna() if str(v).strip()})
    return {
        "total_predictions": int(len(df)),
        "horizons_available": horizons_available,
        "prediction_target_dates": target_dates,
        "prediction_rows_valid": complete,
        "prediction_model_versions": versions,
    }


def check_system_readiness() -> Dict[str, Any]:
    """
    Orchestrate startup health check and readiness evaluation.

    Evaluates:
    - Connectivity to Supabase
    - Latest actual gold date
    - Latest pipeline run & generation date
    - Prediction counts and horizon availability
    - State: READY, STALE, MISSING, NETWORK_ERROR, CONFIG_ERROR, PIPELINE_RUNNING
    """
    logger.info("Starting system readiness check...")

    try:
        client = get_supabase_client()
    except DatabaseConfigError as ce:
        logger.error("Configuration error detected: %s", ce)
        return _readiness_payload(STATE_CONFIG_ERROR, error=str(ce))

    # Step 1: Health check query
    try:
        test_res = execute_with_retry(
            lambda: client.table("gold_prices").select("date, close").order("date", desc=True).limit(1).execute(),
            max_retries=3,
            operation_name="health_check_gold_prices",
        )
    except DatabaseConnectionError as nwe:
        logger.warning("Database connection error during readiness check: %s", nwe)
        return _readiness_payload(STATE_NETWORK_ERROR, error=str(nwe))
    except Exception as e:
        logger.error("Unexpected error during readiness check: %s", e)
        return _readiness_payload(STATE_NETWORK_ERROR, error=str(e))

    # Extract latest gold price date & price
    latest_gold_date = None
    latest_gold_close = None
    if test_res.data:
        latest_gold_date = pd.to_datetime(test_res.data[0]["date"])
        latest_gold_close = float(test_res.data[0]["close"]) if test_res.data[0].get("close") is not None else None

    try:
        # Step 2: Query pipeline runs
        latest_run, last_successful_run = fetch_pipeline_runs()

        # Step 3: Query model metadata
        model_metadata = fetch_model_metadata()

        # Step 4: Read targets, not only row counts. Target dates and
        # generation timestamps are different concepts and must never be
        # conflated.
        prediction_rows = _fetch_all_rows_with_retry(
            "predictions",
            "date, horizon, ensemble_pred, model_version",
            "date",
        )
    except DatabaseConfigError as exc:
        return _readiness_payload(STATE_CONFIG_ERROR, error=str(exc))
    except (DatabaseConnectionError, DatabaseAPIError) as exc:
        logger.warning("Database error during readiness queries: %s", exc)
        return _readiness_payload(STATE_NETWORK_ERROR, error=str(exc))
    snapshot = _prediction_snapshot(prediction_rows, latest_gold_date)

    # Determine Prediction Generation Date (from last_successful_run, NOT target date!)
    pred_generated_date = None
    if last_successful_run:
        pred_generated_date = last_successful_run.get("finished_at") or last_successful_run.get("started_at")
        if isinstance(pred_generated_date, str):
            pred_generated_date = pd.to_datetime(pred_generated_date)
    elif latest_run and latest_run.get("status") == "success":
        pred_generated_date = latest_run.get("finished_at") or latest_run.get("started_at")
        if isinstance(pred_generated_date, str):
            pred_generated_date = pd.to_datetime(pred_generated_date)

    # Model version comes from the successful execution that generated the
    # predictions, never from a failed newer run.
    model_version = (
        (last_successful_run or {}).get("model_version")
        or (model_metadata or {}).get("version")
    )

    if (
        last_successful_run
        and model_version
        and snapshot["prediction_model_versions"]
        and model_version not in snapshot["prediction_model_versions"]
    ):
        snapshot["prediction_rows_valid"] = False

    if latest_run and latest_run.get("status") in ("running", "in_progress", "queued"):
        state = STATE_PIPELINE_RUNNING
    elif latest_gold_date is None:
        state = STATE_MISSING
    elif not snapshot["prediction_rows_valid"]:
        state = STATE_STALE if snapshot["total_predictions"] else STATE_MISSING
    elif pred_generated_date is None:
        state = STATE_STALE
    else:
        state = STATE_READY

    logger.info(
        "Readiness evaluation: %s (market_date=%s, generated=%s, valid=%s)",
        state,
        latest_gold_date,
        pred_generated_date,
        snapshot["prediction_rows_valid"],
    )
    return _readiness_payload(
        state,
        latest_gold_date=latest_gold_date,
        latest_gold_close=latest_gold_close,
        pred_generated_date=pred_generated_date,
        latest_run=latest_run,
        last_successful_run=last_successful_run,
        model_version=model_version,
        model_metadata=model_metadata,
        **snapshot,
    )
