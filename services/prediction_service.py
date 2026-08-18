"""
Prediction Service — Chronos-T5 + N-HiTS inference, ensemble combination,
and idempotent persistence to the Supabase `predictions` table.

This module reuses the existing model implementations:
  * Chronos forecasting : app.models.chronos_t5_model.ChronosT5Model
  * N-HiTS forecasting  : app.models.day10_nhits.inference_only (INFERENCE ONLY, no training)
  * Ensemble combination: app.models.ensemble_model.combine + ensemble_model.WEIGHTS

It does NOT retrain models and does NOT duplicate model/ensemble logic.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from supabase import create_client

# Ensure project root is importable so `app.*` packages resolve.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prediction_service")

# NOTE: The heavy model modules (torch/darts/chronos) are imported lazily inside
# the functions that need them. This keeps this module importable in lightweight
# environments (e.g. tests, or a dashboard-only deploy) without pulling in torch/darts.

# Credentials are read from the environment (never hard-coded).
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_VERSION = "chronos-t5-small + nhits_vfinal"


# -------------------------------------------------
# LOAD MODEL-READY DATA
# -------------------------------------------------
def fetch_gold_series():
    """Load the gold close series (model-ready data) from Supabase."""
    response = supabase.table("gold_prices").select("date, close").order("date").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        raise RuntimeError("No rows found in `gold_prices`. Run data ingestion first.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        raise RuntimeError("`gold_prices` contains no usable close values.")
    return df


# -------------------------------------------------
# CHRONOS-T5 INFERENCE
# -------------------------------------------------
def run_chronos_prediction(series_values, horizon_days=30):
    """Run Chronos-T5 inference for the next `horizon_days` steps (inference only)."""
    from app.models.chronos_t5_model import ChronosT5Model
    try:
        model = ChronosT5Model()
        raw = model.predict_next(series_values, prediction_length=horizon_days)
    except Exception as e:
        raise RuntimeError(f"Chronos-T5 inference failed to load/run: {e}") from e
    arr = np.array(raw, dtype=float).flatten()
    if len(arr) < horizon_days:
        raise RuntimeError(
            f"Chronos returned only {len(arr)} values, expected >= {horizon_days}."
        )
    arr = arr[:horizon_days]
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "Chronos produced non-finite (NaN/inf) predictions; refusing to proceed."
        )
    return arr.tolist()


# -------------------------------------------------
# N-HiTS INFERENCE
# -------------------------------------------------
def run_nhits_prediction(horizon_days=30):
    """Run N-HiTS inference using the existing trained artifact (inference only)."""
    from app.models.day10_nhits import inference_only as run_nhits_inference
    try:
        preds = run_nhits_inference()  # dict: next_day / next_week / next_month
    except Exception as e:
        raise RuntimeError(f"N-HiTS inference failed to load/run: {e}") from e
    series = list(preds["next_month"])
    if len(series) < horizon_days:
        series = series + [series[-1]] * (horizon_days - len(series))
    series = [float(x) for x in series[:horizon_days]]
    if not np.all(np.isfinite(series)):
        raise ValueError(
            "N-HiTS produced non-finite (NaN/inf) predictions; refusing to proceed."
        )
    return series


# -------------------------------------------------
# ENSEMBLE (reuses app.models.ensemble_model.combine + WEIGHTS)
# -------------------------------------------------
def compute_ensemble(chronos_30, nhits_30):
    # Lightweight import (no torch/darts/psycopg2) so this module stays testable.
    from services.ensemble_logic import combine, WEIGHTS

    chronos_30 = list(chronos_30)[:30]
    nhits_30 = list(nhits_30)[:30]
    if len(chronos_30) < 30 or len(nhits_30) < 30:
        raise ValueError("Forecast length < 30; cannot build ensemble.")
    if not (np.all(np.isfinite(chronos_30)) and np.all(np.isfinite(nhits_30))):
        raise ValueError(
            "Non-finite values detected in model forecasts; refusing to build ensemble."
        )

    w_c, w_n = WEIGHTS["next_day"]
    ensemble = {
        "next_day": float(w_c * chronos_30[0] + w_n * nhits_30[0]),
        "next_week": combine(chronos_30[:7], nhits_30[:7], *WEIGHTS["next_week"])[:7],
        "next_month": combine(chronos_30[:30], nhits_30[:30], *WEIGHTS["next_month"])[:30],
    }
    return ensemble


# -------------------------------------------------
# PERSISTENCE (idempotent upsert on (date, horizon))
# -------------------------------------------------
def save_predictions(ensemble, chronos_30, nhits_30, last_date, model_version=MODEL_VERSION):
    """Upsert next-day / next-week / next-month predictions to Supabase."""
    # --- GUARD: never persist invalid/empty predictions ---
    # This protects previously stored VALID predictions from being overwritten
    # by a partial/corrupted run. We validate before any write.
    try:
        ch_all = [float(x) for x in chronos_30[:30]]
        nh_all = [float(x) for x in nhits_30[:30]]
        ens_vals = [float(ensemble["next_day"])]
        ens_vals += [float(x) for x in ensemble["next_week"]]
        ens_vals += [float(x) for x in ensemble["next_month"]]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Model outputs contain non-numeric values: {e}")
    if not (np.all(np.isfinite(ch_all)) and np.all(np.isfinite(nh_all))):
        raise ValueError(
            "Model outputs contain non-finite (NaN/inf) values; "
            "refusing to persist. Previous valid predictions are preserved."
        )
    if not np.all(np.isfinite(ens_vals)):
        raise ValueError(
            "Refusing to persist non-finite (NaN/inf) ensemble predictions. "
            "Previous valid predictions are preserved."
        )
    if len(ensemble["next_week"]) < 7 or len(ensemble["next_month"]) < 30:
        raise ValueError(
            "Ensemble horizons are incomplete; refusing to persist partial predictions."
        )

    last_date = pd.to_datetime(last_date)

    specs = [
        ("1d", ensemble["next_day"], [chronos_30[0]], [nhits_30[0]]),
        ("7d", ensemble["next_week"], chronos_30[:7], nhits_30[:7]),
        ("30d", ensemble["next_month"], chronos_30[:30], nhits_30[:30]),
    ]

    records = []
    for horizon, ens, ch, nh in specs:
        ens = [ens] if isinstance(ens, (int, float)) else list(ens)
        ch = list(ch)
        nh = list(nh)
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(len(ens))]
        for i, d in enumerate(future_dates):
            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "horizon": horizon,
                "chronos_pred": round(float(ch[i]), 4) if i < len(ch) else None,
                "nhits_pred": round(float(nh[i]), 4) if i < len(nh) else None,
                "ensemble_pred": round(float(ens[i]), 4),
                "model_version": model_version,
            })

    if not records:
        raise RuntimeError("No prediction records were produced.")

    supabase.table("predictions").upsert(records).execute()
    return len(records)


# -------------------------------------------------
# RUN STATUS LEDGER
# -------------------------------------------------
def _sanitize(msg):
    """Strip secret values (DB password, API keys) before logging/recording."""
    for secret in (
        os.getenv("DATABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
        os.getenv("SUPABASE_SERVICE_KEY", ""),
    ):
        if secret:
            msg = msg.replace(secret, "***")
    return msg


def record_run_status(started_at, success, error=None, model_version=MODEL_VERSION,
                      records_processed=None, predictions_generated=None):
    row = {
        "run_date": datetime.now().date().isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "status": "success" if success else "failed",
        "error": (_sanitize(str(error))[:2000] if error else None),
        "model_version": model_version,
        "records_processed": records_processed,
        "predictions_generated": predictions_generated,
    }
    supabase.table("pipeline_runs").upsert(row).execute()


# -------------------------------------------------
# DRY-RUN MOCK FORECAST (structure test only)
# -------------------------------------------------
def mock_forecast(series, horizon_days=30):
    last = float(series[-1])
    window = series[-30:]
    drift = (window[-1] - window[0]) / len(window) if len(window) > 1 else 0.0
    return [last + drift * (i + 1) for i in range(horizon_days)]


# -------------------------------------------------
# ORCHESTRATED PREDICTION STAGE (stages 5-9)
# -------------------------------------------------
def run_prediction_pipeline(dry_run=False, model_version=MODEL_VERSION):
    """
    Stages:
      5. Load model-ready data from Supabase
      6. Chronos-T5 inference
      7. N-HiTS inference
      8. Ensemble combination
      9. Save predictions to Supabase

    Returns (ensemble, chronos_30, nhits_30, predictions_generated).
    """
    log.info("  STAGE 5: Load model-ready data from Supabase `gold_prices`")
    try:
        df = fetch_gold_series()
        series = df["close"].tolist()
        last_date = df["date"].max()
        log.info("  Loaded %d gold rows; last_date=%s", len(df), last_date.date())
    except Exception as e:
        if dry_run:
            log.warning("  [dry-run] could not load real data (%s); using synthetic series", e)
            series = [1800.0 + 10.0 * i + 5.0 * ((i * 7) % 13) for i in range(400)]
            last_date = datetime.now()
        else:
            raise

    if dry_run:
        log.info("  STAGE 6: Chronos-T5 inference [dry-run MOCK]")
        chronos_30 = mock_forecast(series, 30)
        log.info("  STAGE 7: N-HiTS inference [dry-run MOCK]")
        nhits_30 = mock_forecast(series, 30)
    else:
        log.info("  STAGE 6: Chronos-T5 inference")
        chronos_30 = run_chronos_prediction(series, 30)
        log.info("  STAGE 7: N-HiTS inference")
        nhits_30 = run_nhits_prediction(30)

    log.info("  STAGE 8: Ensemble (Chronos-T5 + N-HiTS)")
    ensemble = compute_ensemble(chronos_30, nhits_30)

    log.info("  STAGE 9: Save predictions to Supabase `predictions`")
    if dry_run:
        log.info("  [dry-run] predicted next_day=%.2f (NOT written to Supabase)",
                 ensemble["next_day"])
        return ensemble, chronos_30, nhits_30, 0

    n = save_predictions(ensemble, chronos_30, nhits_30, last_date, model_version)
    log.info("  Saved %d prediction rows (idempotent upsert on date+horizon)", n)
    return ensemble, chronos_30, nhits_30, n


if __name__ == "__main__":
    ensemble, _, _, _ = run_prediction_pipeline(dry_run=False)
    print("Next day:", round(ensemble["next_day"], 2))
    print("Next week[0]:", round(ensemble["next_week"][0], 2))
    print("Next month[0]:", round(ensemble["next_month"][0], 2))
