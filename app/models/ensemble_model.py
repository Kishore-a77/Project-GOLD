"""
DAY 11 – FINAL ENSEMBLE (Chronos + NHITS)
SQLite-compatible version (FIXED VERSION)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from app.db.supabase_client import engine
from sqlalchemy import text

# Correct imports
from app.viewmodels.prediction_service import run_chronos_predictions
from app.models.day10_nhits import inference_only as run_nhits_predictions

# -------------------------------------------------
# CONFIG
# -------------------------------------------------


# Ensemble weights
WEIGHTS = {
    "next_day": (0.7, 0.3),
    "next_week": (0.6, 0.4),
    "next_month": (0.5, 0.5),
}


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def combine(c_vals, n_vals, wc, wn):
    """
    Combines Chronos + NHITS predictions safely.
    Removes NaNs and ensures stable output.
    """

    c_vals = np.array(c_vals, dtype=float)
    n_vals = np.array(n_vals, dtype=float)

    L = max(len(c_vals), len(n_vals))

    if len(c_vals) < L:
        c_vals = np.concatenate(
            [c_vals, np.full(L - len(c_vals), np.nan)]
        )

    if len(n_vals) < L:
        n_vals = np.concatenate(
            [n_vals, np.full(L - len(n_vals), np.nan)]
        )

    result = []

    for c, n in zip(c_vals, n_vals):

        if not np.isnan(c) and not np.isnan(n):
            value = wc * c + wn * n

        elif not np.isnan(c):
            value = c

        elif not np.isnan(n):
            value = n

        else:
            value = np.nan

        result.append(value)

    # forward-fill NaNs if any
    result = pd.Series(result).ffill().bfill().tolist()

    return result


def compute_mape(y_true, y_pred):

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0

    if mask.sum() == 0:
        return float("nan")

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def evaluate(true_vals, pred_vals):

    true_vals = np.array(true_vals, dtype=float)
    pred_vals = np.array(pred_vals, dtype=float)

    mae = float(np.mean(np.abs(true_vals - pred_vals)))

    rmse = float(
        np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    )

    mape = float(compute_mape(true_vals, pred_vals))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


# -------------------------------------------------
# SUPABASE LOADERS / WRITERS
# -------------------------------------------------
def load_actuals_from_supabase():
    df = pd.read_sql('SELECT date, gold_close AS "GOLD_CLOSE" FROM features ORDER BY date', engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")

def save_ensemble_to_supabase(forecast_df):
    # Ensure proper format
    out = forecast_df.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    
    # Create table if it doesn't exist, clear old predictions, and insert new
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ensemble_forecast (
                date TIMESTAMP PRIMARY KEY,
                ensemble_pred DOUBLE PRECISION
            )
        """))
        conn.execute(text("TRUNCATE TABLE ensemble_forecast"))
        conn.commit()
    
    out.to_sql("ensemble_forecast", engine, if_exists="append", index=False)
    print("[Ensemble] Saved to Supabase table: ensemble_forecast")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def run_ensemble(save_csv=True, save_sqlite=True):

    print("\n[Ensemble] Fetching Chronos predictions...")

    chronos = run_chronos_predictions()

    print("[Ensemble] Fetching NHITS predictions...")

    nhits = run_nhits_predictions()

    ensemble = {}

    # -------------------------------------------------
    # NEXT DAY
    # -------------------------------------------------
    w_c, w_n = WEIGHTS["next_day"]

    ensemble["next_day"] = float(
        w_c * chronos["next_day"] +
        w_n * nhits["next_day"]
    )

    # -------------------------------------------------
    # NEXT WEEK
    # -------------------------------------------------
    w_c, w_n = WEIGHTS["next_week"]

    ensemble["next_week"] = combine(
        chronos["next_week"],
        nhits["next_week"],
        w_c,
        w_n
    )[:7]

    # -------------------------------------------------
    # NEXT MONTH
    # -------------------------------------------------
    w_c, w_n = WEIGHTS["next_month"]

    ensemble["next_month"] = combine(
        chronos["next_month"],
        nhits["next_month"],
        w_c,
        w_n
    )[:30]

    # -------------------------------------------------
    # LOAD ACTUAL DATA
    # -------------------------------------------------
    df = load_actuals_from_supabase()

    actual = df["GOLD_CLOSE"].values[-30:]

    pred = np.array(
        ensemble["next_month"][:30],
        dtype=float
    )

    # ensure same length
    min_len = min(len(actual), len(pred))

    actual = actual[-min_len:]
    pred = pred[-min_len:]

    # remove NaNs
    mask = ~np.isnan(pred)

    actual = actual[mask]
    pred = pred[mask]

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    metrics = evaluate(actual, pred)

    print("\n[Ensemble] Metrics:")
    print(metrics)

    # -------------------------------------------------
    # FORECAST DATES
    # -------------------------------------------------
    last_date = pd.to_datetime(df.index[-1])

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(ensemble["next_month"]),
        freq="D"
    )

    # -------------------------------------------------
    # FORECAST DATAFRAME
    # -------------------------------------------------
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "ensemble_pred": ensemble["next_month"]
    })

    # clean dataframe
    forecast_df = forecast_df.dropna()

    forecast_df["date"] = pd.to_datetime(
        forecast_df["date"]
    )

    forecast_df = forecast_df.sort_values("date")

    forecast_df = forecast_df.reset_index(drop=True)

    # -------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------
    if save_csv:

        out_dir = Path("models/ensemble")

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        csv_path = out_dir / "ensemble_next_30.csv"

        forecast_df.to_csv(
            csv_path,
            index=False
        )

        print(f"\n[Ensemble] CSV saved -> {csv_path}")

    # -------------------------------------------------
    # SAVE SUPABASE
    # -------------------------------------------------
    if save_sqlite:

        save_ensemble_to_supabase(
            forecast_df.set_index("date")
        )

    print("\n[Ensemble] Forecast Preview:")
    print(forecast_df.head())

    return {
        "predictions": ensemble,
        "metrics": metrics
    }


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":

    run_ensemble()