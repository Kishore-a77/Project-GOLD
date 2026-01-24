"""
DAY 11 – FINAL ENSEMBLE (Chronos + NHITS)
SQLite-compatible version (Snowflake removed)
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

# Correct imports from your actual files
from app.viewmodels.prediction_service import run_chronos_predictions
from app.models.day10_nhits import inference_only as run_nhits_predictions

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DB_PATH = Path("database/gold_data.db")

# Ensemble weights
WEIGHTS = {
    "next_day": (0.7, 0.3),
    "next_week": (0.5, 0.5),
    "next_month": (0.3, 0.7),
}


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def combine(c_vals, n_vals, wc, wn):
    c_vals = np.array(c_vals, dtype=float)
    n_vals = np.array(n_vals, dtype=float)

    L = max(len(c_vals), len(n_vals))
    if len(c_vals) < L:
        c_vals = np.concatenate([c_vals, [np.nan] * (L - len(c_vals))])
    if len(n_vals) < L:
        n_vals = np.concatenate([n_vals, [np.nan] * (L - len(n_vals))])

    result = []
    for c, n in zip(c_vals, n_vals):
        if not np.isnan(c) and not np.isnan(n):
            result.append(wc * c + wn * n)
        elif not np.isnan(c):
            result.append(c)
        elif not np.isnan(n):
            result.append(n)
        else:
            result.append(np.nan)

    return result


def compute_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(true_vals, pred_vals):
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)

    mae = float(np.mean(np.abs(true_vals - pred_vals)))
    rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
    mape = float(compute_mape(true_vals, pred_vals))

    return {"mae": mae, "rmse": rmse, "mape": mape}


# -------------------------------------------------
# SQLITE LOADERS / WRITERS
# -------------------------------------------------
def load_actuals_from_sqlite():
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


def save_ensemble_to_sqlite(forecast_df):
    conn = sqlite3.connect(DB_PATH)

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_forecast (
            date TEXT PRIMARY KEY,
            ensemble_pred REAL
        )
    """)

    forecast_df.reset_index().rename(
        columns={
            "index": "date",
            "ensemble_pred": "ensemble_pred"
        }
    ).to_sql(
        "ensemble_forecast",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def run_ensemble(save_csv=True, save_sqlite=True):
    print("\n[Ensemble] Fetching Chronos predictions...")
    chronos = run_chronos_predictions()

    print("[Ensemble] Fetching NHITS predictions...")
    nhits = run_nhits_predictions()

    ensemble = {}

    # next-day
    w_c, w_n = WEIGHTS["next_day"]
    ensemble["next_day"] = float(
        w_c * chronos["next_day"] + w_n * nhits["next_day"]
    )

    # next-week
    w_c, w_n = WEIGHTS["next_week"]
    ensemble["next_week"] = combine(
        chronos["next_week"], nhits["next_week"], w_c, w_n
    )[:7]

    # next-month
    w_c, w_n = WEIGHTS["next_month"]
    ensemble["next_month"] = combine(
        chronos["next_month"], nhits["next_month"], w_c, w_n
    )[:30]

    # -------------------------------------------------
    # Evaluation (SQLite actuals)
    # -------------------------------------------------
    df = load_actuals_from_sqlite()
    actual = df["GOLD_CLOSE"].values[-30:].tolist()
    pred = ensemble["next_month"][:30]

    metrics = evaluate(actual, pred)

    print("\n[Ensemble] Metrics on last 30 days:")
    print(metrics)

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------
    last_date = df.index[-1]
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")

    forecast_df = pd.DataFrame(
        {"ensemble_pred": ensemble["next_month"]},
        index=dates
    )

    if save_csv:
        out_dir = Path("models/ensemble")
        out_dir.mkdir(parents=True, exist_ok=True)
        forecast_df.reset_index().rename(
            columns={"index": "date"}
        ).to_csv(out_dir / "ensemble_next_30.csv", index=False)
        print(f"[Ensemble] CSV saved -> {out_dir / 'ensemble_next_30.csv'}")

    if save_sqlite:
        save_ensemble_to_sqlite(forecast_df)
        print("[Ensemble] Saved to SQLite table: ensemble_forecast")

    return {"predictions": ensemble, "metrics": metrics}


if __name__ == "__main__":
    run_ensemble()
