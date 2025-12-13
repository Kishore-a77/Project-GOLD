"""
DAY 11 – FINAL ENSEMBLE (Chronos + NHITS)
Fully fixed version — correct imports — no autodetection — no errors.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Correct imports from your actual files:
from app.viewmodels.prediction_service import run_chronos_predictions            # Chronos :contentReference[oaicite:0]{index=0}
from app.models.day10_nhits import inference_only as run_nhits_predictions       # NHITS  :contentReference[oaicite:1]{index=1}
from app.viewmodels.snowflake_data_loader import load_processed_data

# Ensemble weights
WEIGHTS = {
    "next_day": (0.7, 0.3),
    "next_week": (0.5, 0.5),
    "next_month": (0.3, 0.7),
}

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


# Metrics
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


# ---- MAIN ----
def run_ensemble(save_csv=True):
    print("\n[Ensemble] Fetching Chronos predictions...")
    chronos = run_chronos_predictions()

    print("[Ensemble] Fetching NHITS predictions...")
    nhits = run_nhits_predictions()

    ensemble = {}

    # next-day
    w_c, w_n = WEIGHTS["next_day"]
    ensemble["next_day"] = float(w_c * chronos["next_day"] + w_n * nhits["next_day"])

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

    # Evaluation
    df = load_processed_data()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    actual = df["GOLD_CLOSE"].values[-30:].tolist()
    pred = ensemble["next_month"][:30]

    metrics = evaluate(actual, pred)

    print("\n[Ensemble] Metrics on last 30 days:")
    print(metrics)

    # Save
    if save_csv:
        out_dir = Path("models/ensemble")
        out_dir.mkdir(parents=True, exist_ok=True)

        last_date = df.index[-1]
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")

        out = pd.DataFrame({"date": dates, "ensemble_pred": ensemble["next_month"]})
        out.to_csv(out_dir / "ensemble_next_30.csv", index=False)

        print(f"[Ensemble] Saved -> {out_dir / 'ensemble_next_30.csv'}")

    return {"predictions": ensemble, "metrics": metrics}


if __name__ == "__main__":
    run_ensemble()
