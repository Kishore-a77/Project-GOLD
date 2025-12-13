import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Darts
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.metrics import mape, mae, rmse
from sklearn.preprocessing import MinMaxScaler

# Your Snowflake loader
from app.viewmodels.snowflake_data_loader import load_processed_data

load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "nhits_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PRED_LEN = 30
INPUT_LEN = 90
VAL_LEN = 200
TEST_LEN = 30


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Fix datetime index, fill missing dates, clean values."""

    # convert DATE column to index
    if "DATE" in df.columns:
        df = df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.set_index("DATE")

    # convert index to datetime
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Remove rows where index is NaT (safe)
    df = df[~df.index.isna()]

    # Build complete daily index
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    if "GOLD_CLOSE" not in df.columns:
        raise ValueError("GOLD_CLOSE column missing")

    # Fill forward/backward + interpolation
    df["GOLD_CLOSE"] = (
        df["GOLD_CLOSE"]
        .ffill()
        .bfill()
        .interpolate()
        .ffill()
        .bfill()
    )

    # Set frequency
    df.index.freq = "D"

    return df



def prepare_series(df: pd.DataFrame):
    values = df[["GOLD_CLOSE"]].values.astype(float)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    series = pd.Series(scaled.reshape(-1), index=df.index)
    series = series.asfreq("D")

    ts = TimeSeries.from_series(series)
    return ts, scaler


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
def train_nhits(ts: TimeSeries):

    train_ts = ts[: -(VAL_LEN + TEST_LEN)]
    val_ts = ts[-(VAL_LEN + TEST_LEN) : -TEST_LEN]
    test_ts = ts[-TEST_LEN:]

    print(f"[NHITS] train={len(train_ts)} val={len(val_ts)} test={len(test_ts)}")

    model = NHiTSModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=PRED_LEN,
        n_epochs=40,
        batch_size=64,
        num_stacks=4,
        num_blocks=3,
        num_layers=2,
        layer_widths=256,
        dropout=0.1,
        optimizer_kwargs={"lr": 1e-4},
        random_state=42,
        force_reset=True,
        save_checkpoints=True
    )

    model.fit(train_ts, val_series=val_ts)

    save_path = str(MODEL_DIR / "nhits_model_vfinal")
    model.save(save_path)

    print("[NHITS] Saved:", save_path)
    return model, train_ts, val_ts, test_ts


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
def predict_nhits(model, ts, scaler):

    forecast = model.predict(n=PRED_LEN, series=ts)
    arr = forecast.all_values().reshape(-1, 1)

    inv = scaler.inverse_transform(arr).reshape(-1)

    return {
        "next_day": float(inv[0]),
        "next_week": inv[:7].tolist(),
        "next_month": inv[:30].tolist()
    }


# ---------------------------------------------------------
# MAIN TRAIN + PREDICT
# ---------------------------------------------------------
def main_train_and_predict():
    df = load_processed_data()
    df = prepare_dataframe(df)

    ts, scaler = prepare_series(df)

    model, train_ts, val_ts, test_ts = train_nhits(ts)

    # Evaluate
    pred_scaled = model.predict(n=len(test_ts), series=train_ts[-INPUT_LEN:])
    pred_inv = scaler.inverse_transform(pred_scaled.all_values().reshape(-1, 1))
    actual_inv = scaler.inverse_transform(test_ts.all_values().reshape(-1, 1))

    pred_series = TimeSeries.from_times_and_values(test_ts.time_index, pred_inv.reshape(-1))
    actual_series = TimeSeries.from_times_and_values(test_ts.time_index, actual_inv.reshape(-1))

    mape_v = mape(actual_series, pred_series)
    mae_v = mae(actual_series, pred_series)
    rmse_v = rmse(actual_series, pred_series)

    print(f"[NHITS] test MAPE={mape_v}, MAE={mae_v}, RMSE={rmse_v}")

    # future forecasts
    preds = predict_nhits(model, ts, scaler)

    last_date = df.index.max()
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=PRED_LEN, freq="D")

    pd.DataFrame({"pred": preds["next_month"]}, index=future_index).to_csv(
        MODEL_DIR / "nhits_next_30.csv", index_label="date"
    )

    print("[NHITS] next 30-day forecast saved")
    return preds

# ---------------------------------------------------------
# INFERENCE ONLY (for Ensemble)
# ---------------------------------------------------------
def inference_only():
    """
    Loads the saved NHITS model and returns predictions
    without retraining. This is used by the ensemble script.
    """
    df = load_processed_data()
    df = prepare_dataframe(df)

    ts, scaler = prepare_series(df)

    # Load saved model
    model_path = str(MODEL_DIR / "nhits_model_vfinal")
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"NHITS model not found at {model_path}. "
            "Run main_train_and_predict() once to create it."
        )

    model = NHiTSModel.load(model_path)

    # Predict next 30 days
    preds = predict_nhits(model, ts, scaler)
    return preds

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    preds = main_train_and_predict()
    print("\nFinal NHITS Predictions:\n", preds)
