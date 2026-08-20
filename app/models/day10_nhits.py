from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from app.db.supabase_client import get_supabase_client

# Darts
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.metrics import mape, mae, rmse
from sklearn.preprocessing import MinMaxScaler

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

MIN_REQUIRED_ROWS = INPUT_LEN + VAL_LEN + TEST_LEN + 5


# ---------------------------------------------------------
# SUPABASE DATA LOADER
# ---------------------------------------------------------
def load_processed_data_supabase() -> pd.DataFrame:
    client = get_supabase_client(use_service_role=True)
    rows = []
    start = 0
    page_size = 1000
    while True:
        end = start + page_size - 1
        response = (
            client.table("gold_features")
            .select("date, close")
            .order("date")
            .range(start, end)
            .execute()
        )
        page = response.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows found in `gold_features`. Run feature engineering first.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.rename(columns={"close": "GOLD_CLOSE"})
    df = df[["date", "GOLD_CLOSE"]]
    return df


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure continuous daily datetime index and clean values."""

    if df.empty:
        raise ValueError("Input dataframe is empty. Cannot train NHITS.")

    df = df.copy()
    df = df.set_index("date").sort_index()

    if df.index.min() is pd.NaT or df.index.max() is pd.NaT:
        raise ValueError("Invalid datetime index after parsing dates.")

    # Reindex to full daily range
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    if "GOLD_CLOSE" not in df.columns:
        raise ValueError("GOLD_CLOSE column missing")

    # Fill gaps safely
    df["GOLD_CLOSE"] = (
        df["GOLD_CLOSE"]
        .ffill()
        .bfill()
        .interpolate()
        .ffill()
        .bfill()
    )

    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(
            f"Not enough data for NHITS. "
            f"Required ≥ {MIN_REQUIRED_ROWS}, found {len(df)}"
        )

    df.index.freq = "D"
    return df


def prepare_series(df: pd.DataFrame):
    values = df[["GOLD_CLOSE"]].values.astype(float)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    series = pd.Series(scaled.reshape(-1), index=df.index).asfreq("D")
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
        save_checkpoints=True,
        pl_trainer_kwargs={
            "enable_progress_bar": True,
            "deterministic": True
        }
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
    df = load_processed_data_supabase()
    df = prepare_dataframe(df)

    ts, scaler = prepare_series(df)

    model, train_ts, val_ts, test_ts = train_nhits(ts)

    # Evaluate on test window
    pred_scaled = model.predict(n=len(test_ts), series=ts)
    pred_inv = scaler.inverse_transform(pred_scaled.all_values().reshape(-1, 1))
    actual_inv = scaler.inverse_transform(test_ts.all_values().reshape(-1, 1))

    pred_series = TimeSeries.from_times_and_values(
        test_ts.time_index, pred_inv.reshape(-1)
    )
    actual_series = TimeSeries.from_times_and_values(
        test_ts.time_index, actual_inv.reshape(-1)
    )

    print(
        f"[NHITS] test "
        f"MAPE={mape(actual_series, pred_series):.4f}, "
        f"MAE={mae(actual_series, pred_series):.4f}, "
        f"RMSE={rmse(actual_series, pred_series):.4f}"
    )

    preds = predict_nhits(model, ts, scaler)

    last_date = df.index.max()
    future_index = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=PRED_LEN,
        freq="D"
    )

    pd.DataFrame(
        {"pred": preds["next_month"]},
        index=future_index
    ).to_csv(
        MODEL_DIR / "nhits_next_30.csv",
        index_label="date"
    )

    print("[NHITS] next 30-day forecast saved")
    return preds


# ---------------------------------------------------------
# INFERENCE ONLY (USED BY ENSEMBLE)
# ---------------------------------------------------------
def inference_only():
    df = load_processed_data_supabase()
    df = prepare_dataframe(df)

    ts, scaler = prepare_series(df)

    model_path = MODEL_DIR / "nhits_model_vfinal"
    if not model_path.exists():
        raise FileNotFoundError(
            f"NHITS model not found at {model_path}. "
            "Run main_train_and_predict() once."
        )

    model = NHiTSModel.load(str(model_path))
    return predict_nhits(model, ts, scaler)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    preds = main_train_and_predict()
    print("\nFinal NHITS Predictions:\n", preds)
