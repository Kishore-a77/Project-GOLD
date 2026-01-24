"""
Prediction Service — Chronos-T5
SQLite-based version (Snowflake fully removed)
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from app.models.chronos_t5_model import ChronosT5Model

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DB_PATH = Path("database/gold_data.db")


# -------------------------------------------------
# LOAD DATA FROM SQLITE
# -------------------------------------------------
def load_data_from_sqlite():
    """
    Loads processed gold price data from SQLite.
    Returns a pandas DataFrame indexed by date.
    """
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


# -------------------------------------------------
# CHRONOS PREDICTION ENTRY POINT
# -------------------------------------------------
def run_chronos_predictions():
    """
    Runs Chronos-T5 predictions for next 1, 7, and 30 days.
    Safely flattens all Chronos output formats.
    """

    # Load data (SQLite instead of Snowflake)
    df = load_data_from_sqlite()

    # Extract univariate series
    series_values = df["GOLD_CLOSE"].values.tolist()

    # Initialize model
    model = ChronosT5Model()

    # Get raw predictions (could be many shapes)
    raw_predictions = model.predict_next(
        series_values,
        prediction_length=30
    )

    # --- SAFELY FLATTEN ALL OUTPUTS ---
    raw_predictions = np.array(raw_predictions).astype(float).flatten()

    # Guaranteed shape = (30,)
    next_day = float(raw_predictions[0])
    next_week = raw_predictions[:7].tolist()
    next_month = raw_predictions[:30].tolist()

    return {
        "next_day": next_day,
        "next_week": next_week,
        "next_month": next_month
    }
