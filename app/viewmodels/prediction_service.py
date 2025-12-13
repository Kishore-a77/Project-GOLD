from app.models.chronos_t5_model import ChronosT5Model
from app.viewmodels.snowflake_data_loader import load_processed_data
import numpy as np

def run_chronos_predictions():
    """
    Runs Chronos-T5 predictions for next 1, 7, and 30 days.
    Safely flattens all Chronos output formats.
    """
    # Load data
    df = load_processed_data()

    # Extract univariate series
    series_values = df['GOLD_CLOSE'].values.tolist()

    # Initialize model
    model = ChronosT5Model()

    # Get raw predictions (could be many shapes)
    raw_predictions = model.predict_next(series_values, prediction_length=30)

    # --- SAFELY FLATTEN ALL OUTPUTS ---
    raw_predictions = np.array(raw_predictions).astype(float).flatten()

    # Now guaranteed shape = (30,)
    next_day = float(raw_predictions[0])
    next_week = raw_predictions[:7].tolist()
    next_month = raw_predictions[:30].tolist()

    return {
        "next_day": next_day,
        "next_week": next_week,
        "next_month": next_month
    }
