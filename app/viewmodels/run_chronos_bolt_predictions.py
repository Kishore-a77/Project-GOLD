from app.models.chronos_bolt_model import ChronosBoltModel
from app.viewmodels.snowflake_data_loader import load_processed_data


def run_chronos_bolt_predictions():
    """
    Runs Chronos-Bolt predictions for next 30 days.
    """
    df = load_processed_data()
    series_values = df["GOLD_CLOSE"].values.tolist()

    model = ChronosBoltModel()

    raw_preds = model.predict_next(series_values, prediction_length=30)

    return {
        "next_day": float(raw_preds[0]),
        "next_week": raw_preds[:7],
        "next_month": raw_preds[:30]
    }
