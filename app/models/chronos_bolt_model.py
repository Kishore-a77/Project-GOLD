import torch
from chronos import ChronosPipeline


class ChronosBoltModel:
    """
    Chronos-Bolt model for medium and long-term forecasting.
    Works on Windows and Python 3.11.
    """

    def __init__(self):
        model_id = "amazon/chronos-bolt-base"
        self.pipeline = ChronosPipeline.from_pretrained(model_id)

    def predict_next(self, series_values, prediction_length=30):
        """
        series_values: list or numpy array of historical values
        prediction_length: number of future steps
        """
        if not isinstance(series_values, (list, tuple)):
            series_values = series_values.tolist()

        preds = self.pipeline.predict(
            context=series_values,
            prediction_length=prediction_length,
            num_samples=20  # ensemble sampling improves stability
        )

        return preds.squeeze().tolist()
