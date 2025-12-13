import numpy as np
import torch
from chronos import ChronosPipeline

class ChronosT5Model:
    def __init__(self):
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small"
        )

    def predict_next(self, series_values, prediction_length=30):
        # Convert list to tensor
        series_tensor = torch.tensor(series_values, dtype=torch.float32)

        # Add batch dimension  (1, T)
        series_tensor = series_tensor.unsqueeze(0)

        # ---- IMPORTANT ----
        # Your Chronos version accepts the context ONLY as a POSITIONAL ARGUMENT
        forecast = self.model.predict(
            series_tensor,                 # <-- correct for your version
            prediction_length=prediction_length
        )

        # Remove batch dimension
        forecast = forecast[0].cpu().detach().numpy().flatten()

        return forecast
