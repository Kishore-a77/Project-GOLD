"""Accuracy evaluation primitives for completed forecast targets.

This module deliberately evaluates only predictions whose target actual exists.
It does not estimate future accuracy, fill missing actuals, or alter ensemble
weights. Persistence can be added later without changing the calculation API.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import sqrt
from typing import Any, Iterable, Mapping, Optional

import numpy as np


MODEL_COLUMNS = {
    "Chronos-T5": "chronos_pred",
    "N-HiTS": "nhits_pred",
    "Ensemble": "ensemble_pred",
}


@dataclass(frozen=True)
class PredictionEvaluation:
    prediction_date: Any
    target_date: Any
    horizon: str
    predicted_price: float
    actual_price: float
    absolute_error: float
    percentage_error: float
    model: str
    model_version: Optional[str] = None

    def as_dict(self):
        return asdict(self)


def _normalise_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return np.datetime64(value, "D").astype(object)


def build_evaluations(
    predictions: Iterable[Mapping[str, Any]],
    actuals: Mapping[Any, float],
) -> list[PredictionEvaluation]:
    """Build evaluations only for rows with a known, finite actual value."""
    actual_by_date = {_normalise_date(k): float(v) for k, v in actuals.items()}
    evaluations = []

    for row in predictions:
        target_date = _normalise_date(row["date"] if "date" in row else row["target_date"])
        actual = actual_by_date.get(target_date)
        if actual is None or not np.isfinite(actual):
            continue

        for model, column in MODEL_COLUMNS.items():
            predicted = row.get(column)
            if predicted is None or not np.isfinite(float(predicted)):
                continue
            predicted = float(predicted)
            absolute_error = abs(predicted - actual)
            percentage_error = (absolute_error / abs(actual) * 100.0) if actual else 0.0
            evaluations.append(
                PredictionEvaluation(
                    prediction_date=row.get("prediction_date"),
                    target_date=target_date,
                    horizon=str(row.get("horizon", "")),
                    predicted_price=predicted,
                    actual_price=actual,
                    absolute_error=absolute_error,
                    percentage_error=percentage_error,
                    model=model,
                    model_version=row.get("model_version"),
                )
            )
    return evaluations


def calculate_metrics(evaluations: Iterable[PredictionEvaluation]):
    """Return MAE, RMSE, and MAPE grouped by model."""
    grouped = {}
    for evaluation in evaluations:
        grouped.setdefault(evaluation.model, []).append(evaluation)

    metrics = {}
    for model, rows in grouped.items():
        errors = np.asarray([r.absolute_error for r in rows], dtype=float)
        pct_errors = np.asarray([r.percentage_error for r in rows], dtype=float)
        metrics[model] = {
            "count": len(rows),
            "mae": float(errors.mean()),
            "rmse": float(sqrt(np.mean(errors ** 2))),
            "mape": float(pct_errors.mean()),
        }
    return metrics
