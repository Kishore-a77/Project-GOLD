"""Offline tests for completed-target accuracy evaluation."""

from services.accuracy_service import build_evaluations, calculate_metrics


def test_future_targets_are_not_evaluated():
    rows = [{
        "prediction_date": "2026-09-01",
        "date": "2026-09-10",
        "horizon": "7d",
        "chronos_pred": 101.0,
        "nhits_pred": 99.0,
        "ensemble_pred": 100.0,
        "model_version": "test",
    }]
    evaluations = build_evaluations(rows, {"2026-09-09": 98.0})
    assert evaluations == []


def test_completed_targets_produce_model_metrics():
    rows = [{
        "prediction_date": "2026-09-01",
        "date": "2026-09-02",
        "horizon": "1d",
        "chronos_pred": 101.0,
        "nhits_pred": 99.0,
        "ensemble_pred": 100.0,
        "model_version": "test",
    }]
    evaluations = build_evaluations(rows, {"2026-09-02": 100.0})
    metrics = calculate_metrics(evaluations)
    assert set(metrics) == {"Chronos-T5", "N-HiTS", "Ensemble"}
    assert metrics["Ensemble"]["mae"] == 0.0
    assert metrics["Chronos-T5"]["mae"] == 1.0
