"""Offline readiness-state tests using controlled database responses."""

from types import SimpleNamespace

import pandas as pd

import app.services.database_service as service


def _rows(base, horizons=("1d", "7d", "30d")):
    lengths = {"1d": 1, "7d": 7, "30d": 30}
    return [
        {
            "date": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "horizon": horizon,
            "ensemble_pred": 100.0 + i,
            "model_version": "test",
        }
        for horizon in horizons
        for i in range(1, lengths[horizon] + 1)
    ]


def _patch_database(monkeypatch, rows, latest_run=None, last_success=None):
    base = pd.Timestamp("2026-09-11")
    monkeypatch.setattr(service, "get_supabase_client", lambda: object())
    monkeypatch.setattr(
        service,
        "execute_with_retry",
        lambda query, **kwargs: SimpleNamespace(data=[{"date": str(base.date()), "close": 100.0}]),
    )
    monkeypatch.setattr(service, "_fetch_all_rows_with_retry", lambda *args, **kwargs: rows)
    monkeypatch.setattr(service, "fetch_pipeline_runs", lambda: (latest_run, last_success))
    monkeypatch.setattr(service, "fetch_model_metadata", lambda: None)


def test_ready_state(monkeypatch):
    run = {"status": "success", "finished_at": pd.Timestamp("2026-09-11 08:00")}
    _patch_database(monkeypatch, _rows(pd.Timestamp("2026-09-11")), run, run)
    assert service.check_system_readiness()["state"] == service.STATE_READY


def test_missing_and_stale_states(monkeypatch):
    run = {"status": "success", "finished_at": pd.Timestamp("2026-09-11 08:00")}
    _patch_database(monkeypatch, [], run, run)
    assert service.check_system_readiness()["state"] == service.STATE_MISSING

    _patch_database(monkeypatch, _rows(pd.Timestamp("2026-09-11"), ("1d",)), run, run)
    assert service.check_system_readiness()["state"] == service.STATE_STALE


def test_pipeline_running_state(monkeypatch):
    run = {"status": "running", "started_at": pd.Timestamp("2026-09-12 08:00")}
    _patch_database(monkeypatch, [], run, None)
    assert service.check_system_readiness()["state"] == service.STATE_PIPELINE_RUNNING


def test_network_error_state(monkeypatch):
    monkeypatch.setattr(service, "get_supabase_client", lambda: object())

    def fail(*args, **kwargs):
        raise service.DatabaseConnectionError("network unavailable")

    monkeypatch.setattr(service, "execute_with_retry", fail)
    assert service.check_system_readiness()["state"] == service.STATE_NETWORK_ERROR
