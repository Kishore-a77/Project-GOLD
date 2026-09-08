"""Lightweight ingestion tests; no Yahoo, Supabase, or model network calls."""

import pandas as pd
import pytest

import services.data_fetch_service as service


def test_incremental_empty_response_is_no_new_data(monkeypatch):
    monkeypatch.setattr(service, "get_latest_date_in_supabase", lambda: pd.Timestamp("2026-09-06").date())
    monkeypatch.setattr(
        service,
        "fetch_gold_data",
        lambda **kwargs: (_ for _ in ()).throw(
            service.NoNewMarketData("no completed candle")
        ),
    )
    assert service.run_data_fetch() == 0


def test_incremental_network_failure_remains_fatal(monkeypatch):
    monkeypatch.setattr(service, "get_latest_date_in_supabase", lambda: pd.Timestamp("2026-09-06").date())
    monkeypatch.setattr(
        service,
        "fetch_gold_data",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("Yahoo network failure")
        ),
    )
    with pytest.raises(RuntimeError, match="Yahoo network failure"):
        service.run_data_fetch()


def test_existing_data_can_continue_without_new_rows(monkeypatch):
    monkeypatch.setattr(service, "get_latest_date_in_supabase", lambda: pd.Timestamp("2099-01-01").date())
    assert service.run_data_fetch() == 0


def test_new_yahoo_rows_are_filtered_to_dates_after_supabase(monkeypatch):
    latest = pd.Timestamp("2026-09-07").date()
    monkeypatch.setattr(service, "get_latest_date_in_supabase", lambda: latest)
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([
            {"date": latest, "close": 2400.0},
            {"date": pd.Timestamp("2026-09-08").date(), "close": 2410.0},
        ])

    monkeypatch.setattr(service, "fetch_gold_data", fake_fetch)
    monkeypatch.setattr(service, "upsert_gold_data", lambda frame: len(frame))
    assert service.run_data_fetch() == 1
    assert captured["start"] == pd.Timestamp("2026-09-05").date()


def test_malformed_yahoo_response_is_fatal(monkeypatch):
    class InvalidTicker:
        def history(self, **kwargs):
            return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(service.yf, "Ticker", lambda symbol: InvalidTicker())
    with pytest.raises(RuntimeError, match="Date column"):
        service.fetch_gold_data(start="2026-09-08", end="2026-09-09")


def test_yahoo_network_failure_is_retried_and_then_fails(monkeypatch):
    attempts = {"count": 0}

    class FailingTicker:
        def history(self, **kwargs):
            attempts["count"] += 1
            raise OSError("temporary network failure")

    monkeypatch.setattr(service.yf, "Ticker", lambda symbol: FailingTicker())
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="after bounded retries"):
        service.fetch_gold_data(start="2026-09-08", end="2026-09-09")
    assert attempts["count"] == service.MAX_RETRIES


def test_empty_yahoo_result_has_explicit_classification(monkeypatch):
    class EmptyTicker:
        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(service.yf, "Ticker", lambda symbol: EmptyTicker())
    with pytest.raises(service.NoNewMarketData):
        service.fetch_gold_data(start="2026-09-08", end="2026-09-09")
