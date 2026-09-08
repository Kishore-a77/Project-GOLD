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


def test_empty_yahoo_result_has_explicit_classification(monkeypatch):
    class EmptyTicker:
        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(service.yf, "Ticker", lambda symbol: EmptyTicker())
    with pytest.raises(service.NoNewMarketData):
        service.fetch_gold_data(start="2026-09-08", end="2026-09-09")
