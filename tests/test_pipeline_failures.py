"""
Controlled failure tests for the daily GOLD prediction pipeline.

These tests simulate each failure scenario required by the robustness review
without needing torch / darts / psycopg2 / a live network. Heavy / external
dependencies are replaced with in-process fakes:

  * yfinance      -> FakeTicker (no network)
  * app.db.supabase_client -> fake module (no psycopg2/sqlalchemy)
  * Supabase client        -> FakeSupabase (in-memory, idempotent upsert)

Run with:  pytest tests/test_pipeline_failures.py -v
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# Fake Supabase
# --------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeTable:
    def __init__(self, store, name, fail_tables=None):
        self._store = store
        self._name = name
        self._fail_tables = fail_tables or set()

    def select(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def single(self, *a, **k):
        return self

    def execute(self):
        if self._name in self._fail_tables:
            raise RuntimeError("Supabase Unavailable (simulated)")
        return FakeResponse(self._store.get(self._name, []))

    def upsert(self, records):
        if self._name in self._fail_tables:
            raise RuntimeError("Supabase Unavailable (simulated)")
        tbl = self._store.setdefault(self._name, [])
        recs = records if isinstance(records, list) else [records]
        if self._name == "predictions":
            idx = {(r["date"], r["horizon"]): i for i, r in enumerate(tbl)}
            for r in recs:
                k = (r["date"], r["horizon"])
                if k in idx:
                    tbl[idx[k]] = r
                else:
                    tbl.append(r)
                    idx[k] = len(tbl) - 1
        else:
            tbl.extend(recs)
        return self


class FakeSupabase:
    def __init__(self, fail_tables=None):
        self.store = {}
        self._fail_tables = fail_tables or set()

    def table(self, name):
        return FakeTable(self.store, name, fail_tables=self._fail_tables)


@pytest.fixture
def fake_supabase():
    return FakeSupabase()


@pytest.fixture(autouse=True)
def env_dummy():
    os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "dummy-anon-key")
    # Intentionally contains a password to verify it is never leaked.
    os.environ["DATABASE_URL"] = "postgresql://gold_user:SUPERSECRET@db.example.com:5432/gold"
    yield


def _inject_fakes(monkeypatch):
    """Inject a fake yfinance and a fake app.db.supabase_client."""
    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, period=None):
            idx = pd.date_range("2020-01-01", periods=5, freq="D")
            return pd.DataFrame({"Close": [1800, 1810, 1820, 1830, 1840]}, index=idx)

    yf_mod = types.ModuleType("yfinance")
    yf_mod.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", yf_mod)

    fake_db = types.ModuleType("app.db.supabase_client")
    fake_db.ensure_schema = lambda *a, **k: True
    fake_db.supabase = None
    monkeypatch.setitem(sys.modules, "app.db.supabase_client", fake_db)


@pytest.fixture(autouse=True)
def inject_fakes(monkeypatch):
    """Inject lightweight fakes before each test so heavy deps are never imported."""
    _inject_fakes(monkeypatch)
    yield


def _seed_gold(fake, n=200, close=1800.0):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    fake.store["gold_prices"] = [
        {"date": d.strftime("%Y-%m-%d"), "close": float(close)} for d in dates
    ]


# --------------------------------------------------------------------------
# 7. Invalid / missing data
# --------------------------------------------------------------------------
def test_fetch_gold_series_empty_raises(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    fake_supabase.store["gold_prices"] = []
    with pytest.raises(RuntimeError):
        ps.fetch_gold_series()


def test_fetch_gold_series_all_nan_close_raises(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    fake_supabase.store["gold_prices"] = [
        {"date": d.strftime("%Y-%m-%d"), "close": None} for d in dates
    ]
    with pytest.raises(RuntimeError):
        ps.fetch_gold_series()


def test_fetch_gold_series_usable(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    _seed_gold(fake_supabase, n=200)
    df = ps.fetch_gold_series()
    assert len(df) == 200


# --------------------------------------------------------------------------
# Never overwrite valid predictions with invalid/empty ones (guard)
# --------------------------------------------------------------------------
def test_save_predictions_rejects_nan(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    ensemble = {"next_day": float("nan"), "next_week": [1.0] * 7, "next_month": [1.0] * 30}
    with pytest.raises(ValueError):
        ps.save_predictions(ensemble, [1.0] * 30, [1.0] * 30, pd.Timestamp("2024-01-01"))
    assert fake_supabase.store.get("predictions", []) == []


def test_save_predictions_rejects_incomplete(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    ensemble = {"next_day": 1.0, "next_week": [1.0] * 7, "next_month": [1.0] * 5}
    with pytest.raises(ValueError):
        ps.save_predictions(ensemble, [1.0] * 30, [1.0] * 30, pd.Timestamp("2024-01-01"))
    assert fake_supabase.store.get("predictions", []) == []


def test_compute_ensemble_short_raises():
    import services.prediction_service as ps
    with pytest.raises(ValueError):
        ps.compute_ensemble([1.0] * 10, [1.0] * 10)


# --------------------------------------------------------------------------
# 5. N-HiTS model loading failure
# --------------------------------------------------------------------------
def test_nhits_failure_preserves_predictions(fake_supabase, monkeypatch):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    _seed_gold(fake_supabase)
    fake_supabase.store["predictions"] = [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 2.0}
    ]

    # Chronos succeeds (patched, avoids heavy torch import); N-HiTS fails.
    monkeypatch.setattr(ps, "run_chronos_prediction", lambda s, h=30: [1.0] * 30)
    monkeypatch.setattr(ps, "run_nhits_prediction",
                        lambda h=30: (_ for _ in ()).throw(FileNotFoundError("N-HiTS artifact missing")))

    with pytest.raises((RuntimeError, FileNotFoundError)):
        ps.run_prediction_pipeline(dry_run=False)

    assert fake_supabase.store["predictions"] == [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 2.0}
    ]


def test_compute_ensemble_nonfinite_raises():
    import services.prediction_service as ps
    chronos = [1.0] * 30
    nhits = [1.0] * 29 + [float("nan")]
    with pytest.raises(ValueError):
        ps.compute_ensemble(chronos, nhits)


# --------------------------------------------------------------------------
# 4. Chronos-T5 loading failure
# --------------------------------------------------------------------------
def test_chronos_failure_preserves_predictions(fake_supabase, monkeypatch):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    _seed_gold(fake_supabase)
    fake_supabase.store["predictions"] = [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 1.0,
         "model_version": "old"}
    ]

    def boom(*a, **k):
        raise RuntimeError("Chronos-T5 model load failed (HF unavailable)")
    monkeypatch.setattr(ps, "run_chronos_prediction", boom)

    with pytest.raises(RuntimeError):
        ps.run_prediction_pipeline(dry_run=False)

    # Existing valid predictions must be untouched.
    assert fake_supabase.store["predictions"] == [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 1.0,
         "model_version": "old"}
    ]


# --------------------------------------------------------------------------
# 6. Ensemble prediction failure
# --------------------------------------------------------------------------
def test_ensemble_failure_preserves_predictions(fake_supabase, monkeypatch):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    _seed_gold(fake_supabase)
    fake_supabase.store["predictions"] = [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 3.0}
    ]

    # Chronos returns NaNs -> compute_ensemble must refuse.
    monkeypatch.setattr(ps, "run_chronos_prediction",
                        lambda s, h=30: [1.0] * 29 + [float("nan")])
    monkeypatch.setattr(ps, "run_nhits_prediction", lambda h=30: [1.0] * 30)

    with pytest.raises(ValueError):
        ps.run_prediction_pipeline(dry_run=False)

    assert fake_supabase.store["predictions"] == [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 3.0}
    ]


# --------------------------------------------------------------------------
# 8. Duplicate daily execution (idempotency)
# --------------------------------------------------------------------------
def test_duplicate_execution_is_idempotent(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    last = pd.Timestamp("2024-01-01")
    ensemble = {"next_day": 1.0, "next_week": [2.0] * 7, "next_month": [3.0] * 30}
    ps.save_predictions(ensemble, [1.5] * 30, [1.2] * 30, last)
    n1 = len(fake_supabase.store["predictions"])

    ensemble2 = {"next_day": 1.1, "next_week": [2.1] * 7, "next_month": [3.1] * 30}
    ps.save_predictions(ensemble2, [1.6] * 30, [1.3] * 30, last)
    n2 = len(fake_supabase.store["predictions"])

    # Same date+horizon keys -> upsert, no row growth (1 + 7 + 30 = 38 rows).
    assert n1 == 38 and n2 == 38
    assert fake_supabase.store["predictions"][0]["ensemble_pred"] == 1.1


# --------------------------------------------------------------------------
# 1. Gold data API unavailable
# --------------------------------------------------------------------------
def test_gold_api_unavailable_propagates(monkeypatch):
    import services.data_fetch_service as dfs

    class BoomTicker:
        def __init__(self, sym):
            pass

        def history(self, period=None):
            raise RuntimeError("Yahoo Finance API unavailable (simulated)")

    boom_mod = types.ModuleType("yfinance")
    boom_mod.Ticker = BoomTicker
    # fetch_gold_data references `yf.Ticker` via the module-level `yf`, so we
    # must patch the module reference it actually uses.
    monkeypatch.setattr(dfs, "yf", boom_mod)

    fake = FakeSupabase()
    dfs.supabase = fake
    with pytest.raises(RuntimeError):
        dfs.run_data_fetch()


# --------------------------------------------------------------------------
# 3. Supabase temporarily unavailable
# --------------------------------------------------------------------------
def test_supabase_unavailable_read_fails_pipeline(monkeypatch):
    import run_daily_pipeline as rdp
    monkeypatch.setattr(sys, "argv", ["run_daily_pipeline.py"])

    # Data reads fail, but the control-plane table (pipeline_runs) is still
    # writable, so the failure can still be recorded.
    failing = FakeSupabase(fail_tables={"gold_prices"})
    monkeypatch.setattr(rdp.data_fetch_service, "supabase", failing)
    monkeypatch.setattr(rdp.feature_service, "supabase", failing)
    monkeypatch.setattr(rdp.prediction_service, "supabase", failing)
    # Seed some previously-valid predictions -> must be preserved.
    failing.store["predictions"] = [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 9.0}
    ]

    code = rdp.main()
    assert code == 1
    runs = failing.store.get("pipeline_runs", [])
    assert any(r["status"] == "failed" for r in runs)
    # No overwrite of preserved predictions.
    assert failing.store["predictions"] == [
        {"date": "2024-01-02", "horizon": "30d", "ensemble_pred": 9.0}
    ]


# --------------------------------------------------------------------------
# End-to-end failure path + failure recorded + non-zero exit
# --------------------------------------------------------------------------
def test_run_daily_pipeline_failure_exit_and_record(monkeypatch):
    import run_daily_pipeline as rdp
    monkeypatch.setattr(sys, "argv", ["run_daily_pipeline.py"])

    fake = FakeSupabase()
    monkeypatch.setattr(rdp.data_fetch_service, "supabase", fake)
    monkeypatch.setattr(rdp.feature_service, "supabase", fake)
    monkeypatch.setattr(rdp.prediction_service, "supabase", fake)

    def boom():
        raise RuntimeError("Yahoo Finance down")
    monkeypatch.setattr(rdp.data_fetch_service, "run_data_fetch", boom)

    code = rdp.main()
    assert code == 1
    runs = fake.store.get("pipeline_runs", [])
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert "data_fetch" in runs[0]["error"]
    assert fake.store.get("predictions", []) == []


# --------------------------------------------------------------------------
# Secrets are never leaked into recorded errors
# --------------------------------------------------------------------------
def test_no_secret_leak_in_recorded_error(fake_supabase):
    import services.prediction_service as ps
    ps.supabase = fake_supabase
    os.environ["SUPABASE_KEY"] = "LEAKME_KEY"
    bad = "connection failed: postgresql://gold_user:SUPERSECRET@db.example.com:5432/gold"
    ps.record_run_status(
        __import__("datetime").datetime.now(),
        success=False,
        error=bad,
    )
    recorded = fake_supabase.store["pipeline_runs"][0]["error"]
    assert "SUPERSECRET" not in recorded
    assert "LEAKME_KEY" not in recorded
    assert "***" in recorded


# --------------------------------------------------------------------------
# 2. FX API unavailable (dashboard dependency)
# --------------------------------------------------------------------------
def test_fx_fallback_on_request_exception(monkeypatch):
    import services.fx_service as fx
    import requests

    def boom(*a, **k):
        raise requests.RequestException("FX API down")
    monkeypatch.setattr(fx.requests, "get", boom)
    assert fx.get_usd_inr_rate() == fx.FALLBACK_USD_INR


def test_fx_success(monkeypatch):
    import services.fx_service as fx

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"rates": {"INR": 82.5}}

    monkeypatch.setattr(fx.requests, "get", lambda *a, **k: R())
    assert fx.get_usd_inr_rate() == 82.5


# --------------------------------------------------------------------------
# 9. GitHub Actions dependency install failure -> job fails loudly
# --------------------------------------------------------------------------
def test_workflow_fails_loudly_on_dependency_failure():
    """Static check: the workflow must not mask failures and must run the script."""
    wf = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "daily_pipeline.yml")
    with open(wf, "r", encoding="utf-8") as f:
        content = f.read()
    # Uses the dedicated pipeline requirements (keeps dashboard deploy light).
    assert "requirements-pipeline.txt" in content
    # Actually invokes the pipeline entry point (returns non-zero on failure).
    assert "python run_daily_pipeline.py" in content
    # Must NOT swallow failures.
    assert "continue-on-error" not in content
    assert "|| true" not in content
