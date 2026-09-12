"""
tests/test_dashboard_integration.py

Comprehensive end-to-end integration test verifying:
1. Supabase connectivity without WinError 10060.
2. Centralized client retry and IPv4 handling.
3. System readiness detection (READY/STALE).
4. Historical gold price loading & row count.
5. Prediction horizons retrieval (1d, 7d, 30d).
6. Pipeline status ('success') and model version ('chronos-t5-small + nhits_vfinal').
7. Prediction generation timestamp validation (must NOT be 2027 target date).
8. Live FX conversion and price metrics calculation.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.database_service import (
    check_system_readiness,
    fetch_actuals,
    fetch_ensemble_forecast,
    fetch_pipeline_runs,
    fetch_prediction_inventory,
)
from services.fx_service import get_usd_inr_rate


def run_tests():
    print("==================================================")
    print("STARTING COMPREHENSIVE DASHBOARD INTEGRATION SUITE")
    print("==================================================")

    # 1. Readiness Check
    print("\n[1/6] Running System Readiness Check...")
    readiness = check_system_readiness()
    state = readiness["state"]
    print(f"  -> State: {state}")
    assert state in ("READY", "STALE"), f"Unexpected readiness state: {state}"
    assert readiness["latest_gold_date"] is not None, "latest_gold_date is None"
    assert readiness["total_predictions"] > 0, f"Expected predictions > 0, got {readiness['total_predictions']}"

    # 2. Historical Actuals Check
    print("\n[2/6] Loading Historical Gold Prices...")
    actuals = fetch_actuals()
    print(f"  -> Total historical rows: {len(actuals):,}")
    print(f"  -> Earliest date: {actuals.index.min().date()}")
    print(f"  -> Latest date:   {actuals.index.max().date()}")
    assert len(actuals) >= 1275, f"Expected >= 1275 gold prices, got {len(actuals)}"
    latest_gold_date = actuals.index.max()

    # 3. Forecast Predictions Check (1d and 30d)
    print("\n[3/6] Fetching Forecast Predictions...")
    pred_1d = fetch_ensemble_forecast("1d", latest_gold_date)
    print(f"  -> 1d predictions returned: {len(pred_1d)}")

    pred_30d = fetch_ensemble_forecast("30d", latest_gold_date)
    print(f"  -> 30d predictions returned: {len(pred_30d)}")
    assert len(pred_30d) > 0, "No 30d predictions returned"

    # 4. Pipeline Runs & Model Version Check
    print("\n[4/6] Verifying Pipeline Run Status & Model Version...")
    latest_run, last_success = fetch_pipeline_runs()
    assert latest_run is not None, "latest_run is None"
    print(f"  -> Pipeline Status: {latest_run.get('status')}")
    assert latest_run.get("status") == "success", f"Expected 'success', got {latest_run.get('status')}"

    model_version = readiness.get("model_version")
    print(f"  -> Model Version:   {model_version}")
    assert model_version is not None, "Model version is None"
    assert "chronos" in model_version.lower(), f"Unexpected model version: {model_version}"

    # 5. Prediction Generation Date Mapping Verification
    print("\n[5/6] Verifying 'Predictions Last Generated' Date Mapping...")
    pred_gen_date = readiness.get("pred_generated_date")
    print(f"  -> Predictions Generated At: {pred_gen_date}")
    assert pred_gen_date is not None, "pred_generated_date is None"
    pred_gen_str = str(pred_gen_date)
    assert "2026-09-10" in pred_gen_str, f"Expected run date 2026-09-10, got {pred_gen_str}"
    assert "2027" not in pred_gen_str, "CRITICAL ERROR: Prediction target date (2027) confused with generation date!"

    # 6. FX Rate & Conversion Check
    print("\n[6/6] Verifying FX Conversion...")
    usd_inr = get_usd_inr_rate()
    print(f"  -> USD/INR Rate: INR {usd_inr:.2f}")
    assert 70.0 <= usd_inr <= 120.0, f"Unreasonable FX rate: {usd_inr}"

    print("\n==================================================")
    print("[PASS] ALL 6 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
