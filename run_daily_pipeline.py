#!/usr/bin/env python3
"""
run_daily_pipeline.py — single entry point for the daily GOLD forecast pipeline.

Flow:
  1. Fetch fresh gold-price data (+ macro/FX)
  2. Store/update data in Supabase
  3. Feature engineering
  4. Load model-ready data
  5. Chronos-T5 inference
  6. N-HiTS inference
  7. Ensemble
  8. Save predictions to Supabase (idempotent upsert on date+horizon)
  9. Record execution status

Failure handling:
  * Any stage failure raises (no silent swallowing) and is recorded in
    pipeline_runs with status='failed'.
  * Each raised PipelineError carries the failing STAGE name so GitHub Actions
    logs are easy to read.
  * The process exits non-zero on failure (GitHub Actions marks the job failed).
  * Invalid/empty predictions are NEVER persisted (see prediction_service), so a
    bad run cannot overwrite previously valid predictions.
  * Secrets (DATABASE_URL password, SUPABASE_KEY) are stripped from all logged
    and recorded error strings.

Usage:
  python run_daily_pipeline.py            # full run
  python run_daily_pipeline.py --dry-run # structure test, no Supabase writes, no heavy models
"""
import argparse
import os
import sys
import traceback
from datetime import datetime

# project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# load .env (local only; CI uses secrets)
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("daily_pipeline")

# imports of services
from app.db import supabase_client
from services import data_fetch_service, feature_service, prediction_service


class PipelineError(RuntimeError):
    """Carries the failing stage name for clear GitHub Actions logs."""

    def __init__(self, stage, message, cause=None):
        self.stage = stage
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


def _sanitize(msg):
    """Remove secret values (DB password, API keys) from a string before logging."""
    for secret in (
        os.getenv("DATABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
        os.getenv("SUPABASE_SERVICE_KEY", ""),
    ):
        if secret:
            msg = msg.replace(secret, "***")
    return msg


def _record_failure(started, err, metrics):
    """Best-effort record of a failure into pipeline_runs. Never raises."""
    try:
        stage = getattr(err, "stage", "unknown")
        safe_err = _sanitize(f"[{stage}] {err}")
        prediction_service.record_run_status(
            started,
            success=False,
            error=safe_err,
            records_processed=metrics.get("records_processed", 0),
            predictions_generated=0,
        )
    except Exception as se:
        # If Supabase is down we cannot record the failure; log clearly and move on.
        log.error("Could not record failure to pipeline_runs (Supabase may be unavailable): %s",
                  _sanitize(str(se)))


def main():
    parser = argparse.ArgumentParser(description="GOLD daily prediction pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Test structure without writing predictions/models")
    args = parser.parse_args()
    dry = args.dry_run
    started = datetime.now()
    metrics = {"records_processed": 0, "predictions_generated": 0}

    log.info("=" * 70)
    log.info("PROJECT GOLD — DAILY PREDICTION PIPELINE (dry_run=%s)", dry)
    log.info("Started: %s", started.isoformat())
    log.info("=" * 70)

    try:
        # STAGE 0: schema
        if not dry:
            log.info("STAGE 0: Ensure Supabase schema")
            try:
                supabase_client.ensure_schema()
            except Exception as e:
                raise PipelineError("schema", f"Supabase schema bootstrap failed: {e}") from e
        else:
            log.info("STAGE 0: [dry-run] skipping schema bootstrap")

        # STAGE 1-2: data ingestion
        log.info("STAGE 1-2: Fetch gold + macro data → Supabase")
        try:
            metrics["records_processed"] = data_fetch_service.run_data_fetch() or 0
        except Exception as e:
            raise PipelineError("data_fetch", f"Gold/macro data ingestion failed: {e}") from e

        # STAGE 3: features
        log.info("STAGE 3: Feature engineering → Supabase")
        try:
            feature_service.run_feature_engineering()
        except Exception as e:
            raise PipelineError("features", f"Feature engineering failed: {e}") from e

        # STAGE 4-8: predictions + ensemble + store
        log.info("STAGE 4-8: Load data → Chronos-T5 → N-HiTS → Ensemble → Supabase")
        try:
            ensemble, chronos_30, nhits_30, predictions_generated = prediction_service.run_prediction_pipeline(dry_run=dry)
        except Exception as e:
            raise PipelineError("predictions", f"Prediction/ensemble stage failed: {e}") from e

        # summary
        log.info("RESULT next_day=%.2f  next_week[0]=%.2f  next_month[0]=%.2f",
                 ensemble["next_day"], ensemble["next_week"][0], ensemble["next_month"][0])
        metrics["predictions_generated"] = predictions_generated

        # STAGE 9: status
        log.info("STAGE 9: Record pipeline run status")
        if not dry:
            try:
                prediction_service.record_run_status(
                    started,
                    success=True,
                    records_processed=metrics["records_processed"],
                    predictions_generated=predictions_generated,
                )
            except Exception as e:
                # Success path: do not fail the run just because status logging failed,
                # but make it very visible in the logs.
                log.error("STAGE 9: Failed to record SUCCESS status: %s", _sanitize(str(e)))

        log.info("=" * 70)
        log.info("PIPELINE COMPLETE ✅")
        log.info("=" * 70)
        return 0

    except PipelineError as pe:
        log.error("=" * 70)
        log.error("PIPELINE FAILED at STAGE [%s]", pe.stage)
        log.error("Reason: %s", _sanitize(str(pe)))
        log.error(traceback.format_exc())
        _record_failure(started, pe, metrics)
        return 1
    except Exception as e:
        log.error("=" * 70)
        log.error("PIPELINE FAILED (unexpected error)")
        log.error("Reason: %s", _sanitize(str(e)))
        log.error(traceback.format_exc())
        _record_failure(started, e, metrics)
        return 1


if __name__ == "__main__":
    sys.exit(main())
