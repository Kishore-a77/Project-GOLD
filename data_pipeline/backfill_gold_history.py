#!/usr/bin/env python3
"""
data_pipeline/backfill_gold_history.py

One-off / rerunnable script to catch Supabase's `gold_prices` table up to the
present, in small chunks, using batch upserts.

Why this is "resumable" without any extra checkpoint file:
  - Every chunk of rows is written with a single batched upsert() call keyed
    on the `date` column (on_conflict='date').
  - Because upserts are idempotent, a chunk that already succeeded can be
    upserted again with no harm.
  - The script always asks Supabase "what's the latest date you have?" before
    fetching anything. So if it dies at 2am on chunk 14 of 40, you just run it
    again — it picks up from the new latest date instead of restarting at
    Feb 2023.

Usage:
  python data_pipeline/backfill_gold_history.py
  python data_pipeline/backfill_gold_history.py --start 2000-01-01
  python data_pipeline/backfill_gold_history.py --chunk-months 3 --batch-size 250
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from services.data_fetch_service import (
    fetch_gold_data,
    upsert_gold_data,
    get_latest_date_in_supabase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_gold_history")

# Both Yahoo Finance (cookie/crumb handshake, rate limiting) and Supabase
# (plain network blips) fail transiently and intermittently. Without retries,
# a single bad handshake kills the whole run even though a retry a few
# seconds later usually succeeds.
CALL_RETRIES = 4
CALL_BACKOFF_SECONDS = 5


def call_with_retries(fn, *args, what="call", retries=CALL_RETRIES, **kwargs):
    """Call fn(*args, **kwargs), retrying transient failures with backoff.

    RuntimeError from fetch_gold_data (Yahoo returned no data at all for the
    window) is NOT retried here -- that's treated as "empty window" by the
    caller, not a transient fault.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            wait = CALL_BACKOFF_SECONDS * attempt
            log.warning("%s failed (attempt %d/%d): %s -- retrying in %ds",
                        what, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries} attempts: {last_err}") from last_err


def daterange_chunks(start_date, end_date, chunk_months):
    """Yield (chunk_start, chunk_end_exclusive) pairs covering [start_date, end_date]."""
    cursor = start_date
    while cursor <= end_date:
        # Advance ~chunk_months months without pulling in a calendar library dependency.
        month = cursor.month - 1 + chunk_months
        year = cursor.year + month // 12
        month = month % 12 + 1
        next_cursor = cursor.replace(year=year, month=month, day=1)
        chunk_end_exclusive = min(next_cursor, end_date + timedelta(days=1))
        yield cursor, chunk_end_exclusive
        cursor = chunk_end_exclusive


def run_backfill(default_start, chunk_months, batch_size, sleep_seconds):
    latest_date = call_with_retries(
        get_latest_date_in_supabase, what="Supabase latest-date lookup"
    )
    today = datetime.now().date()

    if latest_date:
        start_date = latest_date + timedelta(days=1)
        log.info("Resuming backfill: latest row in Supabase is %s -> starting from %s",
                  latest_date, start_date)
    else:
        start_date = default_start
        log.info("No existing rows found. Starting full backfill from %s", start_date)

    if start_date > today:
        log.info("Already caught up (latest=%s). Nothing to backfill.", latest_date)
        return 0

    total_written = 0
    chunks = list(daterange_chunks(start_date, today, chunk_months))
    log.info("Backfill plan: %d chunk(s) of ~%d month(s) each, batch_size=%d",
              len(chunks), chunk_months, batch_size)

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        log.info("Chunk %d/%d: fetching %s to %s (exclusive)...",
                  idx, len(chunks), chunk_start, chunk_end)
        try:
            df = call_with_retries(
                fetch_gold_data, start=chunk_start, end=chunk_end,
                what=f"Yahoo Finance fetch (chunk {idx}/{len(chunks)})",
            )
        except RuntimeError as e:
            # Yahoo Finance returned nothing for this window (e.g. a very
            # recent chunk with no trading days yet) even after retries.
            # Safe to skip -- it's an empty window, not a partial failure.
            log.warning("Chunk %d/%d: no data returned (%s); skipping.", idx, len(chunks), e)
            continue

        valid = df[df['close'].notna()]
        if valid.empty:
            log.warning("Chunk %d/%d: no valid rows after filtering; skipping.", idx, len(chunks))
            continue

        written = upsert_gold_data(valid, batch_size=batch_size)
        total_written += written
        log.info("Chunk %d/%d: upserted %d rows (running total: %d).",
                  idx, len(chunks), written, total_written)

        if sleep_seconds and idx < len(chunks):
            time.sleep(sleep_seconds)

    log.info("Backfill complete. %d rows written this run.", total_written)
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Resumable historical gold-price backfill")
    parser.add_argument("--start", default="2000-01-01",
                         help="Earliest date to backfill from if Supabase is empty (YYYY-MM-DD).")
    parser.add_argument("--chunk-months", type=int, default=6,
                         help="Months of data to fetch from Yahoo Finance per chunk.")
    parser.add_argument("--batch-size", type=int, default=500,
                         help="Rows per Supabase upsert() call.")
    parser.add_argument("--sleep", type=float, default=3.0,
                         help="Seconds to sleep between chunks (be gentle on the API "
                              "and reduce crumb/rate-limit failures).")
    args = parser.parse_args()

    default_start = datetime.strptime(args.start, "%Y-%m-%d").date()

    try:
        run_backfill(default_start, args.chunk_months, args.batch_size, args.sleep)
        return 0
    except Exception as e:
        log.error("Backfill failed: %s", e, exc_info=True)
        log.error("Safe to re-run — it will resume from the last committed date.")
        return 1


if __name__ == "__main__":
    sys.exit(main())