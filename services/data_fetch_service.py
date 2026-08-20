"""
Data Fetch Service
Fetches latest gold & macro data from Yahoo Finance and stores in Supabase.
"""

import os
import time
import logging
import yfinance as yf
import pandas as pd
from supabase import create_client
from datetime import datetime, timedelta

logger = logging.getLogger("data_fetch_service")

# Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Backend pipeline writes must bypass Row-Level Security, so prefer the
# service_role key over the anon key (this table currently has no RLS policy
# blocking anon writes, but relying on that is fragile -- if RLS is ever
# enabled on gold_prices, anon-key writes will start failing with 401/42501).
if not SUPABASE_SERVICE_KEY:
    logger.warning(
        "SUPABASE_SERVICE_KEY not set; falling back to anon key for gold_prices writes."
    )
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY or SUPABASE_KEY)

# How many rows to send to Supabase per upsert call.
BATCH_SIZE = 500
# Retries per batch before giving up (network blips / transient 5xx from PostgREST).
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def fetch_gold_data(period=None, start=None, end=None):
    """Fetch gold price data from Yahoo Finance.

    Supports two modes:
      - period="5y" / "1mo" / etc: relative window ending today (legacy behavior).
      - start=<date>, end=<date>: explicit date range. This is what lets us
        request "everything since the last row we have", instead of being
        capped at a fixed 1-month window.

    start/end accept anything pandas.Timestamp can parse (str, date, datetime).
    `end` is treated as exclusive by yfinance, so callers should pass the day
    *after* the last day they want included.
    """
    ticker = yf.Ticker("GC=F")  # Gold Futures

    if start is not None:
        df = ticker.history(start=str(start), end=str(end) if end else None)
    else:
        df = ticker.history(period=period or "5y")

    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no gold (GC=F) data.")

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_macro_data():
    """Fetch macro indicators (DXY, VIX, 10Y Treasury).

    Macro data is OPTIONAL for the prediction pipeline (only gold close is used).
    A failure to fetch one indicator is logged clearly and skipped rather than
    aborting the whole pipeline, but it is never silently swallowed.
    """
    indicators = {
        'DXY': 'DX-Y.NYB',      # US Dollar Index
        'VIX': '^VIX',          # Volatility Index
        'T10Y': '^TNX'          # 10-Year Treasury Yield
    }

    macro_data = {}
    for name, ticker in indicators.items():
        try:
            data = yf.Ticker(ticker).history(period="5y")
            if data is not None and not data.empty:
                macro_data[name] = data['Close']
            else:
                logger.warning("Macro indicator '%s' returned no data; skipping.", name)
        except Exception as e:  # yfinance can raise many exception types
            logger.warning("Macro indicator '%s' unavailable (skipped): %s", name, e)

    if not macro_data:
        logger.warning("No macro indicators fetched; continuing without macro data.")
        return pd.DataFrame(columns=['date'] + list(indicators.keys()))

    macro_df = pd.DataFrame(macro_data)
    macro_df.index = pd.to_datetime(macro_df.index).date
    macro_df = macro_df.reset_index()
    macro_df.columns = ['date'] + list(indicators.keys())
    return macro_df


def get_latest_date_in_supabase():
    """Get the most recent date already stored in Supabase.

    Raises (does NOT silently swallow) if Supabase is unreachable, so the
    pipeline fails loudly and clearly instead of guessing it's a first run.
    Returns None only when there is genuinely no data yet (first run).
    """
    try:
        response = supabase.table('gold_prices').select('date').order('date', desc=True).limit(1).execute()
        if response.data:
            return pd.to_datetime(response.data[0]['date']).date()
    except Exception as e:
        logger.error("Could not read latest date from Supabase (gold_prices): %s", e)
        raise
    return None


def _row_to_record(record):
    """Map one dataframe row (as dict) to a Supabase-ready record, or None to skip."""
    if pd.isna(record.get('close')):
        logger.warning("Skipping gold row with invalid close on %s", record.get('date'))
        return None
    mapped = {
        'date': str(record['date']),
        'open': float(record['open']) if pd.notna(record['open']) else None,
        'high': float(record['high']) if pd.notna(record['high']) else None,
        'low': float(record['low']) if pd.notna(record['low']) else None,
        'close': float(record['close']) if pd.notna(record['close']) else None,
        'volume': int(record['volume']) if pd.notna(record['volume']) else None,
    }
    # Remove None values so we don't overwrite existing columns with NULL
    # when a field is temporarily missing from the Yahoo Finance response.
    return {k: v for k, v in mapped.items() if v is not None}


def upsert_gold_data(df, batch_size=BATCH_SIZE):
    """Upsert gold rows into Supabase in batches instead of one request per row.

    This is what makes both the daily pipeline and the historical backfill fast
    and resumable:
      - Each batch is committed with a single upsert() call (on_conflict='date'),
        so N rows costs N/batch_size requests instead of N requests.
      - Because every batch is an idempotent upsert keyed on `date`, a crash
        partway through only loses the *current* batch. Whatever batches
        already succeeded are safely persisted in Supabase, so simply
        re-running the caller picks up right where it left off (see
        get_latest_date_in_supabase / backfill script) with no separate
        checkpoint file needed.
    """
    records = [r for r in (_row_to_record(rec) for rec in df.to_dict('records')) if r is not None]

    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                supabase.table('gold_prices').upsert(batch, on_conflict='date').execute()
                written += len(batch)
                logger.info(
                    "Upserted batch %d-%d (%d rows)",
                    i, i + len(batch) - 1, len(batch)
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(
                    "Batch upsert failed (attempt %d/%d) for rows %d-%d: %s",
                    attempt, MAX_RETRIES, i, i + len(batch) - 1, e
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        else:
            # All retries exhausted for this batch — fail loudly rather than
            # silently dropping rows, matching the rest of the pipeline's
            # "never swallow errors" convention.
            raise RuntimeError(
                f"Failed to upsert batch rows {i}-{i + len(batch) - 1} "
                f"after {MAX_RETRIES} attempts: {last_err}"
            )

    logger.info("Upserted %d/%d gold records", written, len(records))
    return written


def run_data_fetch():
    """Main entry point: fetch latest data and store in Supabase.

    Returns the number of gold rows written (0 if nothing new / nothing valid).
    Raises on hard failures (API down, Supabase down) so the pipeline fails loudly.
    """
    logger.info("Starting gold data fetch at %s", datetime.now())

    # Check what's already in Supabase (raises if Supabase is unreachable)
    latest_date = get_latest_date_in_supabase()

    if latest_date:
        # Fetch from latest date onwards — an explicit date range, not a fixed
        # "1mo" window, so this also works correctly when the DB is far behind
        # (e.g. after an interrupted backfill) instead of silently leaving a gap.
        start_date = latest_date + timedelta(days=1)
        end_date = datetime.now().date() + timedelta(days=1)  # yfinance end is exclusive
        if start_date >= end_date:
            logger.info("Already up to date (latest=%s); nothing to fetch.", latest_date)
            return 0
        logger.info("Fetching incremental gold data from %s to %s...", start_date, end_date)
        df = fetch_gold_data(start=start_date, end=end_date)
        df = df[df['date'] >= start_date]
    else:
        # First run - fetch full history
        logger.info("No existing data found. Fetching full gold history...")
        df = fetch_gold_data(period="5y")

    if df.empty:
        logger.info("No new gold data to fetch; already up to date.")
        return 0

    valid = df[df['close'].notna()]
    if valid.empty:
        logger.warning(
            "Fetched gold data contained no valid close values; skipping upsert "
            "to avoid corrupting gold_prices."
        )
        return 0

    written = upsert_gold_data(valid)
    logger.info("Data fetch complete. Added/updated %d rows.", written)
    return written


if __name__ == "__main__":
    run_data_fetch()