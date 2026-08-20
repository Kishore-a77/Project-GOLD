"""
Feature Engineering Service
Reads gold prices from Supabase, computes technical indicators, stores back.
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import datetime

logger = logging.getLogger("feature_service")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Backend pipeline writes must bypass Row-Level Security, so prefer the
# service_role key. Falling back to the anon key keeps this importable in
# environments that only set SUPABASE_KEY, but any table with an RLS policy
# (gold_features, predictions, pipeline_runs) WILL reject anon-key writes
# with a 401/42501 -- if you see that error, SUPABASE_SERVICE_KEY is missing
# or wrong in your environment/.env/GitHub Actions secrets.
if not SUPABASE_SERVICE_KEY:
    logging.getLogger("feature_service").warning(
        "SUPABASE_SERVICE_KEY not set; falling back to anon key. "
        "Writes to RLS-protected tables (gold_features, pipeline_runs) will fail."
    )
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY or SUPABASE_KEY)

# PostgREST caps an unpaginated select() at 1000 rows by default. Once
# gold_prices grows past that (as it now has, post-backfill), a plain
# .select('*').execute() silently truncates -- and because results are
# ordered ascending by date, it's the newest rows that get dropped, which is
# exactly the data the models need most. PAGE_SIZE below drives explicit
# pagination so this always fetches everything, at any table size.
PAGE_SIZE = 1000
FEATURE_BATCH_SIZE = int(os.getenv("SUPABASE_FEATURE_BATCH_SIZE", "200"))
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def fetch_all_prices():
    """Fetch ALL gold prices from Supabase, paginating past the 1000-row cap."""
    rows = []
    start = 0
    while True:
        end = start + PAGE_SIZE - 1
        response = (
            supabase.table('gold_prices')
            .select('*')
            .order('date')
            .range(start, end)
            .execute()
        )
        page = response.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    logger.info("Fetched %d price records (paginated)", len(df))
    return df


def compute_features(df):
    """Compute technical indicators."""
    df = df.sort_values('date').reset_index(drop=True)
    
    # Simple Moving Averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    return df


def _is_transient_error(exc: Exception) -> bool:
    """Return True if the exception looks like a transient HTTP/2 or network error."""
    if isinstance(exc, (EOFError, ConnectionResetError)):
        return True
    exc_str = str(exc).lower()
    transient_keywords = [
        "streamreset",
        "eof",
        "connection reset",
        "timeout",
        "remote reset",
        "connection reset by peer",
    ]
    return any(kw in exc_str for kw in transient_keywords)


def upsert_features(df, batch_size=None):
    """Upsert feature data to Supabase in batches with retry and exponential backoff."""
    if batch_size is None:
        batch_size = FEATURE_BATCH_SIZE

    feature_cols = ['date', 'close', 'sma_7', 'sma_30', 'rsi_14',
                    'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14']
    df = df[feature_cols].copy()
    df = df.dropna()

    records = df.to_dict('records')
    for record in records:
        record['date'] = str(record['date'])

    written = 0
    total_batches = (len(records) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(records))
        batch = records[batch_start:batch_end]
        batch_num = batch_idx + 1

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            start_time = time.time()
            try:
                supabase.table('gold_features').upsert(batch, on_conflict='date').execute()
                elapsed = time.time() - start_time
                written += len(batch)
                logger.info(
                    "Feature batch %d/%d (rows %d-%d, size=%d) attempt %d/%d: SUCCESS in %.2fs",
                    batch_num, total_batches, batch_start, batch_end - 1,
                    len(batch), attempt, MAX_RETRIES, elapsed
                )
                break
            except Exception as e:
                elapsed = time.time() - start_time
                last_err = e
                logger.warning(
                    "Feature batch %d/%d (rows %d-%d, size=%d) attempt %d/%d: FAILED after %.2fs: %s",
                    batch_num, total_batches, batch_start, batch_end - 1,
                    len(batch), attempt, MAX_RETRIES, elapsed, e
                )
                if attempt < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    time.sleep(backoff)
        else:
            raise RuntimeError(
                f"Failed to upsert feature batch rows {batch_start}-{batch_end - 1} "
                f"(batch {batch_num}/{total_batches}) after {MAX_RETRIES} attempts: {last_err}"
            )

    print(f"Upserted {written} feature records")
    return written


def run_feature_engineering():
    """Main entry point."""
    print(f"Starting feature engineering at {datetime.now()}")

    df = fetch_all_prices()
    print(f"Fetched {len(df)} price records")

    df = compute_features(df)
    upsert_features(df)

    print("Feature engineering complete.")
    return len(df)


if __name__ == "__main__":
    run_feature_engineering()