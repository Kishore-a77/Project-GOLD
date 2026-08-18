"""
Data Fetch Service
Fetches latest gold & macro data from Yahoo Finance and stores in Supabase.
"""

import os
import logging
import yfinance as yf
import pandas as pd
from supabase import create_client
from datetime import datetime, timedelta

logger = logging.getLogger("data_fetch_service")

# Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_gold_data(period="5y"):
    """Fetch gold price data from Yahoo Finance."""
    ticker = yf.Ticker("GC=F")  # Gold Futures
    df = ticker.history(period=period)
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


# In upsert_gold_data(), change the record to match your schema:
def upsert_gold_data(df):
    records = df.to_dict('records')
    written = 0
    for record in records:
        # Guard: never overwrite a valid row with a null close price.
        if pd.isna(record.get('close')):
            logger.warning("Skipping gold row with invalid close on %s", record.get('date'))
            continue
        mapped = {
            'date': str(record['date']),
            'open': float(record['open']) if pd.notna(record['open']) else None,
            'high': float(record['high']) if pd.notna(record['high']) else None,
            'low': float(record['low']) if pd.notna(record['low']) else None,
            'close': float(record['close']) if pd.notna(record['close']) else None,
            'volume': int(record['volume']) if pd.notna(record['volume']) else None,
        }
        # Remove None values
        mapped = {k: v for k, v in mapped.items() if v is not None}
        supabase.table('gold_prices').upsert(mapped).execute()
        written += 1
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
        # Fetch from latest date onwards
        start_date = latest_date + timedelta(days=1)
        logger.info("Fetching incremental gold data from %s onwards...", start_date)
        df = fetch_gold_data(period="1mo")
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