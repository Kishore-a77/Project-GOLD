"""
Data Fetch Service
Fetches latest gold & macro data from Yahoo Finance and stores in Supabase.
"""

import os
import yfinance as yf
import pandas as pd
from supabase import create_client
from datetime import datetime, timedelta

# Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_gold_data(period="5y"):
    """Fetch gold price data from Yahoo Finance."""
    ticker = yf.Ticker("GC=F")  # Gold Futures
    df = ticker.history(period=period)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_macro_data():
    """Fetch macro indicators (DXY, VIX, 10Y Treasury)."""
    indicators = {
        'DXY': 'DX-Y.NYB',      # US Dollar Index
        'VIX': '^VIX',          # Volatility Index
        'T10Y': '^TNX'          # 10-Year Treasury Yield
    }
    
    macro_data = {}
    for name, ticker in indicators.items():
        try:
            data = yf.Ticker(ticker).history(period="5y")
            macro_data[name] = data['Close']
        except Exception as e:
            print(f"Warning: Could not fetch {name}: {e}")
    
    macro_df = pd.DataFrame(macro_data)
    macro_df.index = pd.to_datetime(macro_df.index).date
    macro_df = macro_df.reset_index()
    macro_df.columns = ['date'] + list(indicators.keys())
    return macro_df


def get_latest_date_in_supabase():
    """Get the most recent date already stored in Supabase."""
    try:
        response = supabase.table('gold_prices').select('date').order('date', desc=True).limit(1).execute()
        if response.data:
            return pd.to_datetime(response.data[0]['date']).date()
    except Exception as e:
        print(f"Error checking latest date: {e}")
    return None


# In upsert_gold_data(), change the record to match your schema:
def upsert_gold_data(df):
    records = df.to_dict('records')
    for record in records:
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
    print(f"Upserted {len(records)} records")


def run_data_fetch():
    """Main entry point: fetch latest data and store in Supabase."""
    print(f"Starting data fetch at {datetime.now()}")
    
    # Check what's already in Supabase
    latest_date = get_latest_date_in_supabase()
    
    if latest_date:
        # Fetch from latest date onwards
        start_date = latest_date + timedelta(days=1)
        print(f"Fetching data from {start_date} onwards...")
        df = fetch_gold_data(period="1mo")
        df = df[df['date'] >= start_date]
    else:
        # First run - fetch full history
        print("No existing data found. Fetching full history...")
        df = fetch_gold_data(period="5y")
    
    if df.empty:
        print("No new data to fetch. Already up to date.")
        return
    
    upsert_gold_data(df)
    print(f"Data fetch complete. Added {len(df)} new rows.")


if __name__ == "__main__":
    run_data_fetch()