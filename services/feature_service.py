"""
Feature Engineering Service
Reads gold prices from Supabase, computes technical indicators, stores back.
"""

import os
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_prices():
    """Fetch all gold prices from Supabase."""
    response = supabase.table('gold_prices').select('*').order('date').execute()
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
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


def upsert_features(df):
    """Upsert feature data to Supabase."""
    feature_cols = ['date', 'close', 'sma_7', 'sma_30', 'rsi_14', 
                    'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr_14']
    df = df[feature_cols].copy()
    df = df.dropna()
    
    records = df.to_dict('records')
    
    for record in records:
        record['date'] = str(record['date'])
        record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        supabase.table('gold_features').upsert(record).execute()
    
    print(f"Upserted {len(records)} feature records")


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