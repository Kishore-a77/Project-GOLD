# run_daily_pipeline.py

import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

DB_PATH = "database/gold_data.db"

# -----------------------------
# Fetch Gold Data
# -----------------------------
def fetch_gold_data():
    # Fetch 5 years of data
    gold = yf.download("GC=F", period="5y", interval="1d")
    gold.reset_index(inplace=True)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(gold["Date"]).dt.strftime("%Y-%m-%d")
    df["gold_close"] = pd.Series(gold["Close"].values.flatten())
    df["gold_high"] = pd.Series(gold["High"].values.flatten())
    df["gold_low"] = pd.Series(gold["Low"].values.flatten())
    df["gold_volume"] = pd.Series(gold["Volume"].values.flatten())

    # Fetch Macro Indicators
    sp500 = yf.download("^GSPC", period="5y", interval="1d")
    usd_idx = yf.download("DX-Y.NYB", period="5y", interval="1d")
    treasury = yf.download("^TNX", period="5y", interval="1d")

    sp500.reset_index(inplace=True)
    usd_idx.reset_index(inplace=True)
    treasury.reset_index(inplace=True)

    sp500_df = pd.DataFrame({
        "date": pd.to_datetime(sp500["Date"]).dt.strftime("%Y-%m-%d"),
        "sp500_close": pd.Series(sp500["Close"].values.flatten())
    })
    
    usd_idx_df = pd.DataFrame({
        "date": pd.to_datetime(usd_idx["Date"]).dt.strftime("%Y-%m-%d"),
        "usd_idx_close": pd.Series(usd_idx["Close"].values.flatten())
    })

    treasury_df = pd.DataFrame({
        "date": pd.to_datetime(treasury["Date"]).dt.strftime("%Y-%m-%d"),
        "treasury_yield": pd.Series(treasury["Close"].values.flatten())
    })

    # Merge all datasets
    df = df.merge(sp500_df, on="date", how="left")
    df = df.merge(usd_idx_df, on="date", how="left")
    df = df.merge(treasury_df, on="date", how="left")

    # Forward fill to handle holiday mismatches between markets
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def generate_features(df):

    df["sma_20"] = df["gold_close"].rolling(20).mean()
    df["sma_50"] = df["gold_close"].rolling(50).mean()
    df["ema_20"] = df["gold_close"].ewm(span=20).mean()

    rsi = RSIIndicator(close=df["gold_close"], window=14)
    df["rsi"] = rsi.rsi()
    
    macd = MACD(close=df["gold_close"])
    df["macd"] = macd.macd()

    # Volatility Features
    bb = BollingerBands(close=df["gold_close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    atr = AverageTrueRange(high=df["gold_high"], low=df["gold_low"], close=df["gold_close"], window=14)
    df["atr"] = atr.average_true_range()

    df.dropna(inplace=True)

    return df


# -----------------------------
# Save to SQLite
# -----------------------------
def save_to_sqlite(df):

    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        "features",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():

    print("\n[INFO] Fetching latest gold data...")

    df = fetch_gold_data()

    print("[INFO] Generating features...")

    df = generate_features(df)

    print("[INFO] Saving to SQLite...")

    save_to_sqlite(df)

    print("\n[SUCCESS] Daily pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()