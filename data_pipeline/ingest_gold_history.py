import sqlite3
import pandas as pd
import yfinance as yf
from pathlib import Path

DB_PATH = Path("database/gold_data.db")

# -------------------------------------------------
# FETCH HISTORICAL GOLD DATA (USD)
# -------------------------------------------------
# Gold futures ticker (industry standard)
ticker = "GC=F"

df = yf.download(
    ticker,
    start="2000-01-01",
    progress=False
)

if df.empty:
    raise RuntimeError("Failed to download gold price history")

df = df.reset_index()
df = df[["Date", "Close"]]
df.columns = ["date", "gold_close"]

# Dummy placeholders (can be enriched later)
df["usd_index"] = None
df["fx_inr"] = None

df["date"] = pd.to_datetime(df["date"])

# -------------------------------------------------
# SAVE TO SQLITE
# -------------------------------------------------
conn = sqlite3.connect(DB_PATH)

df.to_sql(
    "gold_prices",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print(f"[OK] gold_prices populated with {len(df)} rows")
