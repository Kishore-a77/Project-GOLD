"""
Ingest historical gold prices from Yahoo Finance
and store them in Supabase.

Author: Kishore A
Project: Project GOLD
"""

import pandas as pd
import yfinance as yf
from database.supabase_client import supabase

# -------------------------------------------------
# FETCH HISTORICAL GOLD DATA (USD)
# -------------------------------------------------

TICKER = "GC=F"

print("Downloading historical gold prices...")

df = yf.download(
    TICKER,
    start="2000-01-01",
    progress=False
)

if df.empty:
    raise RuntimeError("Failed to download historical gold price data.")

# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------

df = df.reset_index()

df = df[["Date", "Close"]]

df.columns = [
    "date",
    "gold_close"
]

# Placeholder columns
df["usd_index"] = None
df["fx_inr"] = None

# Convert date to string for PostgreSQL
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

# Replace NaN with None (Supabase/Postgres expects None for NULL)
df = df.where(pd.notnull(df), None)

rows = df.to_dict(orient="records")

print(f"Uploading {len(rows)} rows to Supabase...")

# -------------------------------------------------
# UPSERT INTO SUPABASE
# -------------------------------------------------

BATCH_SIZE = 500

for i in range(0, len(rows), BATCH_SIZE):

    batch = rows[i:i + BATCH_SIZE]

    response = (
        supabase
        .table("gold_prices")
        .upsert(batch)
        .execute()
    )

print("-------------------------------------------------")
print(f"✅ Successfully uploaded {len(rows)} historical records.")
print("Table: gold_prices")
print("-------------------------------------------------")