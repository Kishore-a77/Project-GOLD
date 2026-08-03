import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("database/gold_data.db")

conn = sqlite3.connect(DB_PATH)

# -------------------------------------------------
# LOAD RAW DATA
# -------------------------------------------------
df = pd.read_sql(
    """
    SELECT
        date,
        gold_close
    FROM gold_prices
    ORDER BY date
    """,
    conn
)

if df.empty:
    conn.close()
    raise RuntimeError("gold_prices table is empty. Cannot build features.")

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# -------------------------------------------------
# FEATURE ENGINEERING (PURE PANDAS)
# -------------------------------------------------

# Moving averages
df["sma_20"] = df["gold_close"].rolling(20).mean()
df["sma_50"] = df["gold_close"].rolling(50).mean()
df["ema_20"] = df["gold_close"].ewm(span=20, adjust=False).mean()

# RSI (14)
delta = df["gold_close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss

df["rsi"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["gold_close"].ewm(span=12, adjust=False).mean()
ema26 = df["gold_close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26

# -------------------------------------------------
# DROP NaNs ONLY FROM FEATURE COLUMNS
# -------------------------------------------------
df = df.dropna(subset=[
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi",
    "macd"
])

# -------------------------------------------------
# SAVE TO SQLITE
# -------------------------------------------------
df.reset_index().to_sql(
    "features",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print(f"[OK] Features rebuilt successfully. Rows: {len(df)}")
