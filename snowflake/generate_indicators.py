from snowflake.snowpark import Session
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Snowflake connection parameters
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": "ACCOUNTADMIN",
    "warehouse": "GOLD_WH",
    "database": "GOLD_PROJECT",
    "schema": "PROCESSED"
}

# Create Snowflake session
session = Session.builder.configs(connection_params).create()
session.use_warehouse("GOLD_WH")
session.use_database("GOLD_PROJECT")
session.use_schema("PROCESSED")

# Load MASTER table
df = session.table("MASTER_GOLD_DATA").to_pandas()

# -------------------------------------------------
#              TECHNICAL INDICATORS
# -------------------------------------------------

# ------ Simple Moving Averages ------
df["SMA10"] = ta.sma(df["GOLD_CLOSE"], length=10)
df["SMA20"] = ta.sma(df["GOLD_CLOSE"], length=20)
df["SMA50"] = ta.sma(df["GOLD_CLOSE"], length=50)

# ------ Exponential Moving Averages ------
df["EMA12"] = ta.ema(df["GOLD_CLOSE"], length=12)
df["EMA26"] = ta.ema(df["GOLD_CLOSE"], length=26)

# ------ RSI ------
df["RSI14"] = ta.rsi(df["GOLD_CLOSE"], length=14)

# ------ MACD (Auto-detect columns) ------
macd = ta.macd(df["GOLD_CLOSE"])

macd_main = [c for c in macd.columns if "MACD_" in c.upper()][0]
macd_signal = [c for c in macd.columns if "MACDS" in c.upper()][0]
macd_hist = [c for c in macd.columns if "MACDH" in c.upper()][0]

df["MACD"] = macd[macd_main]
df["MACD_SIGNAL"] = macd[macd_signal]
df["MACD_HIST"] = macd[macd_hist]

# ------ Bollinger Bands (Auto-detect columns) ------
bb = ta.bbands(df["GOLD_CLOSE"], length=20)

bb_lower = [c for c in bb.columns if c.upper().startswith("BBL")][0]
bb_middle = [c for c in bb.columns if c.upper().startswith("BBM")][0]
bb_upper = [c for c in bb.columns if c.upper().startswith("BBU")][0]

df["BB_LOWER"] = bb[bb_lower]
df["BB_MIDDLE"] = bb[bb_middle]
df["BB_UPPER"] = bb[bb_upper]

# ------ ATR ------
df["ATR"] = ta.atr(df["GOLD_HIGH"], df["GOLD_LOW"], df["GOLD_CLOSE"])

# ------ Stochastic Oscillator (Auto-detect) ------
stoch = ta.stoch(df["GOLD_HIGH"], df["GOLD_LOW"], df["GOLD_CLOSE"])

stoch_k = [c for c in stoch.columns if "K" in c.upper()][0]
stoch_d = [c for c in stoch.columns if "D" in c.upper()][0]

df["STOCH_K"] = stoch[stoch_k]
df["STOCH_D"] = stoch[stoch_d]

# ------ Momentum ------
df["MOMENTUM10"] = ta.mom(df["GOLD_CLOSE"], length=10)

# ------ Rate of Change ------
df["ROC"] = ta.roc(df["GOLD_CLOSE"], length=10)

# Fill missing values
df = df.fillna(method="ffill").fillna(0)

# -------------------------------------------------
#               SAVE TO SNOWFLAKE
# -------------------------------------------------

session.write_pandas(
    df,
    table_name="MASTER_WITH_INDICATORS",
    auto_create_table=True,
    overwrite=True
)

print("\nMASTER_WITH_INDICATORS table created successfully!\n")
