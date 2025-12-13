import requests
import yfinance as yf
import pandas as pd
from snowflake.snowpark import Session
from dotenv import load_dotenv
import os
import logging

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")

def get_session():
    """Create Snowflake Snowpark session."""
    connection_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": "ACCOUNTADMIN",
        "warehouse": "GOLD_WH",
        "database": "GOLD_PROJECT",
        "schema": "RAW_DATA"
    }
    return Session.builder.configs(connection_params).create()


# ----------------------------
# Fetch data from FRED
# ----------------------------
def fetch_fred_series(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "observation_start": "2000-01-01"
    }

    logging.info(f"Fetching FRED series: {series_id}")
    r = requests.get(url, params=params)

    if r.status_code != 200:
        logging.error(f"❌ FRED API failed for {series_id}")
        return None

    data = r.json().get("observations", [])
    if not data:
        logging.warning(f"❌ No data for {series_id}")
        return None

    df = pd.DataFrame(data)
    df = df[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Create OHLC placeholder columns
    df["open"] = df["value"]
    df["high"] = df["value"]
    df["low"] = df["value"]
    df["close"] = df["value"]
    df["adj_close"] = df["value"]
    df["volume"] = 0

    df.columns = ["DATE", "VALUE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]

    return df


# ----------------------------
# Fetch from Yahoo Finance (VIX)
# ----------------------------
def fetch_vix():
    logging.info("Downloading VIX (^VIX) from Yahoo Finance...")

    df = yf.download("^VIX", period="20y")

    if df.empty:
        logging.error("❌ VIX returned no data.")
        return None

    df = df.reset_index()

    # Yahoo sometimes returns 6 columns (no Adj Close)
    if len(df.columns) == 6:
        df.columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        df["ADJ_CLOSE"] = df["CLOSE"]
        df = df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]]
    else:
        df.columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]

    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    return df


# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    session = get_session()
    logging.info(f"Snowflake session: {session.session_id}")

    # --- CPI ---
    cpi_df = fetch_fred_series("CPIAUCSL")
    if cpi_df is not None:
        logging.info(f"Uploading {len(cpi_df)} rows into CPI_RAW...")
        session.write_pandas(cpi_df, "CPI_RAW", overwrite=True)
        logging.info("✔ CPI upload complete.")

    # --- 10Y Yield ---
    yield_df = fetch_fred_series("DGS10")
    if yield_df is not None:
        logging.info(f"Uploading {len(yield_df)} rows into US10Y_RAW...")
        session.write_pandas(yield_df, "US10Y_RAW", overwrite=True)
        logging.info("✔ 10Y Yield upload complete.")

    # --- VIX ---
    vix_df = fetch_vix()
    if vix_df is not None:
        logging.info(f"Uploading {len(vix_df)} rows into VIX_RAW...")
        session.write_pandas(vix_df, "VIX_RAW", overwrite=True)
        logging.info("✔ VIX upload complete.")

    logging.info("Day 4 ingestion completed.")
    session.close()
