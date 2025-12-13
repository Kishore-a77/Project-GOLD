#!/usr/bin/env python3
"""
Incremental downloader for Gold (GC=F) from Yahoo using yfinance
and uploader to Snowflake (GOLD_PROJECT.RAW_DATA.GOLD_PRICES_RAW).

Behavior (Incremental):
- Query Snowflake for the MAX(PRICE_DATE) in GOLD_PRICES_RAW
- Download GC=F data starting from next_calendar_day after that date
- If new rows exist, append them to the Snowflake table
- If table does not exist, create it and insert full dataset

Run:
    python snowflake/auto_download_and_upload_incremental.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Snowpark Session import
from snowflake.snowpark import Session

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gold_ingest")

# -------------------------
# Config / ENV
# -------------------------
load_dotenv()  # loads .env from project root

SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "GOLD_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "GOLD_PROJECT"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "RAW_DATA"),
}

TABLE_NAME = "GOLD_PRICES_RAW"  # target table
FULL_TABLE_IDENTIFIER = f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.{TABLE_NAME}"

# -------------------------
# Snowflake helpers
# -------------------------
def get_snowflake_session():
    missing = [k for k, v in SNOWFLAKE_CONFIG.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Snowflake config in .env: {missing}")
    logger.info("Creating Snowflake session...")
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
    # set role/warehouse/db/schema explicitly to be safe
    session.sql(f"USE ROLE {SNOWFLAKE_CONFIG['role']}").collect()
    session.sql(f"USE WAREHOUSE {SNOWFLAKE_CONFIG['warehouse']}").collect()
    session.sql(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}").collect()
    session.sql(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}").collect()
    return session

def get_max_date_in_table(session):
    """Return max PRICE_DATE from target table. If table missing, return None."""
    try:
        # check existence
        check = session.sql(f"""
            SELECT COUNT(*) AS CNT
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{SNOWFLAKE_CONFIG['schema'].upper()}'
              AND TABLE_NAME = '{TABLE_NAME.upper()}'
        """).collect()
        if check and check[0]["CNT"] == 0:
            logger.info(f"Table {TABLE_NAME} does not exist in Snowflake.")
            return None

        res = session.sql(f"SELECT MAX(PRICE_DATE) AS MX FROM {TABLE_NAME}").collect()
        if not res:
            return None
        mx = res[0]["MX"]
        if mx is None:
            return None
        if isinstance(mx, datetime):
            return mx.date()
        # in case returned as date
        return mx
    except Exception as e:
        logger.exception("Error fetching max date from Snowflake.")
        raise

# -------------------------
# Data download and transform
# -------------------------
def download_gc_data(start_date=None, end_date=None):
    ticker = "GC=F"
    logger.info(f"Downloading {ticker} data from {start_date} to {end_date} ...")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,   # Important fix for missing OHLC
        actions=False,
        threads=True
    )

    if df is None or df.empty:
        logger.warning("Yahoo returned EMPTY dataframe.")
        return pd.DataFrame()

    # Flatten multi-index columns (Yahoo sometimes changes format)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Reset index
    df = df.reset_index()

    # Check if "Date" column exists
    if "Date" not in df.columns:
        logger.error(f"Missing 'Date' in Yahoo data. Columns received: {df.columns}")
        return pd.DataFrame()

    # Rename to Snowflake schema names
    df = df.rename(columns={
        "Date": "PRICE_DATE",
        "Open": "OPEN",
        "High": "HIGH",
        "Low": "LOW",
        "Close": "CLOSE",
        "Adj Close": "ADJ_CLOSE",
        "Volume": "VOLUME"
    })

    # Date handling
    df["PRICE_DATE"] = pd.to_datetime(df["PRICE_DATE"], errors="coerce").dt.date
    df = df.dropna(subset=["PRICE_DATE"])

    # Ensure columns exist even if Yahoo removes them
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Remove duplicates & sort
    df = df.drop_duplicates(subset=["PRICE_DATE"], keep="last")
    df = df.sort_values("PRICE_DATE").reset_index(drop=True)

    logger.info(f"Downloaded {len(df)} rows successfully.")
    return df

# -------------------------
# Upsert/Append
# -------------------------
def append_to_snowflake(session, df: pd.DataFrame):
    if df is None or df.empty:
        logger.info("No new rows to upload.")
        return 0

    logger.info(f"Uploading {len(df)} rows to Snowflake table {FULL_TABLE_IDENTIFIER} ...")

    try:
        df_upload = df.copy()
        df_upload["PRICE_DATE"] = pd.to_datetime(df_upload["PRICE_DATE"])

        result = session.write_pandas(
            df_upload,
            table_name=TABLE_NAME,
            database=SNOWFLAKE_CONFIG['database'],
            schema=SNOWFLAKE_CONFIG['schema'],
            auto_create_table=True
        )

        logger.info(f"write_pandas result: {result}")

        rows_inserted = result.get("rows_inserted") or result.get("number_of_rows_loaded")
        chunks = result.get("chunks") or result.get("chunks_count")

        logger.info(f"Inserted rows: {rows_inserted}, Chunks: {chunks}")

        return int(rows_inserted or 0)

    except Exception:
        logger.exception("Failed to write to Snowflake.")
        raise


# -------------------------
# Main process
# -------------------------
def main():
    logger.info("Starting incremental gold data ingestion...")

    session = get_snowflake_session()

    last_date = get_max_date_in_table(session)
    if last_date is None:
        # table missing or empty — choose a start date far back
        logger.info("No existing data. Will download full history from 1990-01-01.")
        start_date = "1990-01-01"
    else:
        logger.info(f"Latest date in Snowflake is: {last_date}")
        # start from next calendar day
        start_dt = last_date + timedelta(days=1)
        start_date = start_dt.isoformat()

    # Set end_date = tomorrow so yfinance returns up to today
    end_date_dt = datetime.utcnow().date() + timedelta(days=1)
    end_date = end_date_dt.isoformat()

    # If start_date >= end_date -> nothing to fetch
    if pd.to_datetime(start_date).date() >= pd.to_datetime(end_date).date():
        logger.info("No new data to download. Exiting.")
        return

    # Download only the required window
    df_new = download_gc_data(start_date=start_date, end_date=end_date)

    # If no rows returned, exit
    if df_new.empty:
        logger.info("No new rows fetched from Yahoo. Exiting.")
        return

    # If last_date exists, drop any rows <= last_date (safety)
    if last_date is not None:
        df_new = df_new[df_new["PRICE_DATE"] > last_date]

    if df_new.empty:
        logger.info("After filtering duplicates, no new rows remain. Exiting.")
        return

    # Append to Snowflake
    inserted = append_to_snowflake(session, df_new)
    logger.info(f"Ingestion finished. Inserted rows: {inserted}")

    # Optionally: close session
    session.close()
    logger.info("Snowflake session closed. Done.")


if __name__ == "__main__":
    main()
