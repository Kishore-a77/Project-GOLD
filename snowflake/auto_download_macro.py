import yfinance as yf
import pandas as pd
from snowflake.snowpark import Session
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

TICKERS = {
    "DXY_RAW": "DX-Y.NYB",   # USD Index
    "CRUDE_OIL_RAW": "CL=F", # Crude Oil
    "SP500_RAW": "^GSPC",    # S&P 500
    "USDINR_RAW": "INR=X"    # USD to INR
}

def get_session():
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

def clean_dataframe(df):
    df.reset_index(inplace=True)

    df["DATE"] = pd.to_datetime(df["Date"]).dt.date.astype(str)

    df.rename(columns={
        "Open": "OPEN",
        "High": "HIGH",
        "Low": "LOW",
        "Close": "CLOSE",
        "Volume": "VOLUME"
    }, inplace=True)

    df = df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
    return df

def create_table_if_not_exists(session, table_name):
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            DATE STRING,
            OPEN FLOAT,
            HIGH FLOAT,
            LOW FLOAT,
            CLOSE FLOAT,
            VOLUME FLOAT
        )
    """).collect()

def download_and_upload(session, table, ticker):
    logging.info(f"Downloading data for {ticker} -> table {table}")

    df = yf.download(ticker, period="20y")
    df.reset_index(inplace=True)

    # Fix column names
    df["DATE"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    df.rename(columns={
        "Open": "OPEN",
        "High": "HIGH",
        "Low": "LOW",
        "Close": "CLOSE",
        "Volume": "VOLUME"
    }, inplace=True)

    df = df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]

    # ------- 🔥 CRITICAL FIX: flatten multi-index ------
    df.columns = df.columns.get_level_values(0)
    df = df.astype(str)

    create_table_if_not_exists(session, table)

    logging.info(f"Uploading {len(df)} rows into {table}...")
    session.write_pandas(
        df,
        table_name=table,
        auto_create_table=False
    )


def main():
    session = get_session()

    for table, ticker in TICKERS.items():
        download_and_upload(session, table, ticker)

    logging.info("Macro data ingestion completed.")
    session.close()

if __name__ == "__main__":
    main()
