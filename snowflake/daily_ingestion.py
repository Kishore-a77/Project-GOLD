# snowflake/daily_ingestion.py
from snowflake.snowpark import Session
from datetime import date
import pandas as pd
from snowflake.snowpark.functions import col
from app.snowflake.snowflake_connection import get_session
from app.utils.data_fetchers import fetch_latest_gold_data

def run_daily_ingestion():
    session = get_session()

    latest_df = fetch_latest_gold_data()  # returns pandas df
    sf_df = session.create_dataframe(latest_df)

    sf_df.write.mode("append").save_as_table(
        "GOLD_PROJECT.RAW_DATA.GOLD_PRICES"
    )

    session.close()
