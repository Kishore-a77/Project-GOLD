import pandas as pd
from datetime import datetime
from snowflake.snowpark import Session
from dotenv import load_dotenv
import os

load_dotenv()

def get_session():
    connection_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": "ACCOUNTADMIN",
        "warehouse": "GOLD_WH",
        "database": "GOLD_PROJECT",
        "schema": "PROCESSED"
    }
    return Session.builder.configs(connection_params).create()

session = get_session()

def save_forecast(horizon, values, auto_create_table=False):
    df = pd.DataFrame([{
        "RUN_TIMESTAMP": datetime.now(),
        "HORIZON": horizon,
        "FORECAST": values
    }])

    print(df)

    session.write_pandas(
        df,
        table_name="CHRONOS_FORECAST",
        schema="PROCESSED",
        overwrite=False,
        auto_create_table=auto_create_table
    )
