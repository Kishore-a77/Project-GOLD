from snowflake.snowpark import Session
import pandas as pd
from snowflake.snowpark.functions import col
from utils.config import get_snowflake_session


def load_processed_data():
    session = get_snowflake_session()

    # Load table from Snowflake
    df = (
        session.table("GOLD_PROJECT.PROCESSED.MASTER_GOLD_DATA")
        .sort(col("DATE"))      # Correct column name (your table uses uppercase DATE)
        .to_pandas()
    )

    # Ensure DATE column is parsed as datetime
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    else:
        raise ValueError("DATE column not found in MASTER_GOLD_DATA Snowflake table.")

    return df
