import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("database/gold_data.db")

def load_processed_data_sqlite():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT *
        FROM features
        ORDER BY date
        """,
        conn
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    return df
