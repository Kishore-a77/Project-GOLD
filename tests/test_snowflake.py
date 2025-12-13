from utils.config import get_snowflake_session

session = get_snowflake_session()

try:
    df = session.table("GOLD_PROJECT.PROCESSED.MASTER_GOLD_DATA").to_pandas()
    print("Rows:", len(df))
    print(df.head())
except Exception as e:
    print("ERROR:", e)
