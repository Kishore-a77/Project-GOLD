# snowflake/update_features.py
from app.snowflake.snowflake_connection import get_session

def update_features():
    session = get_session()

    session.sql("""
        INSERT OVERWRITE INTO GOLD_PROJECT.PROCESSED.MASTER_GOLD_DATA
        SELECT
            *,
            AVG(gold_close) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS sma_20,
            ...
        FROM GOLD_PROJECT.RAW_DATA.GOLD_PRICES
    """).collect()

    session.close()
