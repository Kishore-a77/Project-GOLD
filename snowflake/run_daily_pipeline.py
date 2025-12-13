# snowflake/run_daily_pipeline.py
from daily_ingestion import run_daily_ingestion
from update_features import update_features

if __name__ == "__main__":
    run_daily_ingestion()
    update_features()
