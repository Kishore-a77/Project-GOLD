import os
from dotenv import load_dotenv
from snowflake.snowpark import Session

# Load .env file
load_dotenv(r"C:\Users\kisho\OneDrive\Desktop\GOLD\.env")

def get_session():
    connection_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA")
    }

    print("Connecting to:", connection_params["account"])
    print("Using schema:", connection_params["schema"])

    return Session.builder.configs(connection_params).create()
