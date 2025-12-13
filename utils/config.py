import os
from snowflake.snowpark import Session
from dotenv import load_dotenv

load_dotenv()

def get_snowflake_session():
    connection_parameters = {
        "ACCOUNT": os.getenv("SNOWFLAKE_ACCOUNT"),
        "USER": os.getenv("SNOWFLAKE_USER"),
        "PASSWORD": os.getenv("SNOWFLAKE_PASSWORD"),
        "ROLE": os.getenv("SNOWFLAKE_ROLE"),
        "WAREHOUSE": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "DATABASE": os.getenv("SNOWFLAKE_DATABASE"),
        "SCHEMA": os.getenv("SNOWFLAKE_SCHEMA"),
    }
    return Session.builder.configs(connection_parameters).create()
