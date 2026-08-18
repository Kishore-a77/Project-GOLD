"""
app/db/supabase_client.py

Provides client initializations and connection helpers for:
1. Supabase REST/GraphQL API Client (via supabase-py)
2. PostgreSQL Direct Database Connection (via psycopg2)
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from supabase import Client, create_client
import psycopg2
from psycopg2.extensions import connection
from sqlalchemy import create_engine, text

# Load .env variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# SQLAlchemy engine (works with pandas read_sql / to_sql out of the box)
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    engine = None

@lru_cache(maxsize=1)
def get_supabase_client(use_service_role: bool = False) -> Client:
    """
    Returns a singleton authenticated Supabase client.
    
    Args:
        use_service_role: If True, uses the service_role key (which bypasses RLS).
                          Otherwise, uses the anon/public key.
    """
    url = SUPABASE_URL
    key = SUPABASE_SERVICE_KEY if use_service_role else SUPABASE_KEY

    if not url:
        raise ValueError("SUPABASE_URL is missing from environment variables.")
    if not key:
        raise ValueError(f"SUPABASE_{'SERVICE_' if use_service_role else ''}KEY is missing from environment variables.")

    # Remove trailing rest/v1/ if present to avoid client build issues
    clean_url = url.rstrip("/")
    if clean_url.endswith("/rest/v1"):
        clean_url = clean_url[:-8].rstrip("/")

    return create_client(clean_url, key)

# Default supabase client (uses anon/public key by default)
supabase = get_supabase_client(use_service_role=False)

# Admin supabase client (uses service_role key to bypass RLS)
def get_supabase_admin() -> Client:
    """
    Returns a Supabase client with service_role privileges.
    """
    return get_supabase_client(use_service_role=True)

def get_postgres_connection() -> connection:
    """
    Creates and returns a raw connection to the PostgreSQL database
    using psycopg2 and the DATABASE_URL.
    
    Remember to close the connection after use.
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing from environment variables.")
    
    return psycopg2.connect(DATABASE_URL)

def test_connections():
    """
    Tests both the Supabase API and direct PostgreSQL connections.
    """
    print("=== Testing Database Connections ===")
    
    # 1. Test Supabase API
    try:
        print("Testing Supabase API connection...")
        client = get_supabase_client()
        # Fetching a single row from a common table (e.g. gold_prices) to test actual connection
        response = client.table("gold_prices").select("*").limit(1).execute()
        print(f"[SUCCESS] Supabase API connection successful! Fetched {len(response.data)} row(s).")
    except Exception as e:
        print(f"[ERROR] Supabase API client connection failed: {e}")

    # 2. Test PostgreSQL direct connection
    try:
        print("Testing PostgreSQL direct connection...")
        conn = get_postgres_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()
            print(f"[SUCCESS] PostgreSQL connection successful!")
            print(f"   DB Version: {db_version[0]}")
        conn.close()
    except Exception as e:
        print(f"[ERROR] PostgreSQL connection failed: {e}")
    print("====================================")

def ensure_schema(sql_path: str = None) -> bool:
    """
    Create required tables if they do not exist (idempotent).

    This uses the direct PostgreSQL connection (DATABASE_URL), which bypasses
    Row-Level Security, so it is the most reliable way to bootstrap the schema.
    Returns True if it ran, False if DATABASE_URL is unavailable (in which case
    the tables must already exist in Supabase).
    """
    if engine is None:
        print("[WARN] DATABASE_URL not set; skipping schema bootstrap. "
              "Tables must already exist in Supabase.")
        return False

    if sql_path is None:
        sql_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "database", "supabase_schema.sql"
        )

    if not os.path.exists(sql_path):
        print(f"[WARN] Schema file not found: {sql_path}; skipping bootstrap.")
        return False

    with open(sql_path, "r") as f:
        sql = f.read()

    statements = [s.strip() for s in sql.split(";") if s.strip()]

    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()

    print(f"[OK] Schema ensured from {sql_path} ({len(statements)} statements)")
    return True


if __name__ == "__main__":
    test_connections()
