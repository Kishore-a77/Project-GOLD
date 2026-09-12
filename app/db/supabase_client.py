"""
app/db/supabase_client.py

Centralized, resilient Supabase client initialization and database helpers.
Supports:
1. Streamlit Cloud (st.secrets) & local (.env) configuration.
2. Robust timeouts and socket configuration (IPv4 prioritization on Windows to prevent WinError 10060).
3. Bounded retries with exponential backoff for transient network issues.
4. Typed exception classification (Config, Connection, API).
5. Backward-compatible interfaces for existing pipeline and service modules.
"""

import os
import sys
import time
import socket
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from dotenv import load_dotenv
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

logger = logging.getLogger("supabase_client")

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION LOADING
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
else:
    load_dotenv(override=False)


class DatabaseError(Exception):
    """Base exception for all database access failures."""
    pass


class DatabaseConfigError(DatabaseError):
    """Raised when Supabase configuration is missing or malformed."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when a transient or permanent network/connection error occurs."""
    pass


class DatabaseAPIError(DatabaseError):
    """Raised when the database/PostgREST returns an API error response."""
    pass


def get_credentials() -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Retrieve Supabase credentials from Streamlit secrets (if available) or environment.
    Never exposes secrets in logs or exceptions.

    Returns:
        (supabase_url, supabase_key, supabase_service_key, database_url)
    """
    url = None
    key = None
    service_key = None
    db_url = None

    # Try Streamlit secrets first if running within Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = st.secrets.get("SUPABASE_URL")
            key = st.secrets.get("SUPABASE_KEY")
            service_key = st.secrets.get("SUPABASE_SERVICE_KEY")
            db_url = st.secrets.get("DATABASE_URL")
    except Exception:
        pass

    # Fallback to environment variables
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_KEY")
    service_key = service_key or os.getenv("SUPABASE_SERVICE_KEY")
    db_url = db_url or os.getenv("DATABASE_URL")

    if not url:
        raise DatabaseConfigError("SUPABASE_URL is missing. Please check your environment or secrets.")
    
    url = url.strip().rstrip("/")
    if url.endswith("/rest/v1"):
        url = url[:-8].rstrip("/")

    if not (url.startswith("http://") or url.startswith("https://")):
        raise DatabaseConfigError("SUPABASE_URL must begin with http:// or https://")

    # If anon key is missing, fall back to service key (or vice-versa for read access)
    effective_key = key or service_key
    if not effective_key:
        raise DatabaseConfigError("SUPABASE_KEY or SUPABASE_SERVICE_KEY is missing.")

    return url, effective_key, service_key, db_url


# ---------------------------------------------------------------------------
# 2. SOCKET & WINDOWS DUAL-STACK NETWORK ADAPTER
# ---------------------------------------------------------------------------
# On Windows, dual-stack DNS resolution returning IPv6 (AF_INET6) on networks
# without IPv6 routing causes socket connect calls to hang until TCP SYN
# retransmission timeout (~21s), producing WinError 10060.
# We patch socket resolution to prioritize AF_INET when resolving Supabase hosts.
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


def _ipv4_preferring_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        # If host is a Supabase domain, force/prioritize AF_INET (IPv4)
        if isinstance(host, str) and "supabase.co" in host.lower():
            res = _ORIGINAL_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)
            if res:
                return res
    except Exception:
        pass
    return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)


try:
    socket.getaddrinfo = _ipv4_preferring_getaddrinfo
except Exception:
    pass


from supabase.client import ClientOptions

# ---------------------------------------------------------------------------
# 3. CLIENT FACTORY
# ---------------------------------------------------------------------------
_CLIENT_CACHE: Dict[bool, Client] = {}


def _build_client(use_service_role: bool = False) -> Client:
    url, anon_key, service_key, _ = get_credentials()
    # Prefer service_key if explicitly requested or whenever available, to ensure
    # RLS does not silently block reads on pipeline_runs or model_metadata tables.
    chosen_key = service_key if (use_service_role and service_key) else (service_key or anon_key)

    # Custom options with explicit 30-second timeout
    options = ClientOptions(
        postgrest_client_timeout=30.0,
        storage_client_timeout=30,
        auto_refresh_token=False,
        persist_session=False,
    )
    return create_client(url, chosen_key, options=options)


def get_supabase_client(use_service_role: bool = False) -> Client:
    """
    Returns a cached singleton Supabase client.
    Reuses one initialized client across Streamlit reruns and sessions.
    """
    if use_service_role not in _CLIENT_CACHE:
        _CLIENT_CACHE[use_service_role] = _build_client(use_service_role=use_service_role)
    return _CLIENT_CACHE[use_service_role]


def clear_client_cache():
    """Clear cached client instances to force re-initialization."""
    _CLIENT_CACHE.clear()


def get_supabase_admin() -> Client:
    """Returns a Supabase client configured with service_role privileges."""
    return get_supabase_client(use_service_role=True)


# Default client instance for backward compatibility
try:
    supabase = get_supabase_client(use_service_role=False)
except Exception:
    # If environment is not configured yet during build/import, defer error to call-time
    supabase = None


# ---------------------------------------------------------------------------
# 4. ROBUST QUERY EXECUTION WITH BOUNDED RETRIES
# ---------------------------------------------------------------------------
TRANSIENT_ERRORS = (
    TimeoutError,
    socket.timeout,
    ConnectionResetError,
    ConnectionRefusedError,
    ConnectionAbortedError,
    OSError,
)

try:
    import httpx
    TRANSIENT_ERRORS = TRANSIENT_ERRORS + (
        httpx.ConnectTimeout,
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
    )
except ImportError:
    pass


def execute_with_retry(
    query_fn: Callable[[], Any],
    max_retries: int = 3,
    backoff_base: float = 1.0,
    operation_name: str = "database query"
) -> Any:
    """
    Execute a Supabase PostgREST query or callable with bounded retries and exponential backoff.
    
    Catches transient network errors (including WinError 10060) and retries.
    Distinguishes config errors, connection errors, and API errors.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return query_fn()
        except DatabaseConfigError:
            raise
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            is_winerror_10060 = "10060" in str(exc)
            logger.warning(
                "Transient error during %s (attempt %d/%d)%s: %s",
                operation_name,
                attempt,
                max_retries,
                " [WinError 10060]" if is_winerror_10060 else "",
                str(exc)
            )
            if attempt < max_retries:
                delay = backoff_base * (2 ** (attempt - 1))
                time.sleep(delay)
        except Exception as exc:
            # Check if inner cause is a transient error
            cause = getattr(exc, "__cause__", None) or getattr(exc, "args", [None])[0]
            if isinstance(cause, TRANSIENT_ERRORS) or "10060" in str(exc):
                last_error = exc
                logger.warning(
                    "Nested transient error during %s (attempt %d/%d): %s",
                    operation_name, attempt, max_retries, str(exc)
                )
                if attempt < max_retries:
                    delay = backoff_base * (2 ** (attempt - 1))
                    time.sleep(delay)
                    continue
            # Non-transient API error
            raise DatabaseAPIError(f"PostgREST query error in {operation_name}: {exc}") from exc

    raise DatabaseConnectionError(
        f"Failed to connect to Supabase after {max_retries} attempts ({operation_name}): {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# 5. POSTGRESQL DIRECT DB HELPERS (psycopg2 / SQLAlchemy)
# ---------------------------------------------------------------------------
def get_postgres_connection():
    """
    Creates and returns a raw connection to PostgreSQL using psycopg2.
    """
    _, _, _, db_url = get_credentials()
    if not db_url:
        raise DatabaseConfigError("DATABASE_URL is missing from environment/secrets.")
    import psycopg2
    return psycopg2.connect(db_url)


def test_connections():
    """Diagnostic helper to verify connections."""
    print("=== Testing Database Connections ===")
    try:
        client = get_supabase_client()
        res = execute_with_retry(
            lambda: client.table("gold_prices").select("date").limit(1).execute(),
            operation_name="test_connections_api"
        )
        print(f"[SUCCESS] Supabase API connection successful! Rows: {len(res.data)}")
    except Exception as e:
        print(f"[ERROR] Supabase API connection failed: {e}")

    try:
        conn = get_postgres_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"[SUCCESS] PostgreSQL direct connection successful: {version[0]}")
        conn.close()
    except Exception as e:
        print(f"[ERROR] PostgreSQL direct connection failed: {e}")
    print("====================================")


def ensure_schema(sql_path: str = None) -> bool:
    """Bootstrap schema idempotently via direct PostgreSQL connection."""
    _, _, _, db_url = get_credentials()
    if not db_url:
        logger.warning("DATABASE_URL not set; skipping schema bootstrap.")
        return False

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        logger.warning("Failed to create SQLAlchemy engine for schema bootstrap: %s", e)
        return False

    if sql_path is None:
        sql_path = str(ROOT_DIR / "database" / "supabase_schema.sql")

    if not os.path.exists(sql_path):
        logger.warning("Schema file not found: %s; skipping bootstrap.", sql_path)
        return False

    with open(sql_path, "r") as f:
        sql = f.read()

    statements = [s.strip() for s in sql.split(";") if s.strip()]

    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()

    logger.info("Schema ensured from %s (%d statements)", sql_path, len(statements))
    return True


if __name__ == "__main__":
    test_connections()
