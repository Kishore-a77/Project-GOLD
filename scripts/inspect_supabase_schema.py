#!/usr/bin/env python3
"""
Inspect the existing Supabase database schema.
Reads DATABASE_URL from environment (or .env) and queries information_schema.
Does not expose credentials in output.
"""
import os
import sys
from datetime import datetime

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    print("Please set DATABASE_URL in .env or environment")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Please install dependencies.")
    sys.exit(1)

def inspect_schema():
    """Connect to Supabase PostgreSQL and inspect schema."""
    try:
        # Connect to the database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("=" * 80)
        print("SUPABASE DATABASE SCHEMA INSPECTION")
        print("=" * 80)
        print(f"Connection time: {datetime.now().isoformat()}")
        print()
        
        # Get list of tables in public schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in public schema.")
            return
            
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        print()
        
        # For each table, get columns
        for table in tables:
            table_name = table['table_name']
            print(f"TABLE: {table_name}")
            print("-" * 60)
            
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = cursor.fetchall()
            
            if not columns:
                print("  No columns found")
            else:
                for col in columns:
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                    max_len = f"({col['character_maximum_length']})" if col['character_maximum_length'] else ""
                    print(f"  {col['column_name']:<30} {col['data_type']}{max_len:<20} {nullable}{default}")
            print()
        
        # Also check for any views
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        views = cursor.fetchall()
        
        if views:
            print("VIEWS:")
            for view in views:
                print(f"  - {view['table_name']}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR inspecting schema: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    inspect_schema()