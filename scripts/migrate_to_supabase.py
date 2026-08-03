# scripts/migrate_to_supabase.py
import sqlite3
import pandas as pd
import sys
from pathlib import Path

# Add project root to path so we can import the Supabase client
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.supabase_client import engine

# 1. Connect to SQLite
sqlite_path = ROOT / "database" / "gold_data.db"
print(f"Connecting to SQLite: {sqlite_path}")
print(f"File exists: {sqlite_path.exists()}")

conn = sqlite3.connect(str(sqlite_path))

# 2. Read data
print("Reading features table...")
features = pd.read_sql_query("SELECT * FROM features", conn)

print("Reading ensemble_forecast table...")
try:
    ensemble = pd.read_sql_query("SELECT * FROM ensemble_forecast", conn)
    print(f"Ensemble rows found: {len(ensemble)}")
except Exception as e:
    print(f"No ensemble_forecast table yet: {e}")
    ensemble = pd.DataFrame(columns=["date", "ensemble_pred"])

conn.close()

print(f"Features rows: {len(features)}")
print(f"Features columns: {list(features.columns)}")
print(features.head(3))

# 3. Convert date columns
for df in [features, ensemble]:
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"])

# 4. Save to Supabase
print("\nSaving to Supabase...")

# Features: replace entire table
features.to_sql("features", engine, if_exists="replace", index=False)
print(f"✓ Loaded {len(features)} rows into features")

# Ensemble: only if we have data
if not ensemble.empty:
    ensemble.to_sql("ensemble_forecast", engine, if_exists="replace", index=False)
    print(f"✓ Loaded {len(ensemble)} rows into ensemble_forecast")
else:
    print("⚠ No ensemble data to migrate")

print("\n[SUCCESS] Migration complete!")