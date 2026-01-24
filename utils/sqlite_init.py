# utils/sqlite_init.py
import sqlite3
import os

DB_PATH = "database/gold_data.db"

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS gold_prices (
    date TEXT PRIMARY KEY,
    gold_close REAL,
    usd_index REAL,
    fx_inr REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS features (
    date TEXT PRIMARY KEY,
    gold_close REAL,
    fx_inr REAL,
    sma_20 REAL,
    sma_50 REAL,
    rsi REAL,
    macd REAL
)
""")

conn.commit()
conn.close()

print("✅ SQLite database initialized successfully")
print("📁 Location: database/gold_data.db")
