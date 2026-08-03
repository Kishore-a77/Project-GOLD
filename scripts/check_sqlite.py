# scripts/check_sqlite.py
import sqlite3
import os

DB_PATH = "database/gold_data.db"

print(f"Looking for: {os.path.abspath(DB_PATH)}")
print(f"File exists: {os.path.exists(DB_PATH)}")

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\nTables found: {[t[0] for t in tables]}")
    
    # If features exists, show sample
    if any(t[0] == 'features' for t in tables):
        df = pd.read_sql("SELECT * FROM features LIMIT 3", conn)
        print(f"\nSample data:\n{df}")
    else:
        print("\nNo 'features' table found!")
        print("Available tables:", [t[0] for t in tables])
    
    conn.close()
else:
    print("\nDatabase file not found!")
    # Search for any .db files
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".db"):
                print(f"Found: {os.path.join(root, f)}")