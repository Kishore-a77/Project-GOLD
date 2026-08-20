"""One-off sanity check: run this locally (with your .env) to confirm the
backfill left no gaps and the row count/date range look right."""
import os
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Total row count
count_resp = supabase.table("gold_prices").select("date", count="exact").execute()
print(f"Total rows: {count_resp.count}")

# Earliest / latest dates
earliest = supabase.table("gold_prices").select("date").order("date", desc=False).limit(1).execute()
latest = supabase.table("gold_prices").select("date").order("date", desc=True).limit(1).execute()
print(f"Earliest date: {earliest.data[0]['date']}")
print(f"Latest date:   {latest.data[0]['date']}")

# Gap check: fetch all dates and look for jumps > 4 calendar days
# (covers weekends + a holiday; flags anything bigger as a real gap)
all_rows = supabase.table("gold_prices").select("date").order("date").execute().data
dates = [date.fromisoformat(r["date"]) for r in all_rows]
gaps = [(dates[i], dates[i+1]) for i in range(len(dates)-1) if (dates[i+1]-dates[i]).days > 4]
if gaps:
    print(f"⚠ Found {len(gaps)} suspicious gap(s):")
    for a, b in gaps[:20]:
        print(f"   {a} -> {b}  ({(b-a).days} days)")
else:
    print("✅ No gaps > 4 calendar days found.")