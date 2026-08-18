import os
from dotenv import load_dotenv
load_dotenv()
url_exists = os.getenv("SUPABASE_URL") is not None
key_exists = os.getenv("SUPABASE_KEY") is not None
print("SUPABASE_URL exists:", url_exists)
print("SUPABASE_KEY exists:", key_exists)