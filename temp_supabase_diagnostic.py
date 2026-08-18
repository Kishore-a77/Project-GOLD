import os
import socket
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

url_exists = bool(url)
key_exists = bool(key)

scheme_valid = False
hostname = None

if url:
    parsed = urllib.parse.urlparse(url)
    scheme_valid = parsed.scheme.lower() == "https"
    hostname = parsed.hostname

dns_success = False

if hostname:
    try:
        socket.gethostbyname(hostname)
        dns_success = True
    except Exception:
        dns_success = False

https_success = False

if url and scheme_valid and hostname:
    try:
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": "Kilo/1.0"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            https_success = True
    except urllib.request.HTTPError:
        # HTTP response means the server was reachable.
        https_success = True
    except Exception:
        https_success = False

print(f"SUPABASE_URL exists: {url_exists}")
print(f"SUPABASE_KEY exists: {key_exists}")
print(f"Scheme valid: {scheme_valid}")
print(f"Hostname: {hostname if hostname else ''}")
print(f"DNS resolution: {'SUCCESS' if dns_success else 'FAILED'}")
print(f"HTTPS connectivity: {'SUCCESS' if https_success else 'FAILED'}")