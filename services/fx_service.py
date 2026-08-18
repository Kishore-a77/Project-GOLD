"""
FX Service — USD→INR rate lookup.

Kept independent of Streamlit so it can be unit-tested in lightweight
environments. The dashboard layer adds the user-facing warning.
"""
import requests

# Fallback rate used only when the live FX API is unavailable.
FALLBACK_USD_INR = 83.0

FX_API_URL = "https://api.frankfurter.app/latest?from=USD&to=INR"


def fetch_usd_inr_rate(timeout: int = 10) -> float:
    """Fetch the live USD→INR rate. Raises requests.RequestException on failure."""
    r = requests.get(FX_API_URL, timeout=timeout)
    r.raise_for_status()
    return float(r.json()["rates"]["INR"])


def get_usd_inr_rate() -> float:
    """Return the USD→INR rate, falling back to a constant if the API is down.

    Never raises — a missing FX rate must not break the dashboard. The caller is
    responsible for surfacing the fallback to the user if desired.
    """
    try:
        return fetch_usd_inr_rate()
    except requests.RequestException:
        return FALLBACK_USD_INR
