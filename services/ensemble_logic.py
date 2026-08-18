"""
Lightweight ensemble logic for the prediction pipeline.

Kept separate from app.models.ensemble_model (the legacy SQLite version) so the
daily pipeline can combine forecasts WITHOUT importing torch / darts / psycopg2.
Only numpy and pandas are required.
"""
import numpy as np
import pandas as pd

# Ensemble weights (chronos, nhits) per horizon.
WEIGHTS = {
    "next_day": (0.7, 0.3),
    "next_week": (0.6, 0.4),
    "next_month": (0.5, 0.5),
}


def combine(c_vals, n_vals, wc, wn):
    """
    Combines Chronos + NHITS predictions safely.
    Removes NaNs and ensures stable output.
    """
    c_vals = np.array(c_vals, dtype=float)
    n_vals = np.array(n_vals, dtype=float)

    L = max(len(c_vals), len(n_vals))

    if len(c_vals) < L:
        c_vals = np.concatenate([c_vals, np.full(L - len(c_vals), np.nan)])

    if len(n_vals) < L:
        n_vals = np.concatenate([n_vals, np.full(L - len(n_vals), np.nan)])

    result = []
    for c, n in zip(c_vals, n_vals):
        if not np.isnan(c) and not np.isnan(n):
            value = wc * c + wn * n
        elif not np.isnan(c):
            value = c
        elif not np.isnan(n):
            value = n
        else:
            value = np.nan
        result.append(value)

    # forward-fill NaNs if any
    result = pd.Series(result).ffill().bfill().tolist()
    return result
