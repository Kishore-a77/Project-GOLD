"""
Ensemble Service
Combines Chronos and NHITS predictions with configurable weights.
"""

import os
from supabase import create_client
import pandas as pd
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_latest_predictions(horizon='30d'):
    """Fetch latest predictions from Supabase."""
    response = (supabase.table('predictions')
                .select('*')
                .eq('horizon', horizon)
                .order('date')
                .execute())
    return pd.DataFrame(response.data)


def compute_ensemble(chronos_pred, nhits_pred, chronos_weight=0.5):
    """Weighted ensemble of two predictions."""
    return chronos_weight * chronos_pred + (1 - chronos_weight) * nhits_pred


def update_ensemble_predictions(horizon='30d', chronos_weight=0.5):
    """Recompute ensemble with custom weights and update Supabase."""
    df = get_latest_predictions(horizon)
    
    for _, row in df.iterrows():
        ensemble = compute_ensemble(
            row['chronos_pred'], 
            row['nhits_pred'], 
            chronos_weight
        )
        
        supabase.table('predictions').update({
            'ensemble_pred': round(ensemble, 2)
        }).eq('id', row['id']).execute()
    
    print(f"Updated ensemble for {len(df)} predictions")


if __name__ == "__main__":
    update_ensemble_predictions()