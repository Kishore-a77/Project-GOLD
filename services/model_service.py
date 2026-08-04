"""
Model Service
Handles model artifact storage, versioning, and loading.
For GitHub Actions, we store models in the repo or GitHub Releases.
"""

import os
import json
from datetime import datetime
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_DIR = "models/nhits_model"
METADATA_FILE = "models/model_metadata.json"


def save_model_metadata(model_name, version, metrics, artifact_path):
    """Save model training metadata to Supabase."""
    record = {
        'model_name': model_name,
        'version': version,
        'mae': metrics.get('mae'),
        'rmse': metrics.get('rmse'),
        'mape': metrics.get('mape'),
        'artifact_path': artifact_path,
        'is_active': True
    }
    
    # Deactivate old versions
    supabase.table('model_metadata').update({'is_active': False}).eq('model_name', model_name).execute()
    
    # Insert new version
    supabase.table('model_metadata').insert(record).execute()
    print(f"Saved metadata for {model_name} v{version}")


def get_active_model(model_name):
    """Get the currently active model version."""
    response = (supabase.table('model_metadata')
                .select('*')
                .eq('model_name', model_name)
                .eq('is_active', True)
                .single()
                .execute())
    return response.data


if __name__ == "__main__":
    # Test
    print(get_active_model('nhits'))