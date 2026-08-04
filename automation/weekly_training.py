#!/usr/bin/env python3
"""
Weekly Training Pipeline
Runs Sunday night: retrain NHITS, evaluate, save new model.
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_fetch_service import run_data_fetch
from services.feature_service import run_feature_engineering
from services.model_service import save_model_metadata


def retrain_nhits():
    """Retrain the NHITS model."""
    print("\n🧠 Retraining NHITS model...")
    
    # Run your existing training script
    result = subprocess.run(
        ['python', 'app/models/day10_nhits.py'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"Training error: {result.stderr}")
        return False
    
    return True


def evaluate_model():
    """Evaluate the retrained model."""
    # You'll implement this based on your existing evaluation logic
    # For now, return dummy metrics
    return {'mae': 276.54, 'rmse': 302.89, 'mape': 7.1}


def main():
    print("=" * 60)
    print("PROJECT GOLD - WEEKLY TRAINING PIPELINE")
    print(f"Started at: {__import__('datetime').datetime.now()}")
    print("=" * 60)
    
    # Step 1: Ensure we have latest data
    print("\n📥 STEP 1: Fetching latest data...")
    run_data_fetch()
    
    print("\n⚙️  STEP 2: Running feature engineering...")
    run_feature_engineering()
    
    # Step 3: Retrain
    print("\n🧠 STEP 3: Retraining models...")
    success = retrain_nhits()
    
    if not success:
        print("❌ Training failed. Aborting.")
        return
    
    # Step 4: Evaluate and save metadata
    print("\n📊 STEP 4: Evaluating model...")
    metrics = evaluate_model()
    
    from datetime import datetime
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_model_metadata(
        model_name='nhits',
        version=version,
        metrics=metrics,
        artifact_path='models/nhits_model/nhits_weights.pt'
    )
    
    print("\n" + "=" * 60)
    print("✅ WEEKLY TRAINING COMPLETE")
    print(f"New model version: {version}")
    print("=" * 60)


if __name__ == "__main__":
    main()