#!/usr/bin/env python3
"""
Daily Pipeline
Runs every morning: fetch data → features → predictions → save to Supabase
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_fetch_service import run_data_fetch
from services.feature_service import run_feature_engineering
from services.prediction_service import run_prediction_pipeline, record_run_status


def main():
    started = datetime.now()
    print("=" * 60)
    print("PROJECT GOLD - DAILY PIPELINE")
    print(f"Started at: {started}")
    print("=" * 60)
    
    records_processed = 0
    predictions_generated = 0
    
    try:
        # Step 1: Fetch latest data
        print("\n📥 STEP 1: Fetching latest data...")
        records_processed = run_data_fetch() or 0
        
        # Step 2: Feature engineering
        print("\n⚙️  STEP 2: Running feature engineering...")
        run_feature_engineering()
        
        # Step 3: Generate predictions
        print("\n🔮 STEP 3: Running predictions...")
        _, _, _, predictions_generated = run_prediction_pipeline()
        
        record_run_status(
            started,
            success=True,
            records_processed=records_processed,
            predictions_generated=predictions_generated,
        )
        
        print("\n" + "=" * 60)
        print("✅ DAILY PIPELINE COMPLETE")
        print(f"   Records processed: {records_processed}")
        print(f"   Predictions generated: {predictions_generated}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n💥 DAILY PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        try:
            record_run_status(
                started,
                success=False,
                error=str(e),
                records_processed=records_processed,
                predictions_generated=0,
            )
        except Exception as se:
            print(f"Failed to record run status: {se}")
        sys.exit(1)


if __name__ == "__main__":
    main()