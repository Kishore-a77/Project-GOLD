#!/usr/bin/env python3
"""
Weekly Training Pipeline
Runs Sunday night: retrain NHITS, evaluate, and promote if better.
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_fetch_service import run_data_fetch
from services.feature_service import run_feature_engineering
from services.model_service import save_model_metadata, get_active_model
from app.models.day10_nhits import (
    train_nhits, 
    prepare_series, 
    prepare_dataframe, 
    load_processed_data_supabase,
    predict_nhits
)

def evaluate_model(model, ts, scaler):
    """Evaluate model on test window and return metrics."""
    # Use the same logic as in day10_nhits.py main_train_and_predict
    PRED_LEN = 30
    INPUT_LEN = 90
    VAL_LEN = 200
    TEST_LEN = 30
    
    train_ts = ts[: -(VAL_LEN + TEST_LEN)]
    val_ts = ts[-(VAL_LEN + TEST_LEN) : -TEST_LEN]
    test_ts = ts[-TEST_LEN:]
    
    pred_scaled = model.predict(n=len(test_ts), series=ts)
    pred_inv = scaler.inverse_transform(pred_scaled.all_values().reshape(-1, 1))
    actual_inv = scaler.inverse_transform(test_ts.all_values().reshape(-1, 1))
    
    pred_series = TimeSeries.from_times_and_values(
        test_ts.time_index, pred_inv.reshape(-1)
    )
    actual_series = TimeSeries.from_times_and_values(
        test_ts.time_index, actual_inv.reshape(-1)
    )
    
    return {
        'mae': mae(actual_series, pred_series),
        'rmse': rmse(actual_series, pred_series),
        'mape': mape(actual_series, pred_series)
    }

def get_model_artifacts_path():
    """Get the path where NHiTS model artifacts are stored."""
    ROOT = Path(__file__).resolve().parents[2]
    return ROOT / "models" / "nhits_model"

def is_better_model(new_metrics, current_metrics):
    """
    Determine if new model is better than current model.
    Lower MAE is better (since it's error).
    If no current model, new model is considered better.
    """
    if current_metrics is None:
        return True
    
    # Compare MAE (lower is better)
    return new_metrics['mae'] < current_metrics['mae']

def main():
    print("=" * 60)
    print("PROJECT GOLD - WEEKLY TRAINING PIPELINE")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    try:
        # Step 1: Ensure we have latest data
        print("\n📥 STEP 1: Fetching latest data...")
        run_data_fetch()
        
        # Step 2: Rebuild model-ready dataset
        print("\n⚙️  STEP 2: Rebuilding feature dataset...")
        run_feature_engineering()
        
        # Step 3: Load data for training and evaluation
        print("\n📊 STEP 3: Loading data for training and evaluation...")
        df = load_processed_data_supabase()
        df = prepare_dataframe(df)
        ts, scaler = prepare_series(df)
        
        # Step 4: Train new model
        print("\n🧠 STEP 4: Training new NHiTS model...")
        
        # Use the same parameters as in day10_nhits.py
        PRED_LEN = 30
        INPUT_LEN = 90
        VAL_LEN = 200
        TEST_LEN = 30
        
        train_ts = ts[: -(VAL_LEN + TEST_LEN)]
        val_ts = ts[-(VAL_LEN + TEST_LEN) : -TEST_LEN]
        test_ts = ts[-TEST_LEN:]
        
        print(f"[NHITS] train={len(train_ts)} val={len(val_ts)} test={len(test_ts)}")
        
        model = train_nhits(ts)
        
        # Step 5: Evaluate new model
        print("\n📈 STEP 5: Evaluating new model...")
        new_metrics = evaluate_model(model, ts, scaler)
        print(f"[NHITS] New model - MAE: {new_metrics['mae']:.4f}, RMSE: {new_metrics['rmse']:.4f}, MAPE: {new_metrics['mape']:.4f}")
        
        # Step 6: Get current active model metrics
        print("\n🔍 STEP 6: Checking current active model...")
        current_model_data = get_active_model('nhits')
        current_metrics = None
        
        if current_model_data:
            current_metrics = {
                'mae': current_model_data.get('mae', float('inf')),
                'rmse': current_model_data.get('rmse', float('inf')),
                'mape': current_model_data.get('mape', float('inf'))
            }
            print(f"[NHITS] Current model - MAE: {current_metrics['mae']:.4f}, RMSE: {current_metrics['rmse']:.4f}, MAPE: {current_metrics['mape']:.4f}")
        else:
            print("[NHITS] No current active model found")
        
        # Step 6: Determine if we should promote the new model
        if is_better_model(new_metrics, current_metrics):
            print("\n✅ STEP 6: New model is better! Promoting to active model...")
            
            # Save the new model to the expected location
            MODEL_DIR = get_model_artifacts_path()
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            save_path = str(MODEL_DIR / "nhits_model_vfinal")
            model.save(save_path)
            print(f"[NHITS] Model saved to: {save_path}")
            
            # Save metadata
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_model_metadata(
                model_name='nhits',
                version=version,
                metrics=new_metrics,
                artifact_path=save_path
            )
            
            print(f"\n🏆 WEEKLY TRAINING COMPLETE - NEW MODEL PROMOTED")
            print(f"   Version: {version}")
            print(f"   MAE: {new_metrics['mae']:.4f}")
            print(f"   RMSE: {new_metrics['rmse']:.4f}")
            print(f"   MAPE: {new_metrics['mape']:.4f}")
        else:
            print("\n❌ STEP 6: New model is not better than current model. Keeping current model.")
            print(f"   New model MAE: {new_metrics['mae']:.4f}")
            print(f"   Current model MAE: {current_metrics['mae']:.4f if current_metrics else 'N/A'}")
            
        print("=" * 60)
        
    except Exception as e:
        print(f"\n💥 WEEKLY TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()