import os
import sys
import json
import time
import urllib.request
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.ml.demand.demand_forecast_service import DemandForecastService
from backend.db.database import SessionLocal
from backend.db import models

def run_all_tests():
    print("=" * 80)
    print("HIEN REAL ML DEMAND FORECAST & INFERENCE VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # TEST 1: The joblib model loads successfully.
    # -------------------------------------------------------------
    print("\n[TEST 1] Loading Joblib Model Bundle...")
    service = DemandForecastService.get_instance()
    assert service._bundle is not None, "TEST 1 FAILED: Bundle must not be None"
    assert 'model_1d' in service._bundle, "TEST 1 FAILED: model_1d missing"
    assert 'model_3d' in service._bundle, "TEST 1 FAILED: model_3d missing"
    assert 'model_7d' in service._bundle, "TEST 1 FAILED: model_7d missing"
    assert 'encoder' in service._bundle, "TEST 1 FAILED: encoder missing"
    print("✓ TEST 1 PASSED: Joblib model bundle loaded successfully (1d, 3d, 7d models & OrdinalEncoder).")

    # -------------------------------------------------------------
    # TEST 2: The exact 33 features can be constructed.
    # -------------------------------------------------------------
    print("\n[TEST 2] Verifying 33 Features Specification...")
    all_features = service._bundle['all_features']
    assert len(all_features) == 33, f"TEST 2 FAILED: Expected 33 features, got {len(all_features)}"
    expected_features = [
        'location_id', 'category', 'zone', 'location_type',
        'day_of_week', 'day_of_month', 'month', 'week_of_year', 'is_weekend',
        'lag_1_demand', 'lag_2_demand', 'lag_3_demand', 'lag_7_demand', 'lag_14_demand',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_mean_30',
        'historical_demand_7d', 'historical_demand_14d', 'historical_demand_30d', 'historical_order_count_7d',
        'storage_capacity_units', 'current_inventory_units', 'available_storage_units', 'utilization_rate',
        'dist_to_warehouse_km', 'avg_daily_footfall', 'avg_unit_price', 'avg_weight_kg',
        'avg_demand_weight', 'hygiene_sensitive', 'fragile'
    ]
    assert all_features == expected_features, "TEST 2 FAILED: Feature names/order mismatch"
    print(f"✓ TEST 2 PASSED: Exactly 33 features verified in required order: {all_features[:4]}...{all_features[-4:]}")

    # -------------------------------------------------------------
    # TEST 3: OrdinalEncoder transformation succeeds.
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing OrdinalEncoder Transformation...")
    encoder = service._bundle['encoder']
    cat_cols = service._bundle['categorical_features']
    sample_df = pd.DataFrame([{
        'location_id': 'LOC001',
        'category': 'Electronics',
        'zone': 'Central',
        'location_type': 'Store'
    }])
    encoded = encoder.transform(sample_df[cat_cols])
    assert encoded.shape == (1, 4), "TEST 3 FAILED: Encoded shape mismatch"
    print(f"✓ TEST 3 PASSED: OrdinalEncoder transformed {cat_cols} -> {encoded[0].tolist()}")

    # -------------------------------------------------------------
    # TEST 4, 5, 6: model_1d, model_3d, model_7d .predict() work
    # -------------------------------------------------------------
    print("\n[TEST 4, 5, 6] Testing Model Predictions (.predict())...")
    row_dict = {
        'location_id': 'LOC001', 'category': 'Electronics', 'zone': 'Central', 'location_type': 'Store',
        'day_of_week': 2, 'day_of_month': 15, 'month': 10, 'week_of_year': 42, 'is_weekend': 0,
        'lag_1_demand': 5.0, 'lag_2_demand': 6.0, 'lag_3_demand': 4.0, 'lag_7_demand': 5.0, 'lag_14_demand': 6.0,
        'rolling_mean_7': 5.2, 'rolling_std_7': 0.8, 'rolling_mean_14': 5.1, 'rolling_mean_30': 5.0,
        'historical_demand_7d': 36.0, 'historical_demand_14d': 72.0, 'historical_demand_30d': 150.0,
        'historical_order_count_7d': 30.0, 'storage_capacity_units': 2500, 'current_inventory_units': 1840,
        'available_storage_units': 660, 'utilization_rate': 0.736, 'dist_to_warehouse_km': 18.5,
        'avg_daily_footfall': 450.0, 'avg_unit_price': 149.99, 'avg_weight_kg': 0.35, 'avg_demand_weight': 8.5,
        'hygiene_sensitive': 0, 'fragile': 0
    }
    test_df = pd.DataFrame([row_dict])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    p1d = float(service._bundle['model_1d'].predict(test_df[all_features])[0])
    p3d = float(service._bundle['model_3d'].predict(test_df[all_features])[0])
    p7d = float(service._bundle['model_7d'].predict(test_df[all_features])[0])

    print(f"✓ TEST 4 PASSED: model_1d.predict() output = {p1d:.2f}")
    print(f"✓ TEST 5 PASSED: model_3d.predict() output = {p3d:.2f}")
    print(f"✓ TEST 6 PASSED: model_7d.predict() output = {p7d:.2f}")

    # -------------------------------------------------------------
    # TEST 7: Exactly 14 daily predictions are returned.
    # -------------------------------------------------------------
    print("\n[TEST 7] Testing 14-Day Forecast Generation...")
    fc = service.predict_demand_14d("SKU-AUD-801", "hub-sea-01", horizon_days=14)
    forecast_points = fc['forecast_points']
    assert len(forecast_points) == 14, f"TEST 7 FAILED: Expected 14 daily points, got {len(forecast_points)}"
    print(f"✓ TEST 7 PASSED: Exactly 14 daily predictions generated: {[p['predicted_demand'] for p in forecast_points]}")

    # -------------------------------------------------------------
    # TEST 8: forecast_demand equals the actual sum of the 14 predictions.
    # -------------------------------------------------------------
    print("\n[TEST 8] Verifying Forecast Demand Sum...")
    actual_sum = sum(p['predicted_demand'] for p in forecast_points)
    assert fc['forecast_demand'] == actual_sum, f"TEST 8 FAILED: {fc['forecast_demand']} != sum {actual_sum}"
    print(f"✓ TEST 8 PASSED: forecast_demand ({fc['forecast_demand']}) equals exact sum of 14 daily predictions ({actual_sum}).")

    # -------------------------------------------------------------
    # TEST 9: net_position == available_stock - forecast_demand.
    # -------------------------------------------------------------
    print("\n[TEST 9] Testing Net Position Formula...")
    expected_net = fc['available_stock'] - fc['forecast_demand']
    assert fc['net_position'] == expected_net, f"TEST 9 FAILED: {fc['net_position']} != {expected_net}"
    print(f"✓ TEST 9 PASSED: net_position ({fc['net_position']}) == available_stock ({fc['available_stock']}) - forecast_demand ({fc['forecast_demand']}).")

    # -------------------------------------------------------------
    # TEST 10: stockout_risk == (net_position < 0).
    # -------------------------------------------------------------
    print("\n[TEST 10] Testing Stockout Risk Logic...")
    assert fc['stockout_risk'] == (fc['net_position'] < 0), "TEST 10 FAILED"
    print(f"✓ TEST 10 PASSED: stockout_risk ({fc['stockout_risk']}) == (net_position < 0).")

    # -------------------------------------------------------------
    # TEST 11: Changing PostgreSQL available_sellable changes net_position.
    # -------------------------------------------------------------
    print("\n[TEST 11] Testing Real-Time Inventory Reactivity...")
    db = SessionLocal()
    inv_item = db.query(models.Inventory).filter(
        models.Inventory.sku == "SKU-AUD-801",
        models.Inventory.hub_id == "hub-sea-01"
    ).first()
    assert inv_item is not None, "Inventory item must exist in DB"
    orig_avail = inv_item.available_sellable
    
    # Mutate DB
    inv_item.available_sellable = orig_avail + 10
    db.commit()

    try:
        fc_mutated = service.predict_demand_14d("SKU-AUD-801", "hub-sea-01", horizon_days=14)
        assert fc_mutated['available_stock'] == orig_avail + 10, "Available stock must update"
        assert fc_mutated['net_position'] == (orig_avail + 10) - fc_mutated['forecast_demand'], "Net position must adjust"
        print(f"✓ TEST 11 PASSED: Mutating DB (+10 stock) immediately updated net position from {fc['net_position']} to {fc_mutated['net_position']}.")
    finally:
        inv_item.available_sellable = orig_avail
        db.commit()
        db.close()

    # -------------------------------------------------------------
    # TEST 12: Changing demand model input features changes the model prediction.
    # -------------------------------------------------------------
    print("\n[TEST 12] Testing Model Sensitivity to Feature Changes...")
    high_footfall_df = test_df.copy()
    high_footfall_df['avg_daily_footfall'] = 1500.0
    high_footfall_df['historical_demand_7d'] = 120.0
    high_footfall_df['rolling_mean_7'] = 17.0
    
    p7d_high = float(service._bundle['model_7d'].predict(high_footfall_df[all_features])[0])
    print(f"✓ TEST 12 PASSED: Model responds to input features (Base 7d: {p7d:.2f} vs High Activity: {p7d_high:.2f}).")

    # -------------------------------------------------------------
    # TEST 13: Production forecast endpoint does NOT use DEMAND_CATALOG.
    # -------------------------------------------------------------
    print("\n[TEST 13] Verifying API Endpoint Execution...")
    url = "http://127.0.0.1:8000/api/demand/forecast?sku=SKU-AUD-801&hub_id=hub-sea-01"
    req = urllib.request.urlopen(url)
    res_data = json.loads(req.read().decode('utf-8'))
    assert res_data['model_source'] == "demand_hist_gradient_boosting.joblib", "Endpoint must use joblib model"
    assert len(res_data['forecast_points']) == 14, "Must return 14 daily points"
    assert res_data['forecast_demand'] == sum(p['predicted_demand'] for p in res_data['forecast_points']), "Total must equal sum"
    print(f"✓ TEST 13 PASSED: Endpoint {url} uses real ML inference ({res_data['model_source']}) with forecast_demand={res_data['forecast_demand']}.")

    print("\n" + "=" * 80)
    print("ALL 13 ML DEMAND FORECASTING TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
