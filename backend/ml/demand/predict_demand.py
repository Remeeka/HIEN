import os
import json
import joblib
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'hien_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'model')

MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, 'demand_hist_gradient_boosting.joblib')
OUTPUT_PREDICTIONS_PATH = os.path.join(MODEL_DIR, 'demand_predictions.csv')

def load_data_and_model():
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_BUNDLE_PATH}. Please run train_demand_model.py first.")
    
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    orders_df = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    locations_df = pd.read_csv(os.path.join(DATA_DIR, 'locations.csv'))
    products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    
    return bundle, orders_df, locations_df, products_df

def generate_predictions():
    print("Generating comprehensive demand predictions for all locations and categories...")
    bundle, orders_df, locations_df, products_df = load_data_and_model()
    
    model_7d = bundle['model_7d']
    model_3d = bundle['model_3d']
    model_1d = bundle['model_1d']
    encoder = bundle['encoder']
    categorical_features = bundle['categorical_features']
    all_features = bundle['all_features']

    # Standardize orders
    orders_df['order_date'] = pd.to_datetime(orders_df['order_timestamp']).dt.normalize()
    orders_df['location_id'] = orders_df['nearest_location_id']
    orders_df['quantity'] = 1
    orders_df['total_price'] = orders_df['unit_price']

    # Aggregate daily demand per (date, location_id, category)
    daily_orders = orders_df.groupby(['order_date', 'location_id', 'category']).agg(
        daily_quantity=('quantity', 'sum'),
        order_count=('order_id', 'count'),
        total_sales=('total_price', 'sum')
    ).reset_index()

    unique_locations = sorted(orders_df['location_id'].unique())
    unique_categories = sorted(orders_df['category'].unique())
    all_dates = pd.date_range(start=orders_df['order_date'].min(), end=orders_df['order_date'].max(), freq='D')

    grid_idx = pd.MultiIndex.from_product([all_dates, unique_locations, unique_categories], names=['order_date', 'location_id', 'category'])
    grid_df = pd.DataFrame(index=grid_idx).reset_index()

    merged = pd.merge(grid_df, daily_orders, on=['order_date', 'location_id', 'category'], how='left')
    merged['daily_quantity'] = merged['daily_quantity'].fillna(0.0)
    merged['order_count'] = merged['order_count'].fillna(0.0)
    merged['total_sales'] = merged['total_sales'].fillna(0.0)
    merged = merged.sort_values(by=['location_id', 'category', 'order_date']).reset_index(drop=True)

    grouped = merged.groupby(['location_id', 'category'])

    # Historical Lags & Rolling
    merged['lag_1_demand'] = grouped['daily_quantity'].shift(1)
    merged['lag_2_demand'] = grouped['daily_quantity'].shift(2)
    merged['lag_3_demand'] = grouped['daily_quantity'].shift(3)
    merged['lag_7_demand'] = grouped['daily_quantity'].shift(7)
    merged['lag_14_demand'] = grouped['daily_quantity'].shift(14)

    shifted = grouped['daily_quantity'].shift(1)
    merged['rolling_mean_7'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).mean().reset_index(drop=True)
    merged['rolling_std_7'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).std().fillna(0.0).reset_index(drop=True)
    merged['rolling_mean_14'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(14, min_periods=1).mean().reset_index(drop=True)
    merged['rolling_mean_30'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(30, min_periods=1).mean().reset_index(drop=True)

    merged['historical_demand_7d'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).sum().reset_index(drop=True)
    merged['historical_demand_14d'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(14, min_periods=1).sum().reset_index(drop=True)
    merged['historical_demand_30d'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(30, min_periods=1).sum().reset_index(drop=True)

    shifted_orders = grouped['order_count'].shift(1)
    merged['historical_order_count_7d'] = shifted_orders.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).sum().reset_index(drop=True)

    # Future date for prediction (the latest date in dataset or tomorrow)
    latest_date = merged['order_date'].max()
    next_date = latest_date + pd.Timedelta(days=1)
    
    # Static metadata
    loc_meta = locations_df[['location_id', 'location_name', 'zone', 'location_type', 'storage_capacity_units', 'current_inventory_units', 'available_storage_units', 'dist_to_warehouse_km', 'avg_daily_footfall']].copy()
    loc_meta['utilization_rate'] = loc_meta['current_inventory_units'] / (loc_meta['storage_capacity_units'] + 1e-5)

    cat_meta = products_df.groupby('category').agg(
        avg_unit_price=('unit_price', 'mean'),
        avg_weight_kg=('weight_kg', 'mean'),
        avg_demand_weight=('demand_weight', 'mean'),
        hygiene_sensitive=('hygiene_sensitive', 'max'),
        fragile=('fragile', 'max')
    ).reset_index()

    merged = pd.merge(merged, loc_meta, on='location_id', how='left')
    merged = pd.merge(merged, cat_meta, on='category', how='left')

    # Filter to the latest known state per (location_id, category) for forecasting
    latest_snapshot = merged[merged['order_date'] == latest_date].copy().reset_index(drop=True)

    # Set temporal features for the upcoming forecast period
    latest_snapshot['day_of_week'] = next_date.dayofweek
    latest_snapshot['day_of_month'] = next_date.day
    latest_snapshot['month'] = next_date.month
    latest_snapshot['week_of_year'] = int(next_date.isocalendar().week)
    latest_snapshot['is_weekend'] = int(latest_snapshot['day_of_week'].iloc[0] in [5, 6])

    # Transform features for model
    feature_matrix = latest_snapshot[all_features].copy()
    feature_matrix[categorical_features] = encoder.transform(feature_matrix[categorical_features])

    # Run predictions
    pred_7d = np.maximum(0.0, model_7d.predict(feature_matrix))
    pred_3d = np.maximum(0.0, model_3d.predict(feature_matrix))
    pred_1d = np.maximum(0.0, model_1d.predict(feature_matrix))

    latest_snapshot['predicted_demand_7d'] = np.round(pred_7d, 2)
    latest_snapshot['predicted_demand_3d'] = np.round(pred_3d, 2)
    latest_snapshot['predicted_demand_1d'] = np.round(pred_1d, 2)

    # Demand Pressure Score (0-100 normalized)
    # Demand Score = min(100, max(0, (predicted_7d / (available_storage_units * 0.15 + 1.0)) * 50.0))
    def calc_score(row):
        cap_factor = max(1.0, row['available_storage_units'] * 0.15)
        score = (row['predicted_demand_7d'] / cap_factor) * 50.0
        return float(min(100.0, max(5.0, round(score, 1))))

    latest_snapshot['demand_score'] = latest_snapshot.apply(calc_score, axis=1)

    # Demand status tier
    def get_tier(score):
        if score >= 75.0:
            return "Critical High Demand"
        elif score >= 50.0:
            return "High Demand"
        elif score >= 25.0:
            return "Moderate Demand"
        else:
            return "Low Demand"

    latest_snapshot['demand_tier'] = latest_snapshot['demand_score'].apply(get_tier)
    latest_snapshot['forecast_as_of_date'] = latest_date.strftime('%Y-%m-%d')
    latest_snapshot['forecast_target_date'] = (next_date + pd.Timedelta(days=6)).strftime('%Y-%m-%d')

    output_cols = [
        'location_id', 'location_name', 'zone', 'category',
        'forecast_as_of_date', 'forecast_target_date',
        'historical_demand_7d', 'historical_demand_14d', 'historical_demand_30d',
        'rolling_mean_7', 'predicted_demand_1d', 'predicted_demand_3d', 'predicted_demand_7d',
        'storage_capacity_units', 'available_storage_units', 'current_inventory_units',
        'demand_score', 'demand_tier'
    ]

    result_df = latest_snapshot[output_cols].sort_values(by='demand_score', ascending=False)
    result_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    print(f"Generated {len(result_df)} forecasts successfully -> {OUTPUT_PREDICTIONS_PATH}")
    return result_df

if __name__ == '__main__':
    generate_predictions()
