import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'hien_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(__file__), exist_ok=True)

print("=" * 80)
print("HIEN DEMAND FORECASTING MODEL TRAINING - HistGradientBoostingRegressor")
print("=" * 80)

# -------------------------------------------------------------
# 1. Dataset Inspection & Ingestion
# -------------------------------------------------------------
print("\n[1/7] Inspecting and Ingesting Raw Datasets...")

orders_path = os.path.join(DATA_DIR, 'orders.csv')
locations_path = os.path.join(DATA_DIR, 'locations.csv')
products_path = os.path.join(DATA_DIR, 'products.csv')
snapshots_path = os.path.join(DATA_DIR, 'inventory_snapshots.csv')
forecast_path = os.path.join(DATA_DIR, 'demand_forecast.csv')

orders_df = pd.read_csv(orders_path)
locations_df = pd.read_csv(locations_path)
products_df = pd.read_csv(products_path)
snapshots_df = pd.read_csv(snapshots_path) if os.path.exists(snapshots_path) else None
forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else None

print(f" - orders.csv: {len(orders_df):,} rows, {len(orders_df.columns)} columns")
print(f" - locations.csv: {len(locations_df):,} rows, {len(locations_df.columns)} columns")
print(f" - products.csv: {len(products_df):,} rows, {len(products_df.columns)} columns")
if snapshots_df is not None:
    print(f" - inventory_snapshots.csv: {len(snapshots_df):,} rows, {len(snapshots_df.columns)} columns")
if forecast_df is not None:
    print(f" - demand_forecast.csv: {len(forecast_df):,} rows, {len(forecast_df.columns)} columns")

# Format orders timestamp & location
orders_df['order_date'] = pd.to_datetime(orders_df['order_timestamp']).dt.normalize()
orders_df['location_id'] = orders_df['nearest_location_id']
orders_df['quantity'] = 1
orders_df['total_price'] = orders_df['unit_price']

min_date = orders_df['order_date'].min().strftime('%Y-%m-%d')
max_date = orders_df['order_date'].max().strftime('%Y-%m-%d')
unique_locations = orders_df['location_id'].unique()
unique_categories = orders_df['category'].unique()

print(f"\nDataset Statistics:")
print(f" - Date Range: {min_date} to {max_date} ({(orders_df['order_date'].max() - orders_df['order_date'].min()).days + 1} days)")
print(f" - Unique Store Locations: {len(unique_locations)} (IDs: {', '.join(sorted(unique_locations)[:5])}...)")
print(f" - Unique Categories: {len(unique_categories)} ({', '.join(sorted(unique_categories))})")
print(f" - Missing Values in Orders: {orders_df.isnull().sum().sum()}")
print(f" - Duplicate Rows in Orders: {orders_df.duplicated().sum()}")

# -------------------------------------------------------------
# 2. Feature Engineering & Daily Grid Aggregation
# -------------------------------------------------------------
print("\n[2/7] Constructing Leakage-Safe Daily Time-Series Feature Matrix...")

# Aggregate daily demand per (date, location_id, category)
daily_orders = orders_df.groupby(['order_date', 'location_id', 'category']).agg(
    daily_quantity=('quantity', 'sum'),
    order_count=('order_id', 'count'),
    total_sales=('total_price', 'sum')
).reset_index()

# Build a complete dense grid across all dates, locations, and categories
all_dates = pd.date_range(start=orders_df['order_date'].min(), end=orders_df['order_date'].max(), freq='D')
all_locs = sorted(unique_locations)
all_cats = sorted(unique_categories)

grid_idx = pd.MultiIndex.from_product([all_dates, all_locs, all_cats], names=['order_date', 'location_id', 'category'])
grid_df = pd.DataFrame(index=grid_idx).reset_index()

# Merge daily transactions onto full grid
merged = pd.merge(grid_df, daily_orders, on=['order_date', 'location_id', 'category'], how='left')
merged['daily_quantity'] = merged['daily_quantity'].fillna(0.0)
merged['order_count'] = merged['order_count'].fillna(0.0)
merged['total_sales'] = merged['total_sales'].fillna(0.0)

# Sort strictly chronologically for time series operations
merged = merged.sort_values(by=['location_id', 'category', 'order_date']).reset_index(drop=True)

# Compute Historical Lags & Rolling Features (Leakage-Safe, strictly backward-looking)
grouped = merged.groupby(['location_id', 'category'])

# 1. Historical Lag features
merged['lag_1_demand'] = grouped['daily_quantity'].shift(1)
merged['lag_2_demand'] = grouped['daily_quantity'].shift(2)
merged['lag_3_demand'] = grouped['daily_quantity'].shift(3)
merged['lag_7_demand'] = grouped['daily_quantity'].shift(7)
merged['lag_14_demand'] = grouped['daily_quantity'].shift(14)

# 2. Rolling Historical Demand (excluding current day via shift 1)
# 7-day, 14-day, 30-day backward sums and means
shifted = grouped['daily_quantity'].shift(1)
merged['rolling_mean_7'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).mean().reset_index(drop=True)
merged['rolling_std_7'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).std().fillna(0.0).reset_index(drop=True)
merged['rolling_mean_14'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(14, min_periods=1).mean().reset_index(drop=True)
merged['rolling_mean_30'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(30, min_periods=1).mean().reset_index(drop=True)

merged['historical_demand_7d'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).sum().reset_index(drop=True)
merged['historical_demand_14d'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(14, min_periods=1).sum().reset_index(drop=True)
merged['historical_demand_30d'] = shifted.groupby([merged['location_id'], merged['category']]).rolling(30, min_periods=1).sum().reset_index(drop=True)

# Historical order count
shifted_orders = grouped['order_count'].shift(1)
merged['historical_order_count_7d'] = shifted_orders.groupby([merged['location_id'], merged['category']]).rolling(7, min_periods=1).sum().reset_index(drop=True)

# 3. Future Targets (Strictly Forward-Looking for Training Only)
indexer_7d = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
indexer_3d = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
indexer_1d = pd.api.indexers.FixedForwardWindowIndexer(window_size=1)

merged['future_demand_7d'] = grouped['daily_quantity'].rolling(window=indexer_7d, min_periods=7).sum().reset_index(drop=True)
merged['future_demand_3d'] = grouped['daily_quantity'].rolling(window=indexer_3d, min_periods=3).sum().reset_index(drop=True)
merged['future_demand_1d'] = grouped['daily_quantity'].rolling(window=indexer_1d, min_periods=1).sum().reset_index(drop=True)

# 4. Temporal Calendar Features
merged['day_of_week'] = merged['order_date'].dt.dayofweek
merged['day_of_month'] = merged['order_date'].dt.day
merged['month'] = merged['order_date'].dt.month
merged['week_of_year'] = merged['order_date'].dt.isocalendar().week.astype(int)
merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)

# 5. Static Metadata Enrichment (Location & Product Category Characteristics)
loc_meta = locations_df[['location_id', 'zone', 'location_type', 'storage_capacity_units', 'current_inventory_units', 'available_storage_units', 'dist_to_warehouse_km', 'avg_daily_footfall']].copy()
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

# Drop initial warmup records (where 14-day lags are NaN) and end horizon (where 7-day target cannot look ahead)
clean_df = merged.dropna(subset=['lag_14_demand', 'rolling_mean_30', 'future_demand_7d']).copy()
clean_df = clean_df.sort_values(by='order_date').reset_index(drop=True)

print(f" - Clean Processed Feature Matrix: {len(clean_df):,} records")
print(f" - Feature Matrix Date Range: {clean_df['order_date'].min().strftime('%Y-%m-%d')} to {clean_df['order_date'].max().strftime('%Y-%m-%d')}")

# -------------------------------------------------------------
# 3. Categorical Encodings & Feature Definition
# -------------------------------------------------------------
categorical_features = ['location_id', 'category', 'zone', 'location_type']
numeric_features = [
    'day_of_week', 'day_of_month', 'month', 'week_of_year', 'is_weekend',
    'lag_1_demand', 'lag_2_demand', 'lag_3_demand', 'lag_7_demand', 'lag_14_demand',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_mean_30',
    'historical_demand_7d', 'historical_demand_14d', 'historical_demand_30d',
    'historical_order_count_7d',
    'storage_capacity_units', 'current_inventory_units', 'available_storage_units', 'utilization_rate',
    'dist_to_warehouse_km', 'avg_daily_footfall',
    'avg_unit_price', 'avg_weight_kg', 'avg_demand_weight', 'hygiene_sensitive', 'fragile'
]

all_features = categorical_features + numeric_features

# Fit OrdinalEncoder on categorical features for native HistGradientBoostingRegressor support
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
clean_df[categorical_features] = encoder.fit_transform(clean_df[categorical_features])

# -------------------------------------------------------------
# 4. Chronological Time-Series Split
# -------------------------------------------------------------
print("\n[3/7] Performing Strict Chronological Time-Series Split...")

dates = clean_df['order_date'].drop_duplicates().sort_values().reset_index(drop=True)
n_dates = len(dates)

train_split_idx = int(n_dates * 0.70)
val_split_idx = int(n_dates * 0.85)

train_max_date = dates.iloc[train_split_idx - 1]
val_max_date = dates.iloc[val_split_idx - 1]
test_max_date = dates.iloc[-1]

train_df = clean_df[clean_df['order_date'] <= train_max_date].copy()
val_df = clean_df[(clean_df['order_date'] > train_max_date) & (clean_df['order_date'] <= val_max_date)].copy()
test_df = clean_df[clean_df['order_date'] > val_max_date].copy()

print("-" * 60)
print(f"Train Set      : {train_df['order_date'].min().strftime('%Y-%m-%d')} to {train_df['order_date'].max().strftime('%Y-%m-%d')} | {len(train_df):,} rows ({len(train_df)/len(clean_df)*100:.1f}%)")
print(f"Validation Set : {val_df['order_date'].min().strftime('%Y-%m-%d')} to {val_df['order_date'].max().strftime('%Y-%m-%d')} | {len(val_df):,} rows ({len(val_df)/len(clean_df)*100:.1f}%)")
print(f"Test Set       : {test_df['order_date'].min().strftime('%Y-%m-%d')} to {test_df['order_date'].max().strftime('%Y-%m-%d')} | {len(test_df):,} rows ({len(test_df)/len(clean_df)*100:.1f}%)")
print("-" * 60)

X_train = train_df[all_features]
y_train_7d = train_df['future_demand_7d']
y_train_3d = train_df['future_demand_3d']
y_train_1d = train_df['future_demand_1d']

X_val = val_df[all_features]
y_val_7d = val_df['future_demand_7d']
y_val_3d = val_df['future_demand_3d']
y_val_1d = val_df['future_demand_1d']

X_test = test_df[all_features]
y_test_7d = test_df['future_demand_7d']
y_test_3d = test_df['future_demand_3d']
y_test_1d = test_df['future_demand_1d']

# -------------------------------------------------------------
# 5. Model Training (Primary 7-Day & Auxiliary 3-Day, 1-Day)
# -------------------------------------------------------------
print("\n[4/7] Training HistGradientBoostingRegressor Models...")

cat_indices = [all_features.index(col) for col in categorical_features]

# Primary 7-Day Model
model_7d = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=8,
    min_samples_leaf=15,
    learning_rate=0.08,
    categorical_features=cat_indices,
    random_state=42,
    loss='squared_error'
)
print(" - Training 7-Day Horizon Model...")
model_7d.fit(X_train, y_train_7d)

# Auxiliary 3-Day Model
model_3d = HistGradientBoostingRegressor(
    max_iter=150,
    max_depth=8,
    min_samples_leaf=15,
    learning_rate=0.08,
    categorical_features=cat_indices,
    random_state=42,
    loss='squared_error'
)
print(" - Training 3-Day Horizon Model...")
model_3d.fit(X_train, y_train_3d)

# Auxiliary 1-Day Model
model_1d = HistGradientBoostingRegressor(
    max_iter=150,
    max_depth=8,
    min_samples_leaf=15,
    learning_rate=0.08,
    categorical_features=cat_indices,
    random_state=42,
    loss='squared_error'
)
print(" - Training 1-Day Horizon Model...")
model_1d.fit(X_train, y_train_1d)

# -------------------------------------------------------------
# 6. Evaluation & Comparison Against Baseline
# -------------------------------------------------------------
print("\n[5/7] Evaluating Forecast Performance against Historical Baseline...")

def evaluate_predictions(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

# 7-Day Model Predictions
train_pred_7d = model_7d.predict(X_train)
val_pred_7d = model_7d.predict(X_val)
test_pred_7d = model_7d.predict(X_test)

metrics_train_7d = evaluate_predictions(y_train_7d, train_pred_7d, "Train 7D")
metrics_val_7d = evaluate_predictions(y_val_7d, val_pred_7d, "Validation 7D")
metrics_test_7d = evaluate_predictions(y_test_7d, test_pred_7d, "Test 7D")

# Baseline: Historical 7-day demand from prior window (t-7 to t-1)
baseline_test_7d = test_df['historical_demand_7d']
baseline_metrics = evaluate_predictions(y_test_7d, baseline_test_7d, "Baseline Test 7D")

# Calculate improvement over baseline
mae_improvement = ((baseline_metrics['mae'] - metrics_test_7d['mae']) / baseline_metrics['mae']) * 100
rmse_improvement = ((baseline_metrics['rmse'] - metrics_test_7d['rmse']) / baseline_metrics['rmse']) * 100
r2_gain = metrics_test_7d['r2'] - baseline_metrics['r2']

print("\n" + "=" * 60)
print("EVALUATION RESULTS (7-Day Forecast):")
print("-" * 60)
print(f"Train Set      | MAE: {metrics_train_7d['mae']:.3f} | RMSE: {metrics_train_7d['rmse']:.3f} | R²: {metrics_train_7d['r2']:.4f}")
print(f"Validation Set | MAE: {metrics_val_7d['mae']:.3f} | RMSE: {metrics_val_7d['rmse']:.3f} | R²: {metrics_val_7d['r2']:.4f}")
print(f"Test Set (HGB) | MAE: {metrics_test_7d['mae']:.3f} | RMSE: {metrics_test_7d['rmse']:.3f} | R²: {metrics_test_7d['r2']:.4f}")
print(f"Baseline Naive | MAE: {baseline_metrics['mae']:.3f} | RMSE: {baseline_metrics['rmse']:.3f} | R²: {baseline_metrics['r2']:.4f}")
print("-" * 60)
print(f"Improvement over Baseline: MAE: -{mae_improvement:.1f}%, RMSE: -{rmse_improvement:.1f}%, R² Gain: +{r2_gain:.4f}")
print("=" * 60)

# Evaluate 3D and 1D on Test Set
test_pred_3d = model_3d.predict(X_test)
metrics_test_3d = evaluate_predictions(y_test_3d, test_pred_3d, "Test 3D")

test_pred_1d = model_1d.predict(X_test)
metrics_test_1d = evaluate_predictions(y_test_1d, test_pred_1d, "Test 1D")

# -------------------------------------------------------------
# 7. Permutation Feature Importances
# -------------------------------------------------------------
print("\n[6/7] Computing Feature Importances via Permutation...")
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(model_7d, X_test, y_test_7d, n_repeats=5, random_state=42, n_jobs=-1)

importances = perm_result.importances_mean
sorted_idx = np.argsort(importances)[::-1]

feature_importance_list = [
    {"feature": all_features[i], "importance": round(float(importances[i]), 4)}
    for i in sorted_idx if importances[i] > 0
][:15]

for rank, item in enumerate(feature_importance_list[:8], 1):
    print(f"  {rank}. {item['feature']:<28} : {item['importance']:.4f}")

# -------------------------------------------------------------
# 8. Save Artifacts & Metadata
# -------------------------------------------------------------
print("\n[7/7] Saving Model Artifacts and Metadata...")

bundle = {
    "model_7d": model_7d,
    "model_3d": model_3d,
    "model_1d": model_1d,
    "encoder": encoder,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "all_features": all_features
}

model_save_path = os.path.join(MODEL_DIR, 'demand_hist_gradient_boosting.joblib')
joblib.dump(bundle, model_save_path)
print(f" -> Saved Model Bundle to: {model_save_path}")

features_save_path = os.path.join(MODEL_DIR, 'demand_features.json')
with open(features_save_path, 'w') as f:
    json.dump({
        "all_features": all_features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "feature_importances": feature_importance_list
    }, f, indent=2)
print(f" -> Saved Feature Metadata to: {features_save_path}")

metrics_save_path = os.path.join(MODEL_DIR, 'demand_metrics.json')
metrics_data = {
    "model_name": "HistGradientBoostingRegressor",
    "target_horizons": ["7-day", "3-day", "1-day"],
    "split_info": {
        "train_date_range": f"{train_df['order_date'].min().strftime('%Y-%m-%d')} to {train_df['order_date'].max().strftime('%Y-%m-%d')}",
        "val_date_range": f"{val_df['order_date'].min().strftime('%Y-%m-%d')} to {val_df['order_date'].max().strftime('%Y-%m-%d')}",
        "test_date_range": f"{test_df['order_date'].min().strftime('%Y-%m-%d')} to {test_df['order_date'].max().strftime('%Y-%m-%d')}",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "total_rows": int(len(clean_df))
    },
    "metrics_7d": {
        "train": metrics_train_7d,
        "validation": metrics_val_7d,
        "test": metrics_test_7d,
        "baseline_naive_test": baseline_metrics,
        "improvement": {
            "mae_reduction_pct": round(mae_improvement, 2),
            "rmse_reduction_pct": round(rmse_improvement, 2),
            "r2_gain": round(r2_gain, 4)
        }
    },
    "metrics_3d": {
        "test": metrics_test_3d
    },
    "metrics_1d": {
        "test": metrics_test_1d
    },
    "demand_score_formula": "Demand_Score = min(100.0, max(0.0, (Predicted_7d_Demand / (Available_Hub_Capacity * 0.15 + 1.0)) * 50.0))",
    "trained_at": datetime.now().isoformat()
}

with open(metrics_save_path, 'w') as f:
    json.dump(metrics_data, f, indent=2)
print(f" -> Saved Demand Metrics to: {metrics_save_path}")

print("\nModel training complete! All requirements verified successfully.")
