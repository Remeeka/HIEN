import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Load existing datasets
input_dir = 'hien_synthetic_dataset'
R_orig = pd.read_csv(os.path.join(input_dir, 'returns.csv'))
O_orig = pd.read_csv(os.path.join(input_dir, 'orders.csv'))
P_orig = pd.read_csv(os.path.join(input_dir, 'products.csv'))
L_orig = pd.read_csv(os.path.join(input_dir, 'locations.csv'))

out_dir = 'hien_dataset'
os.makedirs(out_dir, exist_ok=True)
ml_dir = os.path.join(out_dir, 'ml')
os.makedirs(ml_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. ENHANCE PRODUCTS (products.csv)
# -------------------------------------------------------------
print("Enriching products.csv...")
products_df = P_orig.copy()
if 'product_id' not in products_df.columns:
    products_df.insert(0, 'product_id', products_df['sku'])
products_df.to_csv(os.path.join(out_dir, 'products.csv'), index=False)
P_dict = products_df.set_index('product_id').to_dict('index')

# -------------------------------------------------------------
# 2. ENHANCE LOCATIONS (locations.csv)
# -------------------------------------------------------------
print("Enriching locations.csv...")
loc_names_store = [
    "Indiranagar Hub Store", "Koramangala Smart Hub", "Whitefield Retail Store", "Jayanagar HIEN Store",
    "HSR Layout Express Hub", "Electronic City Store", "Malleshwaram Store", "MG Road Experience Center",
    "Hebbal North Hub", "Marathahalli Retail Hub", "BTM Layout Store", "Banashankari Smart Store",
    "Rajajinagar Store", "Sarjapur Road Hub", "Bellandur Retail Point", "Yelahanka North Store",
    "Kalyan Nagar Hub", "Basavanagudi Store", "JP Nagar Smart Store", "RT Nagar Hub",
    "Vijayanagar Store", "Kengeri Satellite Hub", "Hoodi Express Store", "Bannerghatta Hub", "Varthur Retail Point"
]
loc_names_ap = [
    "Kadugodi Access Point", "Domlur Pickup Kiosk", "Ulsoor Micro Locker", "Brookefield Access Point",
    "Mahadevapura Drop Kiosk", "Thubarahalli Micro Point", "CV Raman Nagar Point", "Frazer Town Locker",
    "Adugodi Access Point", "Kaggadasapura Kiosk", "Kudlu Gate Micro Hub", "Singasandra Access Point",
    "Nagawara Kiosk", "Peenya Industrial Access Point", "Yeshwanthpur Micro Point"
]

center_lat = 12.9716
center_lon = 77.5946

loc_rows = []
for idx, r in L_orig.iterrows():
    lid = r['location_id']
    ltype = r['location_type']
    if ltype == 'Store':
        name = loc_names_store[int(lid.replace('LOC','')) - 1]
    else:
        name = loc_names_ap[int(lid.replace('LOC','')) - 26]
    
    x = float(r['x_km'])
    y = float(r['y_km'])
    lat = round(center_lat + (y / 110.9), 6)
    lon = round(center_lon + (x / 108.2), 6)
    cap = int(r['storage_capacity_units'])
    curr_inv = int(cap * rng.uniform(0.25, 0.65))
    avail_cap = cap - curr_inv
    
    loc_rows.append({
        'location_id': lid,
        'location_name': f"HIEN {name}",
        'city': 'Bengaluru',
        'state': 'Karnataka',
        'latitude': lat,
        'longitude': lon,
        'x_km': x,
        'y_km': y,
        'zone': r['zone'],
        'location_type': ltype,
        'storage_capacity_units': cap,
        'current_inventory_units': curr_inv,
        'available_storage_units': avail_cap,
        'verification_slots_per_day': int(r['verification_slots_per_day']),
        'operating_hours': '08:00 - 22:00' if ltype == 'Store' else '09:00 - 21:00',
        'delivery_radius_km': 10.0 if ltype == 'Store' else 6.0,
        'hub_enabled': bool(r['hub_enabled']),
        'local_delivery_cost_per_km': 4.50 if ltype == 'Store' else 5.20,
        'warehouse_handling_cost_inr': 45.0,
        'avg_daily_footfall': int(r['avg_daily_footfall']),
        'dist_to_warehouse_km': float(r['dist_to_warehouse_km'])
    })

# Add Central Warehouse WH001
loc_rows.append({
    'location_id': 'WH001',
    'location_name': 'HIEN Central Distribution Warehouse - Nelamangala',
    'city': 'Bengaluru',
    'state': 'Karnataka',
    'latitude': center_lat,
    'longitude': center_lon,
    'x_km': 0.0,
    'y_km': 0.0,
    'zone': 'Central',
    'location_type': 'Central Warehouse',
    'storage_capacity_units': 50000,
    'current_inventory_units': 24500,
    'available_storage_units': 25500,
    'verification_slots_per_day': 2500,
    'operating_hours': '24/7',
    'delivery_radius_km': 60.0,
    'hub_enabled': True,
    'local_delivery_cost_per_km': 3.50,
    'warehouse_handling_cost_inr': 85.0,
    'avg_daily_footfall': 200,
    'dist_to_warehouse_km': 0.0
})

locations_df = pd.DataFrame(loc_rows)
locations_df.to_csv(os.path.join(out_dir, 'locations.csv'), index=False)
L_dict = locations_df.set_index('location_id').to_dict('index')

# -------------------------------------------------------------
# 3. GENERATE CUSTOMERS (customers.csv) AND UPDATE ORDERS (orders.csv)
# -------------------------------------------------------------
print("Generating customers.csv and assigning customer_id to orders.csv...")
n_customers = 10000
cust_ids = [f"CUS{i:06d}" for i in range(1, n_customers + 1)]

seg_choices = ['Frequent', 'Regular', 'Occasional', 'New']
seg_weights = [0.10, 0.30, 0.35, 0.25]
cust_segs = rng.choice(seg_choices, size=n_customers, p=seg_weights)

loc_pool = [r['location_id'] for r in loc_rows if r['location_id'] != 'WH001']
loc_footfalls = [r['avg_daily_footfall'] for r in loc_rows if r['location_id'] != 'WH001']
loc_foot_sum = sum(loc_footfalls)
loc_p = [f / loc_foot_sum for f in loc_footfalls]

cust_home_locs = rng.choice(loc_pool, size=n_customers, p=loc_p)
cust_radius = rng.choice([5.0, 7.5, 10.0, 12.0], size=n_customers, p=[0.3, 0.4, 0.2, 0.1])

cust_x = []
cust_y = []
cust_lat = []
cust_lon = []
for i in range(n_customers):
    hl = cust_home_locs[i]
    hx = L_dict[hl]['x_km']
    hy = L_dict[hl]['y_km']
    cx = round(float(hx + rng.normal(0, 2.0)), 2)
    cy = round(float(hy + rng.normal(0, 2.0)), 2)
    cust_x.append(cx)
    cust_y.append(cy)
    cust_lat.append(round(center_lat + (cy / 110.9), 6))
    cust_lon.append(round(center_lon + (cx / 108.2), 6))

customers_df = pd.DataFrame({
    'customer_id': cust_ids,
    'customer_latitude': cust_lat,
    'customer_longitude': cust_lon,
    'customer_x_km': cust_x,
    'customer_y_km': cust_y,
    'home_location_id': cust_home_locs,
    'preferred_delivery_radius_km': cust_radius,
    'customer_segment': cust_segs
})
customers_df.to_csv(os.path.join(out_dir, 'customers.csv'), index=False)

cust_order_weights = np.where(cust_segs == 'Frequent', 6.0,
                     np.where(cust_segs == 'Regular', 3.0,
                     np.where(cust_segs == 'Occasional', 1.5, 0.8)))

cust_by_loc = {}
for loc_id in loc_pool:
    mask = (customers_df['home_location_id'] == loc_id).values
    c_ids = customers_df['customer_id'].values[mask]
    c_w = cust_order_weights[mask]
    cust_by_loc[loc_id] = {
        'ids': c_ids,
        'weights': c_w / c_w.sum()
    }

orders_df = O_orig.copy()
if 'product_id' not in orders_df.columns:
    orders_df.insert(2, 'product_id', orders_df['sku'])

assigned_custs = []
for idx, r in orders_df.iterrows():
    near_loc = r['nearest_location_id']
    pool_info = cust_by_loc.get(near_loc)
    if pool_info is not None and len(pool_info['ids']) > 0:
        cid = rng.choice(pool_info['ids'], p=pool_info['weights'])
    else:
        cid = rng.choice(cust_ids)
    assigned_custs.append(cid)

orders_df.insert(1, 'customer_id', assigned_custs)
orders_df.to_csv(os.path.join(out_dir, 'orders.csv'), index=False)
print(f"Orders enriched with customer IDs. Total unique customers in orders: {orders_df['customer_id'].nunique()}")

# -------------------------------------------------------------
# 4. ENHANCE RETURNS (returns.csv)
# -------------------------------------------------------------
print("Enriching returns.csv with eligibility_reason and product_id...")
returns_df = R_orig.copy()
if 'product_id' not in returns_df.columns:
    returns_df.insert(2, 'product_id', returns_df['sku'])

hub_enabled_map = {r['location_id']: r['hub_enabled'] for r in loc_rows}

eligibility_reasons = []
for idx, r in returns_df.iterrows():
    h_elig = r['hien_eligible']
    final_st = r['final_status']
    gi = r['ai_condition_grade']
    fraud = r['ai_fraud_risk_score']
    is_fraud = r['is_fraud_swap']
    hyg = r['hygiene_sensitive']
    conf = r['ai_confidence']
    reason = r['return_reason']
    loc_enabled = hub_enabled_map.get(r['return_location_id'], True)
    dem_sc = r['local_demand_score']
    has_pack = r['has_original_packaging']
    cat = r['category']
    
    if is_fraud or fraud >= 0.35:
        el_reason = "Fraud risk"
    elif hyg == 1:
        el_reason = "Hygiene restriction"
    elif not has_pack and cat in ['Electronics', 'Home & Kitchen'] and not h_elig:
        el_reason = "Missing accessories"
    elif not loc_enabled or conf < 0.75:
        el_reason = "Insufficient verification"
    elif gi == 'Damaged':
        el_reason = "Severe damage"
    elif gi == 'Minor Wear':
        el_reason = "Minor cosmetic issue"
    elif h_elig:
        if final_st == 'Expired - Sent to Warehouse':
            el_reason = "No suitable local demand"
        elif dem_sc >= 0.80:
            el_reason = "High local demand"
        elif gi == 'New/Sealed':
            el_reason = "Excellent condition"
        else:
            el_reason = "Functional and resalable"
    else:
        el_reason = "Failed functional test"
    
    eligibility_reasons.append(el_reason)

returns_df['eligibility_reason'] = eligibility_reasons
returns_df.to_csv(os.path.join(out_dir, 'returns.csv'), index=False)

# -------------------------------------------------------------
# 5. CREATE AI TRAINING DATA & ML SPLITS (ai_training_data.csv, ml/train.csv, etc.)
# -------------------------------------------------------------
print("Creating ai_training_data.csv and ML splits...")
nR = len(returns_df)
acc_complete = []
func_tests = []
prev_returns = []
hub_dists = []
avail_caps = []

for idx, r in returns_df.iterrows():
    reason = r['return_reason']
    has_pack = r['has_original_packaging']
    grade = r['ai_condition_grade']
    
    p_acc = 0.95 if has_pack else 0.70
    if reason in ['Wrong item', 'Defective']:
        p_acc -= 0.15
    acc = 1 if rng.random() < p_acc else 0
    acc_complete.append(acc)
    
    if grade in ['New/Sealed', 'Like New']:
        p_pass = 0.94 if reason not in ['Defective', 'Damaged in transit'] else 0.65
        ftest = 'Pass' if rng.random() < p_pass else ('Minor Flaw' if rng.random() < 0.7 else 'Fail')
    elif grade == 'Minor Wear':
        ftest = 'Pass' if rng.random() < 0.60 else ('Minor Flaw' if rng.random() < 0.75 else 'Fail')
    else:
        ftest = 'Fail' if rng.random() < 0.85 else 'Minor Flaw'
    func_tests.append(ftest)
    
    p_ret = rng.choice([0, 1, 2, 3, 4], p=[0.60, 0.25, 0.10, 0.04, 0.01])
    prev_returns.append(p_ret)
    hub_dists.append(round(float(rng.uniform(0.5, 8.5)), 2))
    
    loc_id = r['return_location_id']
    avail_caps.append(L_dict[loc_id]['available_storage_units'])

ai_data = pd.DataFrame({
    'return_id': returns_df['return_id'],
    'product_id': returns_df['product_id'],
    'product_category': returns_df['category'],
    'unit_price': returns_df['unit_price'],
    'return_reason': returns_df['return_reason'],
    'days_since_purchase': returns_df['days_since_purchase'],
    'ai_condition_grade': returns_df['ai_condition_grade'],
    'ai_confidence': returns_df['ai_confidence'],
    'ai_fraud_risk_score': returns_df['ai_fraud_risk_score'],
    'has_original_packaging': returns_df['has_original_packaging'].astype(int),
    'accessories_complete': acc_complete,
    'functional_test_result': func_tests,
    'hygiene_sensitive': returns_df['hygiene_sensitive'],
    'fragile': returns_df['fragile'],
    'local_demand_score': returns_df['local_demand_score'],
    'previous_return_count': prev_returns,
    'distance_to_nearest_hub_km': hub_dists,
    'storage_capacity_available': avail_caps,
    'eligibility_label': returns_df['hien_eligible'].astype(int)
})

ai_data.to_csv(os.path.join(out_dir, 'ai_training_data.csv'), index=False)

# Train/Val/Test Split (70% / 15% / 15%)
shuffled_indices = rng.permutation(len(ai_data))
n_train = int(0.70 * len(ai_data))
n_val = int(0.15 * len(ai_data))

train_idx = shuffled_indices[:n_train]
val_idx = shuffled_indices[n_train:n_train+n_val]
test_idx = shuffled_indices[n_train+n_val:]

train_df = ai_data.iloc[train_idx]
val_df = ai_data.iloc[val_idx]
test_df = ai_data.iloc[test_idx]

train_df.to_csv(os.path.join(ml_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(ml_dir, 'validation.csv'), index=False)
test_df.to_csv(os.path.join(ml_dir, 'test.csv'), index=False)

print(f"ML splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# -------------------------------------------------------------
# 6. ENHANCE INVENTORY SNAPSHOTS (inventory_snapshots.csv)
# -------------------------------------------------------------
print("Generating comprehensive inventory_snapshots.csv...")
start_date = datetime(2026, 7, 1)
days = 90

returns_df['date'] = pd.to_datetime(returns_df['return_timestamp']).dt.date
sold_returns = returns_df[returns_df['sold_via_order_id'].notna()].copy()
sold_returns['sale_date'] = pd.to_datetime(sold_returns['resold_timestamp']).dt.date

daily_ret = returns_df[returns_df['hien_eligible']].groupby(['date', 'return_location_id', 'product_id']).size().to_dict()
daily_sale = sold_returns.groupby(['sale_date', 'return_location_id', 'product_id']).size().to_dict()

all_skus = products_df['product_id'].values
active_locs = [r['location_id'] for r in loc_rows if r['location_id'] != 'WH001']

current_stock = {(loc, sku): int(rng.integers(0, 3)) for loc in active_locs for sku in all_skus}

snapshot_rows = []
for d in range(days):
    curr_d = (start_date + timedelta(days=d)).date()
    curr_str = curr_d.strftime('%Y-%m-%d')
    for loc in active_locs:
        loc_cap = L_dict[loc]['storage_capacity_units']
        tot_units = sum(current_stock[(loc, s)] for s in all_skus)
        avail_cap = max(0, loc_cap - tot_units)
        
        for sku in all_skus:
            opening = current_stock[(loc, sku)]
            ret_u = daily_ret.get((curr_d, loc, sku), 0)
            sale_u = daily_sale.get((curr_d, loc, sku), 0)
            sale_u = min(sale_u, opening + ret_u)
            
            trans_in = 1 if (opening + ret_u - sale_u == 0 and rng.random() < 0.05) else 0
            trans_out = 1 if (opening > 3 and rng.random() < 0.10) else 0
            
            closing = max(0, opening + ret_u + trans_in - sale_u - trans_out)
            current_stock[(loc, sku)] = closing
            
            if opening > 0 or ret_u > 0 or sale_u > 0 or closing > 0:
                snapshot_rows.append({
                    'snapshot_date': curr_str,
                    'location_id': loc,
                    'product_id': sku,
                    'opening_inventory_units': opening,
                    'returned_inventory_units': ret_u,
                    'local_sales_units': sale_u,
                    'transfer_in_units': trans_in,
                    'transfer_out_units': trans_out,
                    'closing_inventory_units': closing,
                    'available_capacity_units': avail_cap
                })

inventory_snapshots_df = pd.DataFrame(snapshot_rows)
inventory_snapshots_df.to_csv(os.path.join(out_dir, 'inventory_snapshots.csv'), index=False)
print(f"Inventory snapshots generated: {len(inventory_snapshots_df)} rows.")

# -------------------------------------------------------------
# 7. ENHANCE DEMAND FORECAST (demand_forecast.csv)
# -------------------------------------------------------------
print("Generating comprehensive demand_forecast.csv...")
forecast_rows = []
orders_df['date'] = pd.to_datetime(orders_df['order_timestamp']).dt.date
order_counts = orders_df.groupby(['product_id', 'nearest_location_id', 'date']).size().to_dict()

for d in range(7, days, 7):
    f_date = (start_date + timedelta(days=d)).date()
    f_str = f_date.strftime('%Y-%m-%d')
    for loc in active_locs:
        for sku in all_skus:
            h7 = sum(order_counts.get((sku, loc, (f_date - timedelta(days=i))), 0) for i in range(1, 8))
            h30 = sum(order_counts.get((sku, loc, (f_date - timedelta(days=i))), 0) for i in range(1, min(d+1, 31)))
            avg_daily = round(h7 / 7.0, 2)
            
            trend_val = (h7 / 7.0) - (h30 / min(d, 30))
            trend = 'Upward' if trend_val > 0.15 else ('Downward' if trend_val < -0.15 else 'Stable')
            
            seasonal_factor = round(1.0 + 0.15 * np.sin(d / 7.0 * np.pi), 2)
            pred_demand = int(np.clip(round(h7 * seasonal_factor * rng.uniform(0.85, 1.20) + rng.normal(0, 0.5)), 0, 50))
            
            if pred_demand == 0:
                dem_level = 'Low'
            elif pred_demand <= 4:
                dem_level = 'Medium'
            elif pred_demand <= 10:
                dem_level = 'High'
            else:
                dem_level = 'Very High'
            
            conf = round(float(np.clip(0.70 + 0.05 * (h30 > 5) + rng.normal(0, 0.05), 0.50, 0.98)), 2)
            
            if h30 > 0 or pred_demand > 0 or rng.random() < 0.20:
                forecast_rows.append({
                    'product_id': sku,
                    'location_id': loc,
                    'forecast_date': f_str,
                    'historical_demand_7d': h7,
                    'historical_demand_30d': h30,
                    'average_daily_demand': avg_daily,
                    'demand_trend': trend,
                    'seasonal_factor': seasonal_factor,
                    'predicted_demand': pred_demand,
                    'demand_confidence': conf,
                    'demand_level': dem_level
                })

demand_forecast_df = pd.DataFrame(forecast_rows)
demand_forecast_df.to_csv(os.path.join(out_dir, 'demand_forecast.csv'), index=False)
print(f"Demand forecast generated: {len(demand_forecast_df)} rows.")

# -------------------------------------------------------------
# 8. BUILD INTELLIGENT MATCHING ENGINE DATASET (customer_inventory_matches.csv)
# -------------------------------------------------------------
print("Building customer_inventory_matches.csv with real matching formula...")

def calc_distance_score(dist_km):
    if dist_km <= 2.0:
        return np.clip(100.0 - dist_km * 4.0, 92.0, 100.0)
    elif dist_km <= 5.0:
        return np.clip(90.0 - (dist_km - 2.0) * 5.0, 75.0, 90.0)
    elif dist_km <= 10.0:
        return np.clip(75.0 - (dist_km - 5.0) * 5.0, 50.0, 75.0)
    elif dist_km <= 15.0:
        return np.clip(50.0 - (dist_km - 10.0) * 5.0, 25.0, 50.0)
    else:
        return np.clip(25.0 - (dist_km - 15.0) * 2.0, 0.0, 25.0)

def calc_condition_score(ai_grade):
    if ai_grade == 'New/Sealed':
        return 98.0
    elif ai_grade == 'Like New':
        return 88.0
    elif ai_grade == 'Minor Wear':
        return 50.0
    else:
        return 15.0

def calc_urgency_score(promised_delivery):
    return 95.0 if promised_delivery == 'Express' else 75.0

def calc_demand_score(local_dem_score):
    return round(float(np.clip(local_dem_score * 100.0, 10.0, 100.0)), 1)

def calc_cost_score(local_cost, wh_cost):
    savings_pct = (wh_cost - local_cost) / wh_cost if wh_cost > 0 else 0
    return round(float(np.clip(50.0 + savings_pct * 50.0, 0.0, 100.0)), 1)

orders_dict = orders_df.set_index('order_id').to_dict('index')
orders_by_sku_dict = {}
for oid, o_row in orders_dict.items():
    s = o_row['product_id']
    orders_by_sku_dict.setdefault(s, []).append((oid, o_row))

matches_rows = []
decisions_rows = []
impact_rows = []

match_counter = 1
dec_counter = 1

for idx, ret in returns_df.iterrows():
    rid = ret['return_id']
    sku = ret['product_id']
    loc_id = ret['return_location_id']
    h_elig = ret['hien_eligible']
    final_st = ret['final_status']
    sold_ord = ret['sold_via_order_id']
    el_reason = ret['eligibility_reason']
    
    loc_x = L_dict[loc_id]['x_km']
    loc_y = L_dict[loc_id]['y_km']
    
    cand_matches = []
    
    if pd.notna(sold_ord) and sold_ord in orders_dict:
        win_ord = orders_dict[sold_ord]
        win_dist = float(win_ord['delivery_distance_km'])
        p_match = 100.0
        d_score = calc_demand_score(ret['local_demand_score'])
        dist_score = calc_distance_score(win_dist)
        cond_score = calc_condition_score(ret['ai_condition_grade'])
        urg_score = calc_urgency_score(win_ord['promised_delivery'])
        
        loc_cost = float(win_ord['delivery_cost'])
        wh_dist = float(np.hypot(win_ord['customer_x_km'], win_ord['customer_y_km']))
        wh_cost = round(3.5 + 0.12 * wh_dist + 0.85, 2)
        cost_sc = calc_cost_score(loc_cost, wh_cost)
        
        overall = round(0.40 * p_match + 0.20 * d_score + 0.15 * dist_score + 0.10 * cond_score + 0.10 * urg_score + 0.05 * cost_sc, 1)
        overall = max(70.5, overall)
        
        winning_match = {
            'match_id': f"MAT{match_counter:06d}",
            'return_id': rid,
            'product_id': sku,
            'inventory_location_id': loc_id,
            'customer_id': win_ord['customer_id'],
            'customer_location_id': win_ord['nearest_location_id'],
            'order_id': sold_ord,
            'distance_km': win_dist,
            'product_match_score': p_match,
            'demand_score': d_score,
            'distance_score': round(dist_score, 1),
            'condition_score': cond_score,
            'urgency_score': urg_score,
            'cost_score': cost_sc,
            'overall_match_score': overall,
            'predicted_delivery_time_hours': float(win_ord['delivery_hours']),
            'local_fulfillment_cost_inr': round(loc_cost * 10.0, 2),
            'warehouse_fulfillment_cost_inr': round(wh_cost * 10.0, 2),
            'recommended_fulfillment_method': 'HIEN Local',
            'match_status': 'Fulfilled',
            'rejection_reason': 'None'
        }
        cand_matches.append(winning_match)
        match_counter += 1
        
        # Competing candidates
        sku_orders = orders_by_sku_dict.get(sku, [])
        cand_count = 0
        for c_oid, c_ord in sku_orders:
            if c_oid == sold_ord:
                continue
            c_dist = round(float(np.hypot(c_ord['customer_x_km'] - loc_x, c_ord['customer_y_km'] - loc_y)), 2)
            c_dist_sc = calc_distance_score(c_dist)
            c_urg_sc = calc_urgency_score(c_ord['promised_delivery'])
            c_wh_dist = float(np.hypot(c_ord['customer_x_km'], c_ord['customer_y_km']))
            c_loc_cost = round(1.2 + 0.35 * c_dist, 2)
            c_wh_cost = round(3.5 + 0.12 * c_wh_dist + 0.85, 2)
            c_cost_sc = calc_cost_score(c_loc_cost, c_wh_cost)
            c_overall = round(0.40 * 100.0 + 0.20 * d_score + 0.15 * c_dist_sc + 0.10 * cond_score + 0.10 * c_urg_sc + 0.05 * c_cost_sc, 1)
            if c_overall >= overall:
                c_overall = round(overall - rng.uniform(2.0, 8.0), 1)
            
            c_status = 'Candidate' if c_overall >= 70.0 else 'Rejected'
            c_rej = 'None' if c_overall >= 70.0 else 'Score below threshold (70.0)'
            
            cand_matches.append({
                'match_id': f"MAT{match_counter:06d}",
                'return_id': rid,
                'product_id': sku,
                'inventory_location_id': loc_id,
                'customer_id': c_ord['customer_id'],
                'customer_location_id': c_ord['nearest_location_id'],
                'order_id': c_oid,
                'distance_km': c_dist,
                'product_match_score': 100.0,
                'demand_score': d_score,
                'distance_score': round(c_dist_sc, 1),
                'condition_score': cond_score,
                'urgency_score': c_urg_sc,
                'cost_score': c_cost_sc,
                'overall_match_score': c_overall,
                'predicted_delivery_time_hours': round(float(np.clip(3.0 + 0.2 * c_dist, 1.5, 12.0)), 1),
                'local_fulfillment_cost_inr': round(c_loc_cost * 10.0, 2),
                'warehouse_fulfillment_cost_inr': round(c_wh_cost * 10.0, 2),
                'recommended_fulfillment_method': 'HIEN Local' if c_overall >= 70.0 else 'Central Warehouse',
                'match_status': c_status,
                'rejection_reason': c_rej
            })
            match_counter += 1
            cand_count += 1
            if cand_count >= 2:
                break
        
        # HIEN Decision
        decisions_rows.append({
            'decision_id': f"DEC{dec_counter:06d}",
            'return_id': rid,
            'product_id': sku,
            'customer_id': win_ord['customer_id'],
            'order_id': sold_ord,
            'inventory_location_id': loc_id,
            'eligibility_prediction': 'YES',
            'eligibility_confidence': ret['ai_confidence'],
            'demand_level': 'High' if d_score >= 60 else 'Medium',
            'match_score': winning_match['overall_match_score'],
            'distance_km': win_dist,
            'local_fulfillment_cost_inr': winning_match['local_fulfillment_cost_inr'],
            'warehouse_fulfillment_cost_inr': winning_match['warehouse_fulfillment_cost_inr'],
            'capacity_available': 'YES',
            'final_decision': 'LOCAL_FULFILLMENT',
            'decision_reason': 'Optimal hyperlocal match verified with score >= 70'
        })
        dec_counter += 1
        
        # Impact record
        d_store_wh = float(L_dict[loc_id]['dist_to_warehouse_km'])
        trad_dist = round(d_store_wh + wh_dist, 2)
        hien_dist = win_dist
        dist_saved = round(max(0.0, trad_dist - hien_dist), 2)
        
        trad_cost = round((float(ret['baseline_return_cost']) + wh_cost) * 10.0, 2)
        hien_cost = round((float(ret['hien_return_cost']) + loc_cost) * 10.0, 2)
        cost_saved = round(max(0.0, trad_cost - hien_cost), 2)
        
        trad_time = round(float(np.clip(48.0 + 0.5 * trad_dist, 24.0, 96.0)), 1)
        hien_time = float(win_ord['delivery_hours'])
        time_saved = round(max(0.0, trad_time - hien_time), 1)
        
        trad_co2 = round(0.12 * trad_dist + 1.2, 3)
        hien_co2 = round(0.12 * hien_dist, 3)
        co2_saved = round(max(0.0, trad_co2 - hien_co2), 3)
        
        impact_rows.append({
            'match_id': winning_match['match_id'],
            'return_id': rid,
            'order_id': sold_ord,
            'product_id': sku,
            'traditional_route_distance_km': trad_dist,
            'hien_route_distance_km': hien_dist,
            'distance_saved_km': dist_saved,
            'traditional_transport_cost_inr': trad_cost,
            'hien_transport_cost_inr': hien_cost,
            'transportation_cost_saved_inr': cost_saved,
            'traditional_processing_time_hours': trad_time,
            'hien_processing_time_hours': hien_time,
            'time_saved_hours': time_saved,
            'traditional_warehouse_processing_required': 1,
            'hien_warehouse_processing_required': 0,
            'traditional_co2_kg': trad_co2,
            'hien_co2_kg': hien_co2,
            'co2_saved_kg': co2_saved
        })
        
    elif h_elig and final_st == 'Expired - Sent to Warehouse':
        d_score = calc_demand_score(ret['local_demand_score'])
        cond_score = calc_condition_score(ret['ai_condition_grade'])
        
        sku_orders = orders_by_sku_dict.get(sku, [])
        if len(sku_orders) > 0:
            c_oid, c_ord = sku_orders[0]
            c_dist = round(float(np.hypot(c_ord['customer_x_km'] - loc_x, c_ord['customer_y_km'] - loc_y)), 2)
            c_dist_sc = calc_distance_score(c_dist)
            c_urg_sc = calc_urgency_score(c_ord['promised_delivery'])
            c_wh_dist = float(np.hypot(c_ord['customer_x_km'], c_ord['customer_y_km']))
            c_loc_cost = round(1.2 + 0.35 * c_dist, 2)
            c_wh_cost = round(3.5 + 0.12 * c_wh_dist + 0.85, 2)
            c_cost_sc = calc_cost_score(c_loc_cost, c_wh_cost)
            c_overall = round(0.40 * 100.0 + 0.20 * d_score + 0.15 * c_dist_sc + 0.10 * cond_score + 0.10 * c_urg_sc + 0.05 * c_cost_sc, 1)
            c_overall = min(68.5, c_overall)
            
            rej_res = 'Exceeds max delivery radius' if c_dist > 10.0 else 'Score below threshold (70.0)'
            cand_matches.append({
                'match_id': f"MAT{match_counter:06d}",
                'return_id': rid,
                'product_id': sku,
                'inventory_location_id': loc_id,
                'customer_id': c_ord['customer_id'],
                'customer_location_id': c_ord['nearest_location_id'],
                'order_id': c_oid,
                'distance_km': c_dist,
                'product_match_score': 100.0,
                'demand_score': d_score,
                'distance_score': round(c_dist_sc, 1),
                'condition_score': cond_score,
                'urgency_score': c_urg_sc,
                'cost_score': c_cost_sc,
                'overall_match_score': c_overall,
                'predicted_delivery_time_hours': 36.0,
                'local_fulfillment_cost_inr': round(c_loc_cost * 10.0, 2),
                'warehouse_fulfillment_cost_inr': round(c_wh_cost * 10.0, 2),
                'recommended_fulfillment_method': 'Central Warehouse',
                'match_status': 'Expired',
                'rejection_reason': rej_res
            })
            match_counter += 1
        else:
            cand_matches.append({
                'match_id': f"MAT{match_counter:06d}",
                'return_id': rid,
                'product_id': sku,
                'inventory_location_id': loc_id,
                'customer_id': 'None',
                'customer_location_id': 'None',
                'order_id': 'None',
                'distance_km': 0.0,
                'product_match_score': 100.0,
                'demand_score': d_score,
                'distance_score': 0.0,
                'condition_score': cond_score,
                'urgency_score': 50.0,
                'cost_score': 50.0,
                'overall_match_score': round(0.40 * 100.0 + 0.20 * d_score + 0.10 * cond_score, 1),
                'predicted_delivery_time_hours': 48.0,
                'local_fulfillment_cost_inr': 0.0,
                'warehouse_fulfillment_cost_inr': 85.0,
                'recommended_fulfillment_method': 'Central Warehouse',
                'match_status': 'No Demand',
                'rejection_reason': 'Dwell time exceeded (14 days) without local order'
            })
            match_counter += 1
            
        decisions_rows.append({
            'decision_id': f"DEC{dec_counter:06d}",
            'return_id': rid,
            'product_id': sku,
            'customer_id': 'None',
            'order_id': 'None',
            'inventory_location_id': loc_id,
            'eligibility_prediction': 'YES',
            'eligibility_confidence': ret['ai_confidence'],
            'demand_level': 'Low',
            'match_score': cand_matches[0]['overall_match_score'] if cand_matches else 55.0,
            'distance_km': cand_matches[0]['distance_km'] if cand_matches else 0.0,
            'local_fulfillment_cost_inr': 0.0,
            'warehouse_fulfillment_cost_inr': 85.0,
            'capacity_available': 'YES',
            'final_decision': 'CENTRAL_WAREHOUSE',
            'decision_reason': 'Holding period expired (14d); transfer to Central Warehouse'
        })
        dec_counter += 1
        
    else:
        d_score = calc_demand_score(ret['local_demand_score'])
        cond_score = calc_condition_score(ret['ai_condition_grade'])
        final_dec = 'CENTRAL_WAREHOUSE' if ret['routing_decision'] == 'Return to Warehouse' else 'MANUAL_INSPECTION'
        
        decisions_rows.append({
            'decision_id': f"DEC{dec_counter:06d}",
            'return_id': rid,
            'product_id': sku,
            'customer_id': 'None',
            'order_id': 'None',
            'inventory_location_id': loc_id,
            'eligibility_prediction': 'NO',
            'eligibility_confidence': ret['ai_confidence'],
            'demand_level': 'Medium',
            'match_score': round(0.40 * 100.0 + 0.10 * cond_score + 0.20 * d_score, 1),
            'distance_km': 0.0,
            'local_fulfillment_cost_inr': 0.0,
            'warehouse_fulfillment_cost_inr': 85.0,
            'capacity_available': 'YES',
            'final_decision': final_dec,
            'decision_reason': f"Ineligible for HIEN: {el_reason}"
        })
        dec_counter += 1

    matches_rows.extend(cand_matches)

matches_df = pd.DataFrame(matches_rows)
matches_df.to_csv(os.path.join(out_dir, 'customer_inventory_matches.csv'), index=False)
print(f"Customer matches generated: {len(matches_df)} rows.")

decisions_df = pd.DataFrame(decisions_rows)
decisions_df.to_csv(os.path.join(out_dir, 'hien_decisions.csv'), index=False)
print(f"HIEN decisions generated: {len(decisions_df)} rows.")

impact_df = pd.DataFrame(impact_rows)
impact_df.to_csv(os.path.join(out_dir, 'impact_comparison.csv'), index=False)
print(f"Impact comparisons generated: {len(impact_df)} rows.")

# -------------------------------------------------------------
# 9. ASSUMPTIONS (assumptions.csv)
# -------------------------------------------------------------
print("Creating assumptions.csv...")
assumptions_data = [
    {"parameter": "grid_dimension_km", "value": "60x60", "unit": "km", "explanation": "Synthetic metro simulation area bounding box"},
    {"parameter": "central_warehouse_coord", "value": "(0.0, 0.0)", "unit": "coordinate", "explanation": "Central Warehouse coordinates at center of grid"},
    {"parameter": "metro_center_latitude", "value": "12.9716", "unit": "degrees", "explanation": "Bengaluru metro area geographic latitude datum"},
    {"parameter": "metro_center_longitude", "value": "77.5946", "unit": "degrees", "explanation": "Bengaluru metro area geographic longitude datum"},
    {"parameter": "max_dwell_time_days", "value": "14", "unit": "days", "explanation": "Max holding duration for local returned inventory before warehouse transfer"},
    {"parameter": "max_hyperlocal_radius_km", "value": "10.0", "unit": "km", "explanation": "Maximum allowable customer delivery radius from a local store hub"},
    {"parameter": "match_score_threshold", "value": "70.0", "unit": "score (0-100)", "explanation": "Minimum overall match score required to recommend HIEN Local fulfillment"},
    {"parameter": "match_weight_product", "value": "0.40", "unit": "weight", "explanation": "Weight for product/SKU exact match component"},
    {"parameter": "match_weight_demand", "value": "0.20", "unit": "weight", "explanation": "Weight for local demand score component"},
    {"parameter": "match_weight_distance", "value": "0.15", "unit": "weight", "explanation": "Weight for customer proximity distance component"},
    {"parameter": "match_weight_condition", "value": "0.10", "unit": "weight", "explanation": "Weight for verified return condition component"},
    {"parameter": "match_weight_urgency", "value": "0.10", "unit": "weight", "explanation": "Weight for order SLA / delivery urgency component"},
    {"parameter": "match_weight_cost", "value": "0.05", "unit": "weight", "explanation": "Weight for relative fulfillment cost component"},
    {"parameter": "transportation_cost_per_km_local", "value": "3.50", "unit": "INR / km", "explanation": "Local EV/bike delivery cost rate per kilometer"},
    {"parameter": "transportation_cost_per_km_warehouse", "value": "1.20", "unit": "INR / km", "explanation": "Inter-city / trunk freight cost rate per kilometer"},
    {"parameter": "warehouse_handling_cost_inr", "value": "85.00", "unit": "INR / unit", "explanation": "Cost of receiving, QA inspecting, and restocking at Central Warehouse"},
    {"parameter": "local_handling_cost_inr", "value": "14.00", "unit": "INR / unit", "explanation": "Cost of verification and shelf placement at Local Hub Store"},
    {"parameter": "estimated_co2_per_km_local", "value": "0.12", "unit": "kg CO2 / km", "explanation": "Local transport emission factor per kilometer"},
    {"parameter": "estimated_co2_per_km_warehouse", "value": "0.12", "unit": "kg CO2 / km", "explanation": "Warehouse freight emission factor per kilometer"},
    {"parameter": "warehouse_processing_fixed_co2", "value": "0.90", "unit": "kg CO2 / unit", "explanation": "Emissions from automated warehouse sorting and lighting"},
    {"parameter": "average_warehouse_processing_hours", "value": "38.2", "unit": "hours", "explanation": "Average turnaround time for central warehouse order processing and transit"},
    {"parameter": "average_local_verification_hours", "value": "4.3", "unit": "hours", "explanation": "Average turnaround time for local verification and same-day delivery"}
]

assumptions_df = pd.DataFrame(assumptions_data)
assumptions_df.to_csv(os.path.join(out_dir, 'assumptions.csv'), index=False)

# -------------------------------------------------------------
# 10. BUSINESS IMPACT SUMMARY (business_impact_summary.csv)
# -------------------------------------------------------------
print("Calculating business_impact_summary.csv...")
tot_returns = len(returns_df)
hien_elig_returns = int(returns_df['hien_eligible'].sum())
local_fulfil_count = len(impact_df)
wh_fulfil_count = len(orders_df) - local_fulfil_count
local_fulfil_rate = round(local_fulfil_count / len(orders_df) * 100.0, 2)

tot_trad_dist = round(impact_df['traditional_route_distance_km'].sum(), 2)
tot_hien_dist = round(impact_df['hien_route_distance_km'].sum(), 2)
tot_dist_saved = round(impact_df['distance_saved_km'].sum(), 2)
dist_reduc_pct = round((tot_dist_saved / tot_trad_dist) * 100.0, 2)

tot_trad_cost = round(impact_df['traditional_transport_cost_inr'].sum(), 2)
tot_hien_cost = round(impact_df['hien_transport_cost_inr'].sum(), 2)
tot_cost_saved = round(impact_df['transportation_cost_saved_inr'].sum(), 2)
cost_reduc_pct = round((tot_cost_saved / tot_trad_cost) * 100.0, 2)

tot_trad_co2 = round(impact_df['traditional_co2_kg'].sum(), 3)
tot_hien_co2 = round(impact_df['hien_co2_kg'].sum(), 3)
tot_co2_saved = round(impact_df['co2_saved_kg'].sum(), 3)
co2_reduc_pct = round((tot_co2_saved / tot_trad_co2) * 100.0, 2)

avg_trad_time = round(impact_df['traditional_processing_time_hours'].mean(), 2)
avg_hien_time = round(impact_df['hien_processing_time_hours'].mean(), 2)
avg_trad_turnaround_days = round(avg_trad_time / 24.0, 2)
avg_hien_turnaround_days = round(avg_hien_time / 24.0, 2)
time_reduc_pct = round(((avg_trad_time - avg_hien_time) / avg_trad_time) * 100.0, 2)

wh_cycles_avoided = local_fulfil_count

summary_data = [{
    'total_returns': tot_returns,
    'hien_eligible_returns': hien_elig_returns,
    'local_fulfillment_count': local_fulfil_count,
    'central_warehouse_count': wh_fulfil_count,
    'local_fulfillment_rate': local_fulfil_rate,
    'total_traditional_distance_km': tot_trad_dist,
    'total_hien_distance_km': tot_hien_dist,
    'total_distance_saved_km': tot_dist_saved,
    'distance_reduction_percent': dist_reduc_pct,
    'total_traditional_cost_inr': tot_trad_cost,
    'total_hien_cost_inr': tot_hien_cost,
    'total_cost_saved_inr': tot_cost_saved,
    'cost_reduction_percent': cost_reduc_pct,
    'total_traditional_co2_kg': tot_trad_co2,
    'total_hien_co2_kg': tot_hien_co2,
    'total_co2_saved_kg': tot_co2_saved,
    'co2_reduction_percent': co2_reduc_pct,
    'average_inventory_turnaround_days': avg_trad_turnaround_days,
    'average_hien_turnaround_days': avg_hien_turnaround_days,
    'turnaround_time_reduction_percent': time_reduc_pct,
    'warehouse_processing_cycles_avoided': wh_cycles_avoided
}]

business_impact_df = pd.DataFrame(summary_data)
business_impact_df.to_csv(os.path.join(out_dir, 'business_impact_summary.csv'), index=False)
print("Business impact summary generated successfully.")
