# HIEN Synthetic Dataset (seed=42)
Simulated 90 days (2026-07-01 to 2026-09-28) on a synthetic 60x60 km metro grid. Central warehouse at (0,0). All values are synthetic.

| File | Rows | Grain | Key columns |
|---|---|---|---|
| locations.csv | 40 | store / access point | location_type, x_km, y_km, storage_capacity_units, verification_slots_per_day, hub_enabled, dist_to_warehouse_km |
| products.csv | 126 | SKU | category, unit_price, weight_kg, hygiene_sensitive, fragile, base_resale_prob |
| returns.csv | 7,000 | one return | true_condition (ground truth), ai_condition_grade, ai_confidence, ai_fraud_risk_score, is_fraud_swap, hien_eligible, routing_decision, final_status, dwell_days, baseline vs hien cost/distance |
| orders.csv | 24,000 | one customer order | customer_x_km/y_km, fulfilled_from_type (HIEN Local / Central Warehouse), delivery_distance_km, delivery_hours, delivery_cost, co2_kg, source_return_id |
| inventory_snapshots.csv | daily | date x location x category | units_on_hand, capacity_utilisation |
| demand_forecast.csv | weekly | location x category x week | forecast_demand_units, actual_demand_units |

## Logic
- Eligibility: AI grade >= Like New, confidence >= 0.75, fraud score < 0.35, not hygiene-sensitive, hub enabled.
- Matching: each order takes the nearest in-pool unit of the same SKU within 10 km, held at most 14 days; otherwise ships from the warehouse.
- AI grading is noisier for Electronics; 2.5% of returns are fraud swaps; some damaged items are misgraded as resalable (`misgraded_as_resalable`).
- Cost model (assumptions, editable in gen.py): baseline = 5.0 + 0.5/km + 0.8/kg; HIEN local handling = 1.4 + 0.15/kg.
- Labels for ML: `true_condition`, `is_fraud_swap`, `actual_demand_units`.

## Headline results from this seed
Eligible returns 43.4% | resold locally 88.7% of eligible | return cost -40.4% | return distance -45.4% | local-fulfilled orders 11.2% | median delivery 4.3h vs 38.2h.
