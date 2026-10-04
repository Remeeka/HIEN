# Hyperlocal Inventory Exchange Network (HIEN)
## Decision-Intelligence Platform for Hyperlocal Reverse Logistics

HIEN is an autonomous decision system that evaluates returned e-commerce merchandise in real time to determine whether items can be verified and fulfilled locally to neighboring customer demand, or routed back through the traditional central distribution warehouse.

---

## 1. System Architecture

```
                    +------------------------+
                    |    Customer Return     |
                    +-----------+------------+
                                |
                                v
                    +------------------------+
                    |  AI Return Eligibility |
                    | (Random Forest Engine) |
                    +-----------+------------+
                                |
                 +--------------+--------------+
                 |                             |
       [Ineligible / High Risk]           [Eligible]
                 |                             |
                 v                             v
    +-------------------------+   +-------------------------+
    |    Central Warehouse    |   |    Local Verification   |
    |   Routing / Liquidation |   |    & Hub Placement      |
    +-------------------------+   +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  | Hyperlocal Inventory    |
                                  | Pool & Demand Forecast  |
                                  | (HistGradientBoosting)  |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  | Multi-Candidate Match   |
                                  | & Scoring Engine        |
                                  +------------+------------+
                                               |
                                +--------------+--------------+
                                |                             |
                        [Score >= 70.0]                [Score < 70.0 /
                                |                      Dwell Expired]
                                v                             |
                    +-------------------------+               v
                    | HIEN Local Fulfillment  |   +-------------------------+
                    | (Same-Day / Low CO2)    |   | Transfer to Central     |
                    +-------------------------+   | Warehouse Fulfillment   |
                                                  +-------------------------+
```

---

## 2. Machine Learning Demand Forecasting Engine

HIEN integrates a production-grade **`sklearn.ensemble.HistGradientBoostingRegressor`** to forecast future consumer purchase demand across all 40 local fulfillment hubs and 9 product categories.

### 2.1 Model Specifications & Chronological Split
- **Algorithm:** `sklearn.ensemble.HistGradientBoostingRegressor` (Histogram-based Gradient Boosting with native categorical support)
- **Primary Forecast Target:** `future_demand_7d` (7-day forward consumer order volume in units)
- **Auxiliary Targets:** `future_demand_3d` (3-day) and `future_demand_1d` (1-day)
- **Time-Series Split:** Strict chronological split (No random shuffling, preventing future data leakage)
  - **Training Set (70.0%):** `2026-07-15` to `2026-09-01` (17,640 records)
  - **Validation Set (14.3%):** `2026-09-02` to `2026-09-11` (3,600 records)
  - **Test Set (15.7%):** `2026-09-12` to `2026-09-22` (3,960 records)

### 2.2 Leakage-Safe Feature Engineering (29 Features)
- **Categorical Hub & Category Encodings:** `location_id`, `category`, `zone`, `location_type`
- **Historical Lags (strictly backward-looking):** `lag_1_demand`, `lag_2_demand`, `lag_3_demand`, `lag_7_demand`, `lag_14_demand`
- **Rolling Demand Windows:** `rolling_mean_7`, `rolling_std_7`, `rolling_mean_14`, `rolling_mean_30`
- **Cumulative Historical Volume:** `historical_demand_7d`, `historical_demand_14d`, `historical_demand_30d`, `historical_order_count_7d`
- **Temporal Calendar:** `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_weekend`
- **Hub Capacity & Physical Constraints:** `storage_capacity_units`, `current_inventory_units`, `available_storage_units`, `utilization_rate`, `dist_to_warehouse_km`, `avg_daily_footfall`
- **Category Characteristics:** `avg_unit_price`, `avg_weight_kg`, `avg_demand_weight`, `hygiene_sensitive`, `fragile`

### 2.3 Evaluation Benchmark vs Baseline
| Evaluation Metric | Baseline Naive Model | HistGradientBoostingRegressor | Improvement |
| :--- | :--- | :--- | :--- |
| **Test Set MAE** | 2.264 units | **1.796 units** | **-20.7% error reduction** |
| **Test Set RMSE** | 3.195 units | **2.489 units** | **-22.1% error reduction** |
| **Test Set $R^2$ Score** | 0.5782 | **0.7440** | **+0.1658 explainability gain** |

### 2.4 Transparent Demand Pressure Score Formula
To translate raw predicted demand units into a standardized decision metric for the reverse matching engine, HIEN computes the **Demand Pressure Score** ($0 - 100$):

$$\text{Demand Pressure Score} = \min\left(100.0, \max\left(0.0, \frac{\text{Predicted 7d Demand Units}}{\text{Available Hub Storage Units} \cdot 0.15 + 1.0} \times 50.0\right)\right)$$

- **Critical High Demand ($\ge 75$):** Immediate priority matching with nearby return inventory.
- **High Demand ($50 - 74$):** High local resale velocity; hold in local buffer.
- **Moderate Demand ($25 - 49$):** Standard hyperlocal exchange pool.
- **Low Demand ($< 25$):** Low forward velocity; route to central warehouse.

---

## 3. Directory & Component Structure

```
HEIN/
├── backend/                                # Python FastAPI Decision Engine
│   ├── main.py                             # REST API Endpoints & Business Logic
│   ├── train_model.py                      # Random Forest Return Eligibility Training Pipeline
│   ├── ml/
│   │   └── demand/
│   │       ├── train_demand_model.py       # HistGradientBoostingRegressor Demand Training
│   │       └── predict_demand.py           # Batch Inference & Demand Scoring Engine
│   └── model/
│       ├── eligibility_model.joblib        # Serialized Eligibility Classifier Pipeline
│       ├── model_metadata.json             # Eligibility Model Metrics
│       ├── demand_hist_gradient_boosting.joblib # Serialized Multi-Horizon Demand Regressor Bundle
│       ├── demand_features.json            # Feature Metadata & Permutation Importances
│       ├── demand_metrics.json             # MAE, RMSE, R2, and Chronological Split Details
│       └── demand_predictions.csv          # Precomputed Hub x Category Forecasts & Scores
│
├── frontend/                               # Next.js 16 + React + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                    # Executive Dashboard Home
│   │   │   ├── returns/page.tsx            # Return Intake Stream
│   │   │   ├── returns/[id]/page.tsx       # Return Detail & AI Model Assessment
│   │   │   ├── verification/page.tsx       # Local Hub Physical QA Workflow
│   │   │   ├── inventory/page.tsx          # Hyperlocal Inventory Pools & Capacity
│   │   │   ├── demand/page.tsx             # Demand Intelligence, Multi-Horizon Tabs & Live Playground
│   │   │   ├── matching/page.tsx           # Multi-Candidate Matching Engine
│   │   │   ├── decisions/page.tsx          # Fulfillment Decisions & Explainability ("Why?")
│   │   │   ├── demo/page.tsx               # 9-Stage Interactive Transaction Simulator (HERO)
│   │   │   ├── impact/page.tsx             # Business & Environmental Impact Analysis
│   │   │   ├── network/page.tsx            # Spatial Grid Metro Map & Routing Topology
│   │   │   └── system/page.tsx             # Data Quality, Integrity & Dual ML Benchmark Status
│   │   ├── components/
│   │   │   ├── Sidebar.tsx                 # Navigation Sidebar
│   │   │   └── Header.tsx                  # Global Search & Live System Clock
│   │   └── lib/api.ts                      # Client API Layer
│   └── package.json
│
├── hien_dataset/                           # Enhanced Dataset (Single Source of Truth)
│   ├── returns.csv                         # 7,000 Return records
│   ├── orders.csv                          # 24,000 Purchase Orders
│   ├── products.csv                        # 126 SKU Catalog
│   ├── customers.csv                       # 10,000 Registered Customers
│   ├── locations.csv                       # 41 Hub Nodes (40 Stores/APs + 1 Central WH)
│   ├── inventory_snapshots.csv             # 422,961 Daily SKU-level inventory flows
│   ├── demand_forecast.csv                 # 43,385 Demand predictions & historical stats
│   ├── customer_inventory_matches.csv      # 8,421 Candidate match evaluations
│   ├── hien_decisions.csv                  # 7,000 Final routing decisions
│   ├── impact_comparison.csv               # 2,693 Fulfilled item impact comparisons
│   ├── business_impact_summary.csv         # Network-wide aggregate savings metrics
│   ├── assumptions.csv                     # 22 Prototype engineering parameters
│   ├── ai_training_data.csv                # 7,000 ML Training samples
│   └── ml/                                 # Stratified ML Splits
│       ├── train.csv                       # 70% (4,900 samples)
│       ├── validation.csv                  # 15% (1,050 samples)
│       └── test.csv                        # 15% (1,050 samples)
```

---

## 4. Getting Started & Running Locally

### Step 1: Train & Generate Demand Predictions
```bash
# Train HistGradientBoostingRegressor on chronological time-series
python backend/ml/demand/train_demand_model.py

# Generate batch forecasts and scores
python backend/ml/demand/predict_demand.py
```

### Step 2: Start the Backend (FastAPI)
```bash
# From project root:
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation available at: `http://127.0.0.1:8000/docs`

### Step 3: Start the Frontend (Next.js)
```bash
# In a separate terminal, navigate to frontend:
cd frontend
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 5. API Endpoints for ML & Demand Intelligence

- `GET /api/demand`: Returns all hub-category forecasts across 1d, 3d, and 7d horizons, historical lags, rolling means, demand tiers, and category aggregates.
- `POST /api/ai/forecast-demand`: Real-time on-demand inference endpoint accepting `{"location_id": "LOC001", "category": "Electronics", "horizon_days": 7}` and returning predicted demand, demand score, and status tier.
- `GET /api/model/metrics`: Returns evaluation metrics (MAE, RMSE, $R^2$, confusion matrix, top feature importances) for both the Return Eligibility Classifier and Demand Regressor.
- `GET /api/system/health`: System-wide verification report across all 11 dataset tables and active model status.

---

## 6. Key UI Pages & Features

1. **Executive Dashboard (`/`)**: Network KPI metrics, fulfillment distribution chart, category breakdowns, dual live decision showcase cards, and recent decision stream.
2. **Demand Intelligence (`/demand`)**: Multi-horizon selector (1d, 3d, 7d), live interactive ML inference playground, category velocity charts, and hub-level forecast matrix.
3. **AI Assessment (`/returns/[id]`)**: Detailed physical return specs, observable signals, and an interactive **"Run AI Assessment"** trigger executing real ML inference on 16 features.
4. **Local Verification (`/verification`)**: Interactive physical QA workflow allowing operators to approve items into the local pool or route to warehouse.
5. **Hyperlocal Inventory (`/inventory`)**: Stock and capacity analytics across all 41 facilities, daily inventory flow trends, and utilization gauges.
6. **Matching Engine (`/matching`)**: Multi-candidate scoring engine breaking down the exact weighted formula with candidate comparisons.
7. **Fulfillment Decisions (`/decisions`)**: Complete system decision logs with structured **"Why?"** explainability cards.
8. **Interactive Demo Simulator (`/demo`) — HERO FEATURE**: Step-by-step 9-stage transaction lifecycle simulator with automated walkthroughs.
9. **Impact Analysis (`/impact`)**: Comprehensive distance, cost, turnaround time, and CO₂ savings analysis.
10. **Network Map (`/network`)**: Interactive 60x60 km spatial grid map plotting all 41 store hubs and warehouse nodes with customer route traces.
11. **System Health & ML Status (`/system`)**: Automated validation results, dual ML model benchmarks (Eligibility RF & Demand HistGradientBoosting).
