# Hyperlocal Inventory Exchange Network (HIEN)
## Complete Synthetic Dataset & Benchmark Specification

---

## 1. Executive Summary & Problem Statement

In conventional e-commerce supply chains, customer returns follow a highly centralized, inefficient, and carbon-intensive reverse logistics workflow:
1. A customer initiates a return from their residence.
2. A logistics carrier transports the item hundreds of kilometers back to a **Central Distribution Warehouse**.
3. The item sits in processing queues for days or weeks awaiting inspection, grading, and restocking.
4. Meanwhile, a neighboring customer only a few kilometers away from the return origin places an order for the exact same item, which is then fulfilled and shipped all the way back from the Central Warehouse.

This traditional cycle incurs severe economic and environmental penalties:
- **Excess Transportation Costs & Delays:** Redundant long-haul transit cycles add 36–72 hours to delivery times.
- **Elevated Greenhouse Gas Emissions:** Double-handling generates up to 7–15 kg of CO₂ per return cycle.
- **Working Capital Lock-in:** High inventory dwell times diminish product residual value.

### The HIEN Solution
The **Hyperlocal Inventory Exchange Network (HIEN)** transforms local retail stores, access points, and micro-hubs into a distributed, intelligent inventory pool. When an item is returned:
1. **AI Return Eligibility** inspects condition, fraud risk, hygiene rules, and local demand feasibility.
2. **Local Verification** validates physical and functional integrity at a nearby Store / Access Point.
3. The verified unit enters the **Hyperlocal Inventory Pool**.
4. The **Intelligent Matching Engine** evaluates real-time incoming local customer orders across multi-criteria attributes (SKU match, proximity, demand strength, condition grade, urgency SLA, and cost delta).
5. If score threshold ($\ge 70.0$) is met, the system executes **Local Fulfillment** via short-range courier, completely avoiding warehouse round-trips.
6. If local demand is absent, capacity is constrained, or the match score is suboptimal, the system intelligently defaults to **Central Warehouse Fulfillment**.

---

## 2. HIEN System Architecture & Workflow

```
                   +------------------------+
                   |    Customer Return     |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   |  AI Return Eligibility |
                   |  (Condition & Fraud)   |
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

## 3. Dataset Inventory & Schema Data Dictionary

The enhanced HIEN dataset is organized into relational tables and dedicated ML split partitions:

```
hien_dataset/
├── returns.csv                    # Historical and simulated customer returns
├── orders.csv                     # Incoming customer demand orders
├── products.csv                   # Product catalog and SKU specifications
├── customers.csv                  # Registered customers and geographic profiles
├── locations.csv                  # Physical stores, access points, micro-hubs & warehouse
├── inventory_snapshots.csv        # Daily stock, returns, transfers, and available capacity
├── demand_forecast.csv            # 7-day and 30-day demand predictions by location/SKU
├── customer_inventory_matches.csv # Candidate scoring engine records and evaluations
├── hien_decisions.csv             # Final automated routing and fulfillment decisions
├── impact_comparison.csv          # Item-level traditional vs. HIEN cost/time/CO2 impact
├── business_impact_summary.csv    # Aggregate system-level performance metrics
├── assumptions.csv                # Prototype engineering and financial parameters
├── ai_training_data.csv           # Clean ML dataset (features vs. eligibility label)
├── ml/
│   ├── train.csv                  # 70% Training split (4,900 records)
│   ├── validation.csv             # 15% Validation split (1,050 records)
│   └── test.csv                   # 15% Test split (1,050 records)
└── README.md                      # Complete system documentation
```

### Table 1: `products.csv`
| Column | Type | Description |
|---|---|---|
| `product_id` | String (PK) | Unique SKU identifier (e.g. `SKU0001` to `SKU0126`). |
| `sku` | String | Secondary SKU code for legacy compatibility. |
| `category` | String | Product category (Apparel, Footwear, Electronics, Home & Kitchen, Beauty, Toys, Books, Sports, Accessories). |
| `unit_price` | Float | Retail selling price in USD / INR scale. |
| `weight_kg` | Float | Physical shipping weight in kilograms. |
| `hygiene_sensitive` | Integer (0/1) | Flag indicating if product cannot be resold locally due to hygiene restrictions (e.g., Beauty). |
| `fragile` | Integer (0/1) | Flag indicating whether item requires specialized handling. |
| `base_resale_prob` | Float | Baseline probability of item retaining resalable condition upon return. |
| `demand_weight` | Float | Relative product popularity weighting in demand generation. |

### Table 2: `locations.csv`
| Column | Type | Description |
|---|---|---|
| `location_id` | String (PK) | Location identifier (`LOC001`–`LOC040`, `WH001`). |
| `location_name` | String | Human-readable node name (e.g., `HIEN Indiranagar Hub Store`). |
| `city` | String | Metro city municipality (`Bengaluru`). |
| `state` | String | State administrative zone (`Karnataka`). |
| `latitude` | Float | Geographic latitude coordinate. |
| `longitude` | Float | Geographic longitude coordinate. |
| `x_km`, `y_km` | Float | Metric Cartesian grid coordinates relative to Central Warehouse origin (0, 0). |
| `zone` | String | Metro geographic zone (`North`, `South`, `East`, `West`, `Central`). |
| `location_type` | String | Facility type: `Store`, `Access Point`, `Central Warehouse`. |
| `storage_capacity_units` | Integer | Maximum inventory storage capacity in units. |
| `current_inventory_units`| Integer | Active stock on hand. |
| `available_storage_units`| Integer | Unallocated capacity (`storage_capacity_units - current_inventory_units`). |
| `verification_slots_per_day` | Integer | Daily QA inspection throughput capacity. |
| `operating_hours` | String | Operational working hours window. |
| `delivery_radius_km` | Float | Maximum standard hyperlocal delivery radius. |
| `hub_enabled` | Boolean | Operational flag indicating whether HIEN intake/dispatch is active. |
| `local_delivery_cost_per_km` | Float | Local courier delivery rate per km. |
| `warehouse_handling_cost_inr` | Float | Baseline handling and receiving cost at facility. |
| `avg_daily_footfall` | Integer | Daily retail customer walk-in traffic. |
| `dist_to_warehouse_km` | Float | Euclidean distance from location to Central Warehouse. |

### Table 3: `customers.csv`
| Column | Type | Description |
|---|---|---|
| `customer_id` | String (PK) | Unique customer code (`CUS000001`–`CUS010000`). |
| `customer_latitude` | Float | Customer home latitude. |
| `customer_longitude`| Float | Customer home longitude. |
| `customer_x_km` | Float | Grid coordinate $X$ in km. |
| `customer_y_km` | Float | Grid coordinate $Y$ in km. |
| `home_location_id` | String (FK) | Nearest physical HIEN hub store. |
| `preferred_delivery_radius_km` | Float | Customer tolerance radius for local delivery. |
| `customer_segment` | String | Buying behavior segment: `Frequent`, `Regular`, `Occasional`, `New`. |

### Table 4: `orders.csv`
| Column | Type | Description |
|---|---|---|
| `order_id` | String (PK) | Unique purchase order ID (`ORD000001`–`ORD024000`). |
| `customer_id` | String (FK) | Purchasing customer identifier (`CUS000001`–`CUS010000`). |
| `order_timestamp` | Timestamp | Timestamp of order placement. |
| `product_id` | String (FK) | Ordered SKU identifier. |
| `sku` | String | Legacy SKU code. |
| `customer_x_km`, `customer_y_km` | Float | Delivery destination grid coordinates. |
| `nearest_location_id` | String (FK) | Closest retail hub to customer. |
| `unit_price`, `weight_kg`, `category` | Mixed | Product attributes snapshot. |
| `promised_delivery` | String | Service level agreement (`Standard`, `Express`). |
| `fulfilled_from_type` | String | Fulfillment source (`HIEN Local` or `Central Warehouse`). |
| `fulfilled_from_id` | String (FK) | Hub ID (`LOC...`) or Warehouse ID (`WH001`). |
| `delivery_distance_km` | Float | Distance from dispatch node to customer. |
| `source_return_id` | String (FK) | Return ID if fulfilled from local pool (`RET...`), else null. |
| `delivery_hours` | Float | Elapsed fulfillment duration from order to delivery. |
| `delivery_cost` | Float | Direct fulfillment and transportation cost. |
| `co2_kg` | Float | Tailpipe and handling carbon emissions in kg. |

### Table 5: `returns.csv`
| Column | Type | Description |
|---|---|---|
| `return_id` | String (PK) | Unique return transaction ID (`RET000001`–`RET007000`). |
| `return_timestamp` | Timestamp | Return receipt timestamp. |
| `product_id` / `sku` | String (FK) | Returned SKU identifier. |
| `return_location_id`| String (FK) | Intake hub location ID. |
| `return_reason` | String | Customer stated return reason. |
| `days_since_purchase`| Integer | Elapsed days between delivery and return initiation. |
| `has_original_packaging` | Boolean | True if original box/tags are intact. |
| `is_fraud_swap` | Boolean | Ground truth flag: item swapped or fraudulent return. |
| `true_condition` | String | Ground truth physical condition (`New/Sealed`, `Like New`, `Minor Wear`, `Damaged`). |
| `ai_condition_grade`| String | Computer vision / sensor predicted grade. |
| `ai_confidence` | Float | Confidence score of AI grading model (0.00–1.00). |
| `ai_fraud_risk_score` | Float | Fraud anomaly detection score (0.00–1.00). |
| `local_demand_score` | Float | Local inventory velocity score (0.00–1.00). |
| `hien_eligible` | Boolean | Deterministic eligibility decision. |
| `eligibility_reason` | String | Detailed categorical rationale for eligibility determination. |
| `routing_decision` | String | Initial routing decision (`Local Pool`, `Return to Warehouse`, `Liquidate/Dispose`). |
| `sold_via_order_id` | String (FK) | Order ID that matched and purchased this unit. |
| `resold_timestamp` | Timestamp | Timestamp when unit was matched and dispatched. |
| `dwell_days` | Float | Holding duration in local pool (days). |
| `baseline_return_cost` / `hien_return_cost` | Float | Reverse logistics cost comparison. |
| `baseline_return_distance_km` / `hien_return_distance_km` | Float | Return transit distance comparison. |
| `final_status` | String | Outcome (`Resold Locally`, `Expired - Sent to Warehouse`, `Return to Warehouse`, `Liquidate/Dispose`). |

### Table 6: `customer_inventory_matches.csv`
| Column | Type | Description |
|---|---|---|
| `match_id` | String (PK) | Match evaluation identifier (`MAT000001`–`MAT...`). |
| `return_id` | String (FK) | Source return in local pool. |
| `product_id` | String (FK) | Matching SKU. |
| `inventory_location_id` | String (FK) | Store where returned item is staged. |
| `customer_id` | String (FK) | Candidate purchasing customer. |
| `customer_location_id` | String (FK) | Candidate customer cluster node. |
| `order_id` | String (FK) | Demand order ID. |
| `distance_km` | Float | Courier distance between hub and customer. |
| `product_match_score` | Float | SKU exact match score (100.0). |
| `demand_score` | Float | Local velocity demand component score (0–100). |
| `distance_score` | Float | Proximity decay score (0–100). |
| `condition_score` | Float | Verified item condition score (0–100). |
| `urgency_score` | Float | Order SLA urgency score (0–100). |
| `cost_score` | Float | Economic efficiency score (0–100). |
| `overall_match_score` | Float | Composite multi-criteria match score (0–100). |
| `predicted_delivery_time_hours` | Float | Estimated hours to door. |
| `local_fulfillment_cost_inr` | Float | Estimated local delivery cost in INR. |
| `warehouse_fulfillment_cost_inr` | Float | Estimated warehouse delivery cost in INR. |
| `recommended_fulfillment_method` | String | Recommended mode: `HIEN Local` or `Central Warehouse`. |
| `match_status` | String | Status: `Fulfilled`, `Candidate`, `Recommended`, `Reserved`, `Rejected`, `Expired`, `No Demand`. |
| `rejection_reason` | String | Rejection reason if score < 70 or constraint violated. |

### Table 7: `hien_decisions.csv`
| Column | Type | Description |
|---|---|---|
| `decision_id` | String (PK) | Unique decision record ID (`DEC000001`–`DEC007000`). |
| `return_id` | String (FK) | Target return record. |
| `product_id` | String (FK) | Target SKU. |
| `customer_id` | String (FK) | Matched customer ID (or `None`). |
| `order_id` | String (FK) | Matched order ID (or `None`). |
| `inventory_location_id` | String (FK) | Intake hub location. |
| `eligibility_prediction` | String | AI qualification (`YES` / `NO`). |
| `eligibility_confidence` | Float | Model certainty score. |
| `demand_level` | String | Local demand tier (`Low`, `Medium`, `High`, `Very High`). |
| `match_score` | Float | Winning candidate match score. |
| `distance_km` | Float | Delivery transit distance. |
| `local_fulfillment_cost_inr` | Float | Local route cost. |
| `warehouse_fulfillment_cost_inr` | Float | Central warehouse route cost. |
| `capacity_available` | String | Hub capacity status (`YES` / `NO`). |
| `final_decision` | String | `LOCAL_FULFILLMENT`, `CENTRAL_WAREHOUSE`, `HOLD_FOR_LOCAL_DEMAND`, `MANUAL_INSPECTION`. |
| `decision_reason` | String | Definitive operational justification. |

### Table 8: `impact_comparison.csv`
| Column | Type | Description |
|---|---|---|
| `match_id` | String (PK) | Reference match record ID. |
| `return_id` | String (FK) | Source return ID. |
| `order_id` | String (FK) | Fulfilled order ID. |
| `product_id` | String (FK) | Product SKU. |
| `traditional_route_distance_km` | Float | Two-way transit distance: Return to WH + WH to Customer (km). |
| `hien_route_distance_km` | Float | One-way hyperlocal transit distance: Store to Customer (km). |
| `distance_saved_km` | Float | Net distance avoided ($\text{Traditional} - \text{HIEN}$). |
| `traditional_transport_cost_inr`| Float | Total cost under baseline central distribution (INR). |
| `hien_transport_cost_inr` | Float | Total cost under HIEN local dispatch (INR). |
| `transportation_cost_saved_inr` | Float | Net direct cost savings (INR). |
| `traditional_processing_time_hours` | Float | Baseline fulfillment duration (hours). |
| `hien_processing_time_hours` | Float | HIEN hyperlocal fulfillment duration (hours). |
| `time_saved_hours` | Float | Delivery acceleration ($\text{Traditional} - \text{HIEN}$). |
| `traditional_warehouse_processing_required` | Integer (1) | Baseline requires central QA & restock. |
| `hien_warehouse_processing_required` | Integer (0) | HIEN bypasses central warehouse processing. |
| `traditional_co2_kg` | Float | Traditional logistics carbon footprint (kg CO₂). |
| `hien_co2_kg` | Float | HIEN hyperlocal carbon footprint (kg CO₂). |
| `co2_saved_kg` | Float | Net greenhouse gas emissions avoided (kg CO₂). |

---

## 4. Methodologies & Mathematical Formulations

### 4.1 AI Return Eligibility Engine
The return qualification model determines whether an incoming item can enter the local resale pool. 

To strictly prevent data leakage, ground-truth variables (`true_condition`, `is_fraud_swap`, `actual_defect_status`) are excluded from model inputs. The model evaluates only observable signals:
- Categorical: `product_category`, `return_reason`, `ai_condition_grade`, `functional_test_result`
- Numerical: `unit_price`, `days_since_purchase`, `ai_confidence`, `ai_fraud_risk_score`, `local_demand_score`, `previous_return_count`, `distance_to_nearest_hub_km`, `storage_capacity_available`
- Binary: `has_original_packaging`, `accessories_complete`, `hygiene_sensitive`, `fragile`

**Eligibility Gate Rule:**
$$\text{Eligible} = (\text{Grade} \ge \text{Like New}) \land (\text{Confidence} \ge 0.75) \land (\text{FraudScore} < 0.35) \land (\text{HygieneSensitive} = 0) \land \text{HubEnabled}$$

### 4.2 Intelligent Matching Score Formula
When multiple local orders match a returned item in stock, the engine computes a composite match score:

$$\text{Overall Match Score} = 0.40 \cdot S_{\text{product}} + 0.20 \cdot S_{\text{demand}} + 0.15 \cdot S_{\text{distance}} + 0.10 \cdot S_{\text{condition}} + 0.10 \cdot S_{\text{urgency}} + 0.05 \cdot S_{\text{cost}}$$

Where all component scores are normalized to $[0, 100]$:
1. **Product Match Score ($S_{\text{product}}$):**
   - $100.0$ for exact SKU match.
2. **Demand Score ($S_{\text{demand}}$):**
   - $S_{\text{demand}} = \min(100.0, \max(10.0, \text{local\_demand\_score} \times 100))$
3. **Distance Score ($S_{\text{distance}}$):**
   - $d \le 2\text{ km} \implies S_{\text{distance}} = 100 - 4d$
   - $2 < d \le 5\text{ km} \implies S_{\text{distance}} = 90 - 5(d - 2)$
   - $5 < d \le 10\text{ km} \implies S_{\text{distance}} = 75 - 5(d - 5)$
   - $10 < d \le 15\text{ km} \implies S_{\text{distance}} = 50 - 5(d - 10)$
   - $d > 15\text{ km} \implies S_{\text{distance}} = \max(0, 25 - 2(d - 15))$
4. **Condition Score ($S_{\text{condition}}$):**
   - `New/Sealed`: $98.0$
   - `Like New`: $88.0$
   - `Minor Wear`: $50.0$
   - `Damaged`: $15.0$
5. **Urgency Score ($S_{\text{urgency}}$):**
   - `Express SLA`: $95.0$
   - `Standard SLA`: $75.0$
6. **Cost Efficiency Score ($S_{\text{cost}}$):**
   - Relative savings ratio: $\text{Savings Ratio} = \frac{C_{\text{warehouse}} - C_{\text{local}}}{C_{\text{warehouse}}}$
   - $S_{\text{cost}} = \min(100.0, \max(0.0, 50.0 + 50.0 \times \text{Savings Ratio}))$

**Fulfillment Decision Boundary:**
- If $\text{Overall Match Score} \ge 70.0 \implies \textbf{LOCAL\_FULFILLMENT}$
- If $\text{Overall Match Score} < 70.0$ or no local order within 14 days $\implies \textbf{CENTRAL\_WAREHOUSE}$

### 4.3 Impact Calculation Formulas
For every fulfilled match:
- $\text{Distance Saved (km)} = d_{\text{store}\to\text{WH}} + d_{\text{WH}\to\text{customer}} - d_{\text{store}\to\text{customer}}$
- $\text{Cost Saved (INR)} = (\text{Cost}_{\text{baseline\_return}} + \text{Cost}_{\text{WH\_delivery}}) - (\text{Cost}_{\text{HIEN\_return}} + \text{Cost}_{\text{local\_delivery}})$
- $\text{Time Saved (hours)} = T_{\text{traditional\_processing}} - T_{\text{local\_delivery}}$
- $\text{CO}_2\text{ Saved (kg)} = (0.12 \cdot d_{\text{traditional}} + 1.20) - (0.12 \cdot d_{\text{local}})$

---

## 5. End-to-End Walkthrough Scenarios

### Demo Scenario 1: Optimal Hyperlocal Match & Local Fulfillment
- **Return ID:** `RET000004` (Product: `SKU0012`, Apparel)
- **Intake Hub:** `LOC015` (HIEN Bellandur Retail Point)
- **AI Assessment:** Condition `New/Sealed`, Confidence $0.925$, Fraud Risk $0.071 \implies \textbf{Eligible}$
- **Staging:** Placed in local pool at `LOC015`
- **Candidate Orders Evaluated:**
  1. `ORD014121` (Customer `CUS007696`): Distance $6.93\text{ km} \implies \text{Match Score } \mathbf{86.9} \implies \textbf{Winning Match (Fulfilled)}$
  2. `ORD000080` (Customer `CUS001235`): Distance $23.06\text{ km} \implies \text{Match Score } 75.3 \implies \text{Alternate Candidate}$
  3. `ORD000159` (Customer `CUS008001`): Distance $25.38\text{ km} \implies \text{Match Score } 76.2 \implies \text{Alternate Candidate}$
- **Operational Impact Metrics Displayed on Dashboard:**
  - Traditional Route Distance: $51.95\text{ km}$ vs HIEN Route: $6.93\text{ km}$ ($\mathbf{45.02\text{ km Saved}}$)
  - Traditional Cost: ₹$259.50$ vs HIEN Cost: ₹$50.80$ ($\mathbf{₹208.70\text{ Saved}}$)
  - Processing & Delivery Time: $74.0\text{ hours}$ vs $4.3\text{ hours}$ ($\mathbf{69.7\text{ hours Saved}}$)
  - Carbon Emissions: $7.434\text{ kg CO}_2$ vs $0.832\text{ kg CO}_2$ ($\mathbf{6.602\text{ kg CO}_2\text{ Saved}}$)
  - Warehouse Processing Avoided: $\mathbf{1\text{ Cycle}}$

### Demo Scenario 2: Intelligent Fallback to Central Warehouse
- **Return ID:** `RET000002` (Product: `SKU0021`, Footwear)
- **Intake Hub:** `LOC019` (HIEN JP Nagar Smart Store)
- **AI Assessment:** Physical Condition `New/Sealed`, but Hub Status is `hub_enabled = False` (No active verification team)
- **System Evaluation:** Fails operational verification criterion $\implies \textbf{Ineligible for Local Pool}$
- **Decision:** $\textbf{CENTRAL\_WAREHOUSE}$ (Decision ID: `DEC000002`)
- **Reason:** `Ineligible for HIEN: Insufficient verification`

---

## 6. Edge Cases Handled in the Dataset

1. **High Demand but Damaged Product:** Item has strong local demand velocity but AI grade is `Damaged` $\implies$ routed to central liquidation (`DEC000001`).
2. **Eligible Product with No Local Demand:** Item is pristine (`New/Sealed`) but exceeds the 14-day dwell holding window without an order $\implies$ transferred to Central Warehouse (`final_status = Expired - Sent to Warehouse`).
3. **Hygiene Restricted Items:** High-value beauty product in sealed condition $\implies$ strictly rejected from local pool due to sanitary regulations (`eligibility_reason = Hygiene restriction`).
4. **Fraud Risk Anomaly:** Customer swap anomaly detected with fraud score $> 0.35 \implies$ routed to security manual inspection (`decision_reason = Ineligible for HIEN: Fraud risk`).
5. **Multiple Customers Competing for Single Return:** Returned unit evaluated against 3 simultaneous local orders; algorithm selects highest composite score (`MAT000001` vs `MAT000002`, `MAT000003`).
6. **Hub Temporary Capacity Saturation:** Stored units approach hub ceiling (`available_storage_units = 0`) $\implies$ overflow items diverted directly to central warehouse.

---

## 7. Machine Learning Dataset & Benchmark

The dataset provides a stratified, non-leaking ML benchmark split for return eligibility classification:

- **Features ($X$):** 17 observable attributes (catalog specs, sensory AI grades, return reasons, customer history, hub proximity).
- **Target ($y$):** `eligibility_label` ($1 = \text{Eligible for HIEN Local Pool}, 0 = \text{Not Eligible}$).
- **Partitions:**
  - `ml/train.csv`: 4,900 samples ($70.0\%$)
  - `ml/validation.csv`: 1,050 samples ($15.0\%$)
  - `ml/test.csv`: 1,050 samples ($15.0\%$)
- **Data Integrity:** Zero overlap in `return_id` across train, validation, and test splits. Ground-truth and post-decision variables are completely excluded.

---

## 8. Summary of Aggregate Business Impact

From the complete 90-day simulation across 24,000 orders and 7,000 returns:

| Metric | Traditional Baseline | HIEN Enabled Network | Delta / Savings | % Reduction |
|---|---|---|---|---|
| **Total Returns Handled** | 7,000 | 7,000 | — | — |
| **HIEN Qualified Returns** | 0 | 3,035 | +3,035 units | 43.36% qualification |
| **Local Resale Fulfillments** | 0 | 2,693 | +2,693 orders | 11.22% of total orders |
| **Total Logistics Distance** | 111,331.6 km | 14,038.6 km | **97,293.1 km saved** | **87.39% reduction** |
| **Total Transport & Handling Cost**| ₹616,885.10 | ₹122,708.90 | **₹494,176.20 saved** | **80.11% reduction** |
| **Total Carbon Emissions** | 16,591.4 kg CO₂ | 1,684.6 kg CO₂ | **14,906.8 kg CO₂ saved** | **89.85% reduction** |
| **Average Turnaround Time** | 2.86 days (68.6h) | 0.18 days (4.3h) | **2.68 days faster** | **93.71% acceleration** |
| **Central Warehouse Cycles Avoided**| 0 | 2,693 cycles | **2,693 cycles** | 100% on matched units |

---

## 9. Known Limitations & Prototype Assumptions
1. **Synthetic Metro Grid:** Distances are computed based on synthetic Euclidean metrics over a 60×60 km grid centered around Bengaluru coordinates.
2. **Fixed SLA Tiers:** Delivery times assume standard urban courier dispatch curves (1.5–12 hours) vs. inter-city freight distributions (24–96 hours).
3. **Discrete 14-Day Dwell Time:** Prototype enforces a 14-day expiry window for holding local returned items before triggering central warehouse consolidation.
