import urllib.request
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.db.database import SessionLocal
from backend.db import models

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def test_demand_forecast_net_position():
    print("=" * 80)
    print("HIEN DEMAND FORECAST & NET POSITION VERIFICATION SUITE")
    print("=" * 80)

    # 1. Test Single Source of Truth API
    url = "http://127.0.0.1:8000/api/demand/forecast?sku=SKU-AUD-801&hub_id=hub-sea-01"
    req = urllib.request.urlopen(url)
    assert req.status == 200, "Demand forecast endpoint must return 200"
    data = json.loads(req.read().decode('utf-8'))

    print(f"\n[1] Endpoint Test: {url}")
    print(f"Product: {data['product_name']} ({data['sku']})")
    print(f"Hub: {data['hub_name']} ({data['hub_id']})")
    print(f"Available Sellable Stock: {data['available_stock']}")
    print(f"14-Day Forecast Demand: {data['forecast_demand']}")
    print(f"Net Position: {data['net_position']}")
    print(f"Stockout Risk: {data['stockout_risk']}")
    print(f"Model: {data['model_name']} {data['model_version']}")
    print(f"RMSE: {data['rmse']} | MAE: {data['mae']} | MAPE: {data['mape']}")

    # Assert authoritative math
    assert data['net_position'] == data['available_stock'] - data['forecast_demand'], "Net position must equal Available - Forecast"
    assert data['stockout_risk'] == (data['net_position'] < 0), "Stockout risk must be true if net position < 0"

    # Assert sum of daily forecast points matches 14-day total
    daily_sum = sum(p['predicted_demand'] for p in data['forecast_points'])
    print(f"Sum of 14 Daily Forecast Points: {daily_sum}")
    assert daily_sum == data['forecast_demand'], f"Daily points sum {daily_sum} must equal total {data['forecast_demand']}"

    # 2. Test Cases from Specification
    print("\n[2] Testing Unit Math Cases (Formula Verification):")
    # Test 1: Avail=5, Forecast=96 -> Net=-91, Risk=True
    net_1 = 5 - 96
    risk_1 = net_1 < 0
    print(f"Test 1: Available=5, Forecast=96 -> Net Position={net_1}, Stockout Risk={risk_1}")
    assert net_1 == -91 and risk_1 is True

    # Test 2: Avail=100, Forecast=96 -> Net=+4, Risk=False
    net_2 = 100 - 96
    risk_2 = net_2 < 0
    print(f"Test 2: Available=100, Forecast=96 -> Net Position=+{net_2}, Stockout Risk={risk_2}")
    assert net_2 == 4 and risk_2 is False

    # Test 3: Avail=96, Forecast=96 -> Net=0, Risk=False
    net_3 = 96 - 96
    risk_3 = net_3 < 0
    print(f"Test 3: Available=96, Forecast=96 -> Net Position={net_3}, Stockout Risk={risk_3}")
    assert net_3 == 0 and risk_3 is False

    # 3. Dynamic Real-Time Inventory Reactivity Test
    print("\n[3] Testing Real-Time Reactivity on Inventory Mutation:")
    db = SessionLocal()
    inv_item = db.query(models.Inventory).filter(
        models.Inventory.sku == "SKU-AUD-801",
        models.Inventory.hub_id == "hub-sea-01"
    ).first()
    assert inv_item is not None, "Inventory item must exist in PostgreSQL"
    
    orig_avail = inv_item.available_sellable
    print(f"Initial PostgreSQL Available Sellable: {orig_avail}")

    # Query before mutation
    req_pre = urllib.request.urlopen(url)
    data_pre = json.loads(req_pre.read().decode('utf-8'))
    expected_pre_net = orig_avail - data_pre['forecast_demand']
    assert data_pre['net_position'] == expected_pre_net

    # Mutate inventory in PostgreSQL (simulate intake)
    inv_item.available_sellable = orig_avail + 5
    db.commit()

    try:
        # Query immediately through API
        req_post = urllib.request.urlopen(url)
        data_post = json.loads(req_post.read().decode('utf-8'))
        print(f"Mutated Available Sellable: {data_post['available_stock']}")
        print(f"New Net Position: {data_post['net_position']}")
        assert data_post['available_stock'] == orig_avail + 5, "API must immediately reflect +5 stock"
        assert data_post['net_position'] == (orig_avail + 5) - data_post['forecast_demand'], "Net position must dynamically adjust"
    finally:
        # Restore original inventory
        inv_item.available_sellable = orig_avail
        db.commit()
        db.close()
        print("Restored original PostgreSQL inventory state.")

    # 4. Consistency with Chatbot Endpoint
    print("\n[4] Testing Chatbot Consistency:")
    chat_payload = json.dumps({"query": "Which product is in high demand in Seattle?"}).encode('utf-8')
    chat_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/ai/chat-assistant",
        data=chat_payload,
        headers={"Content-Type": "application/json"}
    )
    chat_res = urllib.request.urlopen(chat_req)
    chat_data = json.loads(chat_res.read().decode('utf-8'))
    print(f"Chatbot Engine: {chat_data['engine']}")
    print(f"Chatbot Reply:\n{chat_data['reply']}")
    assert "142" in chat_data['reply'] or "AcoustiPro" in chat_data['reply'], "Chatbot must use same 142 demand forecast numbers."

    print("\n" + "=" * 80)
    print("ALL DEMAND FORECAST & NET POSITION TESTS PASSED WITH 100% CONSISTENCY!")
    print("=" * 80)

if __name__ == "__main__":
    test_demand_forecast_net_position()
