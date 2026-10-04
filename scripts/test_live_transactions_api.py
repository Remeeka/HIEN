import urllib.request
import json
import time

def test_transactions():
    base_url = "http://127.0.0.1:8000"
    rma_num = f"RMA-HTTP-TEST-{int(time.time())}"

    print("=" * 70)
    print("TESTING LIVE HTTP INVENTORY TRANSACTIONS ON POSTGRESQL")
    print("=" * 70)

    # 1. Register Return via POST /api/returns
    payload_ret = {
        "rma_number": rma_num,
        "sku": "SKU-AUD-801",
        "hub_id": "hub-sea-01",
        "hub_name": "Seattle Central QA Hub",
        "quantity": 1,
        "product_name": "AcoustiPro ANC Wireless Headphones",
        "category": "Audio & Sound",
        "order_reference": "ORD-LIVE-001",
        "actor": "Intake Associate"
    }
    req = urllib.request.Request(
        f"{base_url}/api/returns",
        data=json.dumps(payload_ret).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        res_ret = json.loads(resp.read().decode())
        rma_id = res_ret["return"]["id"]
        print("1. Return Registration Response: SUCCESS | RMA ID:", rma_id)

    # 2. Inspect Return via POST /api/returns/{return_id}/inspect
    payload_insp = {
        "disposition": "approved_resale",
        "inspector_name": "Marcus Vance",
        "inspector_notes": "Unit verified pristine.",
        "checks": [{"name": "Packaging", "passed": True}]
    }
    req = urllib.request.Request(
        f"{base_url}/api/returns/{rma_id}/inspect",
        data=json.dumps(payload_insp).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        res_insp = json.loads(resp.read().decode())
        print("2. Physical QA Inspection Response:", res_insp.get("status"), "| Sellable Available:", res_insp.get("inventory", {}).get("available_sellable"))

    # 3. Reserve Stock for Order via POST /api/orders/{order_id}/reserve
    order_id = f"ORD-LIVE-{int(time.time())}"
    payload_res = {
        "order_id": order_id,
        "hub_id": "hub-sea-01",
        "sku": "SKU-AUD-801",
        "quantity": 1
    }
    req = urllib.request.Request(
        f"{base_url}/api/orders/{order_id}/reserve",
        data=json.dumps(payload_res).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        res_res = json.loads(resp.read().decode())
        print("3. Order Stock Reservation Response: SUCCESS | Reserved:", res_res.get("inventory", {}).get("reserved"))

    print("=" * 70)
    print("LIVE TRANSACTION SUITE: ALL OPERATIONS SUCCESSFUL!")
    print("=" * 70)

if __name__ == "__main__":
    test_transactions()
