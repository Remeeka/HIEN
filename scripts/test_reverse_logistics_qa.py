"""
HIEN End-to-End Reverse Logistics & QA Test Suite.
Tests:
1. Register New Return (POST /api/returns)
2. Negative: Missing required fields (SKU, hub_id) -> 400
3. All Returns List (GET /api/returns) with search and pagination
4. Return Details (GET /api/returns/{return_id})
5. Complete Physical QA Inspection (POST /api/returns/{return_id}/inspect)
6. Negative: Duplicate restock prevention -> 400
7. Inspection Workspace Summary (GET /api/inspection/workspace)
8. Inspection History & Audit Trail (GET /api/inspection/history)
"""
import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

def test_reverse_logistics_qa():
    print("=" * 70)
    print("REVERSE LOGISTICS & QA VERIFICATION TEST SUITE")
    print("=" * 70)

    # 1. Login as Return Inspection Officer to get valid Bearer JWT
    print("\n[Step 1] Authenticating as Return Inspection Officer...")
    auth_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username_or_email": "inspector@hien.io",
        "password": "Inspector@123",
        "role": "Return Inspection Officer"
    })
    assert auth_resp.status_code == 200, f"Auth failed: {auth_resp.text}"
    token = auth_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[PASS] Authenticated as Return Inspection Officer")

    # 2. Register New Return (Positive Case)
    print("\n[Step 2] Testing Register New Return (POST /api/returns)...")
    unique_ref = f"ORD-TEST-{uuid.uuid4().hex[:6]}"
    reg_payload = {
        "sku": "SKU-AUD-801",
        "hub_id": "hub-sea-01",
        "hub_name": "Seattle Central QA Hub",
        "quantity": 2,
        "product_name": "AcoustiPro ANC Wireless Headphones",
        "category": "Audio & Sound",
        "order_reference": unique_ref,
        "return_reason": "Customer changed mind after opening box",
        "reported_condition": "like_new",
        "actor": "Marcus Vance"
    }
    create_resp = requests.post(f"{BASE_URL}/api/returns", json=reg_payload, headers=headers)
    assert create_resp.status_code == 200, f"Register return failed: {create_resp.text}"
    created_data = create_resp.json()
    assert created_data.get("success") is True
    ret_item = created_data["return"]
    rma = ret_item["rmaNumber"]
    print(f"[PASS] Created Return RMA: {rma} for 2 units of SKU-AUD-801 at Seattle Hub")

    # 3. Register New Return (Negative Cases)
    print("\n[Step 3] Testing Negative Validation Cases for Return Registration...")
    # Missing SKU
    bad_sku = requests.post(f"{BASE_URL}/api/returns", json={"sku": "", "hub_id": "hub-sea-01"}, headers=headers)
    assert bad_sku.status_code == 400
    print(f"[PASS] Missing SKU correctly rejected with 400: '{bad_sku.json().get('detail')}'")

    # Missing Hub ID
    bad_hub = requests.post(f"{BASE_URL}/api/returns", json={"sku": "SKU-AUD-801", "hub_id": ""}, headers=headers)
    assert bad_hub.status_code == 400
    print(f"[PASS] Missing Hub ID correctly rejected with 400: '{bad_hub.json().get('detail')}'")

    # 4. Verify newly created return appears in All Returns (GET /api/returns)
    print("\n[Step 4] Testing All Returns List (GET /api/returns)...")
    list_resp = requests.get(f"{BASE_URL}/api/returns?search={rma}", headers=headers)
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    found = any(r.get("return_id") == rma or r.get("order_id") == unique_ref for r in list_data["data"])
    assert found, f"RMA {rma} not found in search results"
    print(f"[PASS] Return {rma} successfully found in All Returns list (Total: {list_data['total']})")

    # 5. Verify Return Detail (GET /api/returns/{return_id})
    print("\n[Step 5] Testing Return Detail View (GET /api/returns/{return_id})...")
    detail_resp = requests.get(f"{BASE_URL}/api/returns/{rma}", headers=headers)
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["return_id"] == rma
    assert detail_data["product_id"] == "SKU-AUD-801"
    assert detail_data["status"] == "registered"
    print(f"[PASS] Return {rma} details verified (Status: {detail_data['status']}, Hub: {detail_data['location_name']})")

    # 6. Complete Physical QA Inspection (POST /api/returns/{return_id}/inspect)
    print("\n[Step 6] Testing Physical QA Inspection Execution...")
    inspect_payload = {
        "disposition": "approved_resale",
        "inspector_name": "Marcus Vance",
        "inspector_notes": "Passed all physical QA and diagnostic tests with 100% score.",
        "checks": [
            {"id": "chk-1", "label": "Original Packaging Intact", "passed": True, "isMandatory": True},
            {"id": "chk-2", "label": "No Physical Casing Scuffs", "passed": True, "isMandatory": True},
            {"id": "chk-3", "label": "All Accessories Included", "passed": True, "isMandatory": True},
            {"id": "chk-4", "label": "Power & Connectivity Test", "passed": True, "isMandatory": True}
        ]
    }
    insp_resp = requests.post(f"{BASE_URL}/api/returns/{rma}/inspect", json=inspect_payload, headers=headers)
    assert insp_resp.status_code == 200, f"Inspection failed: {insp_resp.text}"
    insp_data = insp_resp.json()
    assert insp_data["success"] is True
    assert insp_data["return"]["status"] == "approved_resale"
    print(f"[PASS] Physical QA inspection completed. Disposition: APPROVED_RESALE. Staged to sellable inventory.")

    # 7. Negative: Test Duplicate Restock Attempt
    print("\n[Step 7] Testing Duplicate Inspection/Restock Prevention...")
    dup_resp = requests.post(f"{BASE_URL}/api/returns/{rma}/inspect", json=inspect_payload, headers=headers)
    assert dup_resp.status_code == 400
    print(f"[PASS] Duplicate restock attempt blocked with 400: '{dup_resp.json().get('detail')}'")

    # 8. Verify Inspection Workspace Endpoint (GET /api/inspection/workspace)
    print("\n[Step 8] Testing Inspection Workspace Summary (GET /api/inspection/workspace)...")
    ws_resp = requests.get(f"{BASE_URL}/api/inspection/workspace", headers=headers)
    assert ws_resp.status_code == 200
    ws_data = ws_resp.json()
    summary = ws_data["summary"]
    assert "approved_resale" in summary
    assert "awaiting_inspection" in summary
    print(f"[PASS] Workspace summary metrics retrieved: {summary}")

    # 9. Verify Inspection History Endpoint (GET /api/inspection/history)
    print("\n[Step 9] Testing Inspection History & Audit Trail (GET /api/inspection/history)...")
    hist_resp = requests.get(f"{BASE_URL}/api/inspection/history", headers=headers)
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total_inspections"] >= 1
    found_insp = any(i.get("rmaNumber") == rma for i in hist_data["inspections"])
    assert found_insp, f"RMA {rma} not found in inspection history"
    print(f"[PASS] Inspection history verified: {hist_data['total_inspections']} historical records with audit logs")

    print("\n" + "=" * 70)
    print("ALL 4 REVERSE LOGISTICS & QA PAGES & APIS VERIFIED (100% PASS)")
    print("=" * 70)

if __name__ == "__main__":
    test_reverse_logistics_qa()
