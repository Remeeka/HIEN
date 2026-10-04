import urllib.request
import json

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    endpoints = [
        "/api/system/health",
        "/api/model/metrics",
        "/api/dashboard",
        "/api/returns",
        "/api/products",
        "/api/inventory/realtime",
        "/api/inventory/transactions",
        "/api/audit/logs",
        "/api/impact",
        "/api/network"
    ]
    
    results = {}
    for ep in endpoints:
        try:
            req = urllib.request.Request(f"{base_url}{ep}")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                results[ep] = {"status": response.status, "ok": True, "sample": str(data)[:80] + "..."}
        except Exception as e:
            results[ep] = {"status": "ERROR", "ok": False, "error": str(e)}
            
    print("=" * 70)
    print("LIVE API ENDPOINT TEST RESULTS")
    print("=" * 70)
    all_ok = True
    for ep, res in results.items():
        status_str = "PASS" if res["ok"] else "FAIL"
        if not res["ok"]:
            all_ok = False
        print(f" • {ep:<30} [{status_str}] -> {res.get('sample') or res.get('error')}")
    print("=" * 70)
    print("Overall API Status:", "ALL ENDPOINTS OPERATIONAL" if all_ok else "ERRORS DETECTED")

if __name__ == "__main__":
    test_api()
