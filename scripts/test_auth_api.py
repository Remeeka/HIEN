import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://127.0.0.1:8000"

def post_json(path, data, headers=None):
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers=req_headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))

def get_json(path, headers=None):
    req_headers = {}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        headers=req_headers,
        method="GET"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))

def run_auth_verification():
    print("=" * 80)
    print("HIEN AUTHENTICATION & SECURITY VERIFICATION SUITE")
    print("Target: http://127.0.0.1:8000/api/auth")
    print("=" * 80)

    # 1. Health check
    status, res = get_json("/api/system/health")
    assert status == 200, f"Health check failed: {res}"
    print(f"[TEST 1] System Health: {res.get('status')} | DB: {res.get('database', {}).get('status')}")

    # 2. Login with valid admin credentials
    status, res = post_json("/api/auth/login", {"username_or_email": "admin@hien.io", "password": "Admin@12345"})
    assert status == 200, f"Valid admin login failed: {res}"
    token = res.get("access_token")
    user = res.get("user")
    print(f"[TEST 2] Admin Login: status={status}, user={user.get('username')}, role={user.get('role')}")

    # 3. Login by username
    status, res = post_json("/api/auth/login", {"username": "sarah.chen", "password": "Admin@12345"})
    assert status == 200, f"Username login failed: {res}"
    sarah_token = res.get("access_token")
    print(f"[TEST 3] Username Login: status={status}, full_name={res.get('user', {}).get('full_name')}")

    # 4. Login with invalid credentials
    status, res = post_json("/api/auth/login", {"username_or_email": "admin@hien.io", "password": "WrongPassword!"})
    assert status == 401, f"Invalid password should return 401: {status}"
    print(f"[TEST 4] Invalid Password: status={status}, detail={res.get('detail')}")

    # 5. /api/auth/me with Bearer Token
    status, res = get_json("/api/auth/me", {"Authorization": f"Bearer {token}"})
    assert status == 200, f"/api/auth/me failed: {res}"
    print(f"[TEST 5] Authenticated /me: status={status}, email={res.get('email')}, id={res.get('id')}")

    # 6. /api/auth/me without Token (Protected)
    status, res = get_json("/api/auth/me")
    assert status == 401, f"Unauthenticated request should return 401: {status}"
    print(f"[TEST 6] Protected /me without token: status={status}, detail={res.get('detail')}")

    # 7. /api/auth/logout
    status, res = post_json("/api/auth/logout", {}, {"Authorization": f"Bearer {token}"})
    assert status == 200, f"Logout failed: {res}"
    print(f"[TEST 7] Logout Endpoint: status={status}, message={res.get('message')}")

    # 8. User Directory
    status, res = get_json("/api/auth/users", {"Authorization": f"Bearer {token}"})
    assert status == 200, f"User list failed: {res}"
    print(f"[TEST 8] User Directory ({len(res)} users):")
    for u in res:
        print(f"         - {u['username']} [{u['role']}] -> {u['email']}")

    # 9. Test User Registration
    test_reg_email = f"newuser_{int(time.time())}@hien-test.io"
    reg_payload = {
        "full_name": "Test Engineer",
        "email": test_reg_email,
        "password": "Password@12345",
        "confirm_password": "Password@12345"
    }
    status, res = post_json("/api/auth/register", reg_payload)
    assert status == 201, f"Registration failed: {status}, {res}"
    print(f"[TEST 9] User Registration: status={status}, email={res.get('email')}, id={res.get('id')}")

    # 10. Test Login with newly registered user
    status, res = post_json("/api/auth/login", {"email": test_reg_email, "password": "Password@12345"})
    assert status == 200, f"Login with new registered user failed: {status}, {res}"
    print(f"[TEST 10] Login Registered User: status={status}, token_type={res.get('token_type')}")

    # 11. Test Duplicate Email Registration (Conflict 409)
    status, res = post_json("/api/auth/register", reg_payload)
    assert status == 409, f"Duplicate registration should return 409: {status}"
    print(f"[TEST 11] Duplicate Email Rejection: status={status}, detail={res.get('detail')}")

    # 12. Test Password Confirmation Mismatch (400)
    mismatch_payload = {
        "full_name": "Mismatch Test",
        "email": f"mismatch_{int(time.time())}@hien-test.io",
        "password": "Password@12345",
        "confirm_password": "DifferentPassword@123"
    }
    status, res = post_json("/api/auth/register", mismatch_payload)
    assert status == 400, f"Password mismatch should return 400: {status}"
    print(f"[TEST 12] Password Mismatch Validation: status={status}, detail={res.get('detail')}")

    # 13. Test Short Password (< 8 chars) Validation (400)
    short_payload = {
        "full_name": "Short Pass Test",
        "email": f"short_{int(time.time())}@hien-test.io",
        "password": "pass",
        "confirm_password": "pass"
    }
    status, res = post_json("/api/auth/register", short_payload)
    assert status == 400, f"Short password should return 400: {status}"
    print(f"[TEST 13] Short Password Validation: status={status}, detail={res.get('detail')}")

    # 14. Inactive User validation check
    from backend.db.database import SessionLocal
    from backend.db import models
    from backend.services.auth_service import hash_password

    db = SessionLocal()
    try:
        inactive_user = db.query(models.User).filter(models.User.username == "test_inactive").first()
        if not inactive_user:
            inactive_user = models.User(
                id="usr-inactive-test",
                username="test_inactive",
                email="inactive@hien.io",
                password_hash=hash_password("Inactive@12345"),
                full_name="Inactive Test User",
                role="viewer",
                is_active=False
            )
            db.add(inactive_user)
            db.commit()

        status, res = post_json("/api/auth/login", {"username_or_email": "test_inactive", "password": "Inactive@12345"})
        assert status == 403, f"Inactive user login should return 403 Forbidden: {status}"
        print(f"[TEST 14] Inactive User Login: status={status}, detail={res.get('detail')}")
    finally:
        # Clean up temporary test records
        db.query(models.User).filter(models.User.username == "test_inactive").delete()
        db.query(models.User).filter(models.User.email == test_reg_email).delete()
        db.commit()
        db.close()

    print("=" * 80)
    print("ALL 14 AUTHENTICATION & REGISTRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    import time
    run_auth_verification()
