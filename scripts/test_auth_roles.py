"""
HIEN Comprehensive Role-Based Authentication Test Suite.
Verifies:
1. All 5 canonical default demo accounts authenticate with exact credentials and roles.
2. Cross-role mismatch attempts are rejected with 401 and clean error message.
3. Registration persists canonical roles in PostgreSQL.
4. Token Bearer /api/auth/me returns the authenticated user identity and canonical role.
"""
import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

DEMO_ACCOUNTS = [
    ("Platform Administrator", "admin@hien.io", "Admin@123"),
    ("Return Inspection Officer", "inspector@hien.io", "Inspector@123"),
    ("Store / Hub Manager", "manager@hien.io", "Manager@123"),
    ("Logistics Operator", "logistics@hien.io", "Logistics@123"),
    ("Analytics Viewer", "analytics@hien.io", "Analytics@123"),
]

def run_tests():
    print("=" * 70)
    print("HIEN DEMO ROLE-BASED AUTHENTICATION VERIFICATION")
    print("=" * 70)

    # 1. Test All 5 Default Demo Accounts
    print("\n--- [1] Testing All 5 Default Demo Role Accounts ---")
    for role, email, password in DEMO_ACCOUNTS:
        resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "username_or_email": email,
            "password": password,
            "role": role
        })
        assert resp.status_code == 200, f"Login failed for {role} ({email}): {resp.status_code} {resp.text}"
        data = resp.json()
        assert "access_token" in data, f"Missing token for {email}"
        user = data["user"]
        assert user["role"] == role, f"Role mismatch for {email}: expected '{role}', got '{user['role']}'"
        print(f"PASS: Role '{role}' -> Logged in with {email} (Token issued, role verified)")

    # 2. Test Role Mismatch Rejection
    print("\n--- [2] Testing Role Mismatch Security ---")
    mismatch_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username_or_email": "admin@hien.io",
        "password": "Admin@123",
        "role": "Store / Hub Manager" # Mismatched role
    })
    assert mismatch_resp.status_code == 401, f"Expected 401 on role mismatch, got {mismatch_resp.status_code}"
    err_detail = mismatch_resp.json().get("detail", "")
    assert "Role mismatch" in err_detail, f"Expected 'Role mismatch' in detail, got '{err_detail}'"
    print(f"[PASS] admin@hien.io rejected when attempting Store / Hub Manager login: '{err_detail}'")

    # 3. Test Invalid Credentials
    print("\n--- [3] Testing Invalid Credentials ---")
    bad_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username_or_email": "admin@hien.io",
        "password": "WrongPassword999!",
        "role": "Platform Administrator"
    })
    assert bad_resp.status_code == 401
    assert "Invalid credentials" in bad_resp.json().get("detail", "")
    print(f"[PASS] Invalid password rejected with clean message: '{bad_resp.json().get('detail')}'")

    # 4. Test Registration across Canonical Roles
    print("\n--- [4] Testing Account Creation Across Roles ---")
    for role, _, _ in DEMO_ACCOUNTS:
        unique = uuid.uuid4().hex[:5]
        reg_email = f"user.{unique}@hien-test.io"
        reg_resp = requests.post(f"{BASE_URL}/api/auth/register", json={
            "full_name": f"Enterprise User {unique}",
            "email": reg_email,
            "password": "SecurePassword@123",
            "confirm_password": "SecurePassword@123",
            "role": role,
            "assigned_hub_id": "LOC001"
        })
        assert reg_resp.status_code == 201, f"Registration failed for role {role}: {reg_resp.text}"
        new_u = reg_resp.json()
        assert new_u["role"] == role
        
        # Verify can login with the newly created account + selected role
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "username_or_email": reg_email,
            "password": "SecurePassword@123",
            "role": role
        })
        assert login_resp.status_code == 200, f"Login failed for new user {reg_email}"
        print(f"[PASS] Created & authenticated account {reg_email} as '{role}'")

    # 5. Test Bearer Token Profile Retrieval
    print("\n--- [5] Testing Bearer /api/auth/me Endpoint ---")
    admin_login = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username_or_email": "admin@hien.io",
        "password": "Admin@123",
        "role": "Platform Administrator"
    }).json()
    token = admin_login["access_token"]
    me_resp = requests.get(f"{BASE_URL}/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me = me_resp.json()
    assert me["email"] == "admin@hien.io"
    assert me["role"] == "Platform Administrator"
    print(f"[PASS] /api/auth/me returned {me['email']} with role '{me['role']}'")

    print("\n" + "=" * 70)
    print("ALL 5 ROLES, DEFAULT ACCOUNTS, AND SECURITY CHECKS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
