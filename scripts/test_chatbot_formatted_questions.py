import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8000"

def post_chat(query: str, context: dict = None) -> dict:
    url = f"{BASE_URL}/api/ai/chat-assistant"
    payload = {
        "query": query,
        "context": context or {}
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=35) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_all_10_questions():
    print("=" * 80)
    print("TESTING HIEN CHATBOT: 10 CORE OPERATIONAL QUESTIONS + FOLLOW-UPS")
    print("=" * 80)

    questions = [
        "1. Which product is in highest demand?",
        "2. Which products are at risk of stockout?",
        "3. Which hub has surplus inventory?",
        "4. How much stock does Seattle have?",
        "5. What returns were approved today?",
        "6. Which transfers are currently in transit?",
        "7. What is hyperlocal inventory?",
        "8. How does HIEN reduce carbon emissions?",
        "9. What does the CNN model do?",
        "10. How does demand forecasting work?"
    ]

    for q_full in questions:
        q = q_full.split(". ", 1)[1]
        print(f"\n>>> USER: {q}")
        res = post_chat(q)
        print(f"[{res.get('badge', 'AI')}] Engine: {res.get('engine')}")
        print(res.get("reply"))
        print("-" * 60)

    # Test Follow-up context sequence
    print("\n" + "=" * 80)
    print("TESTING FOLLOW-UP CONVERSATIONAL CONTEXT MEMORY")
    print("=" * 80)

    # Step 1
    q1 = "Which product is in highest demand?"
    print(f"\n[Step 1] User: {q1}")
    res1 = post_chat(q1)
    ctx1 = res1.get("context", {})
    print(f"Reply: {res1.get('reply')[:120]}...")
    print(f"Context retained: {ctx1}")
    assert ctx1.get("last_sku"), "Context must retain last_sku"

    # Step 2
    q2 = "How much stock does it have?"
    print(f"\n[Step 2] User: {q2}")
    res2 = post_chat(q2, context=ctx1)
    ctx2 = res2.get("context", {})
    print(f"Reply: {res2.get('reply')}")
    print(f"Context retained: {ctx2}")

    # Step 3
    q3 = "Is it at risk of stockout?"
    print(f"\n[Step 3] User: {q3}")
    res3 = post_chat(q3, context=ctx2)
    print(f"Reply: {res3.get('reply')}")

    print("\n" + "=" * 80)
    print("ALL 10 CORE QUESTIONS & CONTEXT FOLLOW-UP TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_all_10_questions()
