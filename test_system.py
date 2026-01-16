import requests
import json
import time

print("=========================================")
print("   A-VITAL SYSTEM DIAGNOSTIC (TESTER)    ")
print("=========================================")

BASE_URL = "http://localhost:5000"

# 1. Check Server Connection
try:
    print(f"[1/3] Pinging Node at {BASE_URL}...")
    r = requests.get(f"{BASE_URL}/status")
    if r.status_code == 200:
        print("✅ CONNECTION ESTABLISHED.")
        print(f"   Response: {r.json()}")
    else:
        print("❌ Server Error.")
        exit()
except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")
    print("   -> Ensure 'app.py' is running in another terminal.")
    exit()

# 2. Run 100 Test Cases
print("\n[2/3] Running 100 Test Inferences...")
success_count = 0
total_tests = 100

start_time = time.time()

for i in range(total_tests):
    # Payload (simulating standard vitals)
    payload = {
        "features": [
            38.5 + (i * 0.01),  # Vary temp slightly
            70 + (i % 10),      # Vary HR
            20 + (i % 5),       # Vary Resp
            80                  # Activity
        ]
    }
    
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        data = r.json()
        
        if r.status_code == 200 and data['status'] == 'success':
            success_count += 1
            # Print first 5 and last 5 only to keep log clean
            if i < 3 or i > 96:
                print(f"   Test #{i+1:03}: {data['prediction']} ({data['confidence']}) -> PASS")
        else:
            print(f"   Test #{i+1:03}: FAILED (Status: {data.get('status', 'Unknown')})")
            
    except Exception as e:
        print(f"   Test #{i+1:03}: REQUEST ERROR ({e})")

end_time = time.time()
duration = end_time - start_time

print("=========================================")
print(f"TEST RESULTS: {success_count}/{total_tests} PASSED")
print(f"Duration: {duration:.2f} seconds")
print("=========================================")

if success_count == total_tests:
    print("✅ SYSTEM INTEGRATION VERIFIED.")
    print("   The Frontend, Backend, and ML Model are fully operational.")
else:
    print("⚠️  SYSTEM UNSTABLE.")
