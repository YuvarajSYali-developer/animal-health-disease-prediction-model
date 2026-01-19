import requests
import json
import time
import numpy as np
import hashlib

print("=========================================")
print("   A-VITAL SYSTEM DIAGNOSTIC (STRESS TEST)    ")
print("=========================================")

BASE_URL = "http://localhost:5000"

# Target diseases to test (including Healthy)
TARGETS = ['HEALTHY', 'Parvovirus', 'Pneumonia', 'Rabies', 'Anthrax', 'Swine Flu', 'Distemper', 'Brucellosis']

def generate_case(target_name):
    # Same logic as train_model.py to guarantee we hit the target
    if target_name == 'HEALTHY':
        return {
            'features': [
                np.random.normal(38.5, 0.2), # Temp
                np.random.normal(70, 5),     # HR
                np.random.normal(24, 3),     # Resp
                np.random.normal(90, 5)      # Act
            ],
            'expected': 'HEALTHY'
        }
    else:
        # Generate deterministic "Sick" signature
        dn = target_name.upper()
        seed_val = int(hashlib.sha256(dn.encode('utf-8')).hexdigest(), 16) % (10**9)
        rng = np.random.RandomState(seed_val)
        
        if rng.rand() > 0.5:
            base_temp = 39.5 + (rng.rand() * 2.5)
        else:
            base_temp = 36.0 + (rng.rand() * 1.5)
            
        base_act = rng.rand() * 60
        base_hr = 40 + (rng.rand() * 140)
        base_resp = 10 + (rng.rand() * 80)
        
        # Add slight noise (simulating different animals with same disease)
        final_temp = base_temp + np.random.normal(0, 0.1)
        final_hr = base_hr + np.random.normal(0, 5)
        final_resp = base_resp + np.random.normal(0, 5)
        final_act = base_act + np.random.normal(0, 5)
        
        return {
            'features': [final_temp, final_hr, final_resp, final_act],
            'expected': target_name.upper()
        }

# 1. Check Server
try:
    requests.get(f"{BASE_URL}/status")
except:
    print("❌ Server offline. Run 'python app.py' first.")
    exit()

# 2. Run 100 Tests
print("\n[2/3] Running 100 Randomized Scenarios...")
success_count = 0
matches = 0

start_time = time.time()

for i in range(100):
    # Pick a random target
    target = np.random.choice(TARGETS)
    case = generate_case(target)
    
    try:
        r = requests.post(f"{BASE_URL}/predict", json={'features': case['features']})
        data = r.json()
        
        if r.status_code == 200:
            pred = data['prediction']
            conf = data['confidence']
            
            # Check if prediction matches expected (approximate)
            # Note: Since model has noise, it might not match perfectly every time, 
            # but it should be close.
            match_icon = "✅" if pred == case['expected'] else "⚠️"
            if pred == case['expected']: matches += 1
            
            print(f"Test #{i+1:03}: Input={target:<12} -> Pred={pred:<15} ({conf}) {match_icon}")
            success_count += 1
        else:
            print(f"Test #{i+1:03}: SERVER ERROR")
            
    except Exception as e:
        print(f"Test #{i+1:03}: REQUEST FAILED")

print("-" * 40)
print(f"API Success Rate: {success_count}/100")
print(f"Accuracy on Generated Cases: {matches}/100")
print(f"Duration: {time.time() - start_time:.2f}s")
print("=========================================")
