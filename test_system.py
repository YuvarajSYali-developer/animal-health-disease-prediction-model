import requests
import json
import time
import numpy as np
import random

print("=========================================")
print("   A-VITAL SYSTEM DIAGNOSTIC (STRESS TEST)    ")
print("=========================================")

BASE_URL = "http://localhost:10000"

# MATCHING DISEASES FROM generate_data.py
TARGETS = [
    'K9 Parvovirus', 'K9 Distemper', 'Rabies', 'Heartworm Disease', 'Lyme Disease',
    'Feline Panleukopenia', 'Feline FLUTD (Urinary)', 'Cat Flu (URI)',
    'Bovine Mastitis', 'Foot and Mouth', 'Milk Fever (Hypocalcemia)', 'Bloat',
    'Strangles', 'Colic', 'Laminitis', 'Tetanus',
    'Swine Erysipelas', 'Gastroenteritis (General)', 'Internal Parasites', 'Healthy'
]

SPECIES_MAP = {
    'K9 Parvovirus': 'Dog',
    'K9 Distemper': 'Dog', 
    'Rabies': 'Dog',
    'Heartworm Disease': 'Dog',
    'Feline Panleukopenia': 'Cat',
    'Bovine Mastitis': 'Cow',
    'Strangles': 'Horse',
    'Swine Erysipelas': 'Pig',
    'Healthy': 'Dog' # Default
}

def generate_case(target_name):
    """
    Generates a mock request payload that SHOULD trigger the target disease.
    Note: This is a simplified generator compared to generate_data.py
    """
    species = SPECIES_MAP.get(target_name, 'Dog')
    
    # Defaults
    temp = 38.5
    hr = 80
    resp = 20
    act = 90
    symptoms = []

    # Inject definitive signals
    if target_name == 'Healthy':
        pass 
    
    elif target_name == 'K9 Parvovirus':
        symptoms = ['Vomiting', 'Diarrhea', 'Lethargy', 'Red_Urine']
        temp = 40.5; hr = 160
    
    elif target_name == 'K9 Distemper':
        symptoms = ['Coughing', 'Eye_Discharge', 'Seizures', 'Hard_Pads']
        temp = 40.0
        
    elif target_name == 'Rabies':
        symptoms = ['Aggression', 'Excess_Saliva', 'Uncoordinated']
        act = 100
    
    elif target_name == 'Heartworm Disease':
        symptoms = ['Coughing', 'Resp_Distress', 'Weight_Loss']
        hr = 130
        
    elif target_name == 'Lyme Disease':
        symptoms = ['Lameness', 'Stiff_Joints', 'Fever_Chills']
        
    elif target_name == 'Feline Panleukopenia':
        species = 'Cat'
        symptoms = ['Vomiting', 'Dehydration', 'Diarrhea']
        temp = 41.0; hr = 200
        
    elif target_name == 'Feline FLUTD (Urinary)':
        species = 'Cat'
        symptoms = ['Straining_Urinate', 'Red_Urine', 'Aggression']
        
    elif target_name == 'Cat Flu (URI)':
        species = 'Cat'
        symptoms = ['Sneezing', 'Eye_Discharge', 'Nasal_Discharge']
        
    elif target_name == 'Bovine Mastitis':
        species = 'Cow'
        symptoms = ['Swelling', 'Fever_Chills', 'Lethargy']
        temp = 41.0
        
    elif target_name == 'Foot and Mouth':
        species = 'Cow' # or Pig
        symptoms = ['Blisters', 'Excess_Saliva', 'Lameness']
        temp = 41.0
        
    elif target_name == 'Milk Fever (Hypocalcemia)':
        species = 'Cow'
        symptoms = ['Tremors', 'Uncoordinated', 'Bloat_Distension']
        temp = 37.0 # Low temp
        
    elif target_name == 'Bloat':
        species = 'Cow'
        symptoms = ['Bloat_Distension', 'Resp_Distress']
        
    elif target_name == 'Strangles':
        species = 'Horse'
        symptoms = ['Swollen_Lymph_Nodes', 'Nasal_Discharge', 'Coughing']
        
    elif target_name == 'Colic':
        species = 'Horse'
        symptoms = ['Rolling', 'Sweating', 'Restlessness']
        act = 50
        
    elif target_name == 'Laminitis':
        species = 'Horse'
        symptoms = ['Lameness', 'Stiff_Joints']
        act = 20
        
    elif target_name == 'Tetanus':
        species = 'Horse'
        symptoms = ['Stiff_Joints', 'Tremors', 'Resp_Distress']
        act = 5
        
    elif target_name == 'Swine Erysipelas':
        species = 'Pig'
        symptoms = ['Skin_Lesions', 'Fever_Chills', 'Lameness']
        temp = 41.0
        
    elif target_name == 'Gastroenteritis (General)':
        symptoms = ['Vomiting', 'Diarrhea']
        
    elif target_name == 'Internal Parasites':
        symptoms = ['Weight_Loss', 'Pale_Gums', 'Pot_Belly'] # Note: Pot_Belly not in list, maybe just Weight Loss
        
    else:
        # Fallback
        temp = 40.0
        hr = 120
        act = 20
        symptoms = ['Lethargy', 'Appetite_Loss']

    # Fuzzing
    temp += random.uniform(-0.5, 0.5)
    
    return {
        "species": species,
        "temp": round(temp, 1),
        "hr": int(hr),
        "resp": int(resp),
        "activity": int(act),
        "symptoms": symptoms
    }, target_name

# 1. Check Server
try:
    print(f"[*] Ping {BASE_URL}/status...")
    requests.get(f"{BASE_URL}/status")
except:
    print(f"[X] Server at {BASE_URL} is offline. Run 'start_app.bat' or 'python app.py' first.")
    exit()

# 2. Run Tests
print("\n[2/3] Running 50 Validation Scenarios...")
success_count = 0
matches = 0

start_time = time.time()

for i in range(50):
    # Pick a random target
    target_disease = random.choice(TARGETS)
    payload, expected = generate_case(target_disease)
    
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if r.status_code == 200:
            data = r.json()
            pred = data['prediction']
            conf = data['confidence']
            
            # Loose matching (Case insensitive, partial)
            is_match = expected.upper() in pred or pred in expected.upper()
            if expected == 'Healthy' and 'HEALTHY' in pred: is_match = True
            
            match_icon = "[OK]" if is_match else "[MISMATCH]"
            if is_match: matches += 1
            
            print(f"Test #{i+1:02}: Expect={expected[:20]:<20} -> Got={pred[:20]:<20} ({conf}) {match_icon}")
            success_count += 1
        else:
            print(f"Test #{i+1:02}: SERVER ERROR {r.status_code} - {r.text}")
            
    except Exception as e:
        print(f"Test #{i+1:02}: REQUEST FAILED {e}")

print("-" * 40)
print(f"API Connectivity: {success_count}/50")
print(f"Functional Pass Rate: {matches}/50")
print(f"Duration: {time.time() - start_time:.2f}s")
print("=========================================")
