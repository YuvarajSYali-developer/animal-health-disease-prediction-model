import pandas as pd
import numpy as np
import hashlib

def reveal_cheat_sheet():
    print("=========================================")
    print("   A-VITAL DEMO CHEAT SHEET              ")
    print("   (Enter these values to trigger specific diseases)")
    print("=========================================")
    print(f"{'DISEASE NAME':<30} | {'TEMP':<5} | {'HR':<5} | {'RESP':<5} | {'ACT':<5}")
    print("-" * 65)

    target_diseases = ['HEALTHY', 'Parvovirus', 'Pneumonia', 'Rabies', 'Anthrax', 'Swine Flu']
    
    for disease in target_diseases:
        if disease == 'HEALTHY':
            # Healthy Logic
            t, h, r, a = 38.5, 70, 24, 90
        else:
            # Deterministic Logic from train_model.py
            dn = disease.upper()
            seed_val = int(hashlib.sha256(dn.encode('utf-8')).hexdigest(), 16) % (10**9)
            rng = np.random.RandomState(seed_val)
            
            base_temp = 38.0 + (rng.rand() * 4.0)
            if 38.0 < base_temp < 39.0: base_temp += 1.5
            
            base_hr = 40 + (rng.rand() * 140)
            base_resp = 10 + (rng.rand() * 80)
            base_act = 0 + (rng.rand() * 60)
            
            t, h, r, a = base_temp, base_hr, base_resp, base_act

        print(f"{disease:<30} | {t:.1f}  | {int(h):<5} | {int(r):<5} | {int(a):<5}")

reveal_cheat_sheet()
