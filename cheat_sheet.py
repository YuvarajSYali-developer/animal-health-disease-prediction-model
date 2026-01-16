import pandas as pd
import numpy as np
import hashlib

def reveal_cheat_sheet():
    print("=========================================")
    print("   A-VITAL DEMO CHEAT SHEET (v4)")
    print("   (Enter these values to trigger specific diseases)")
    print("=========================================")
    print(f"{'DISEASE NAME':<30} | {'TEMP':<5} | {'HR':<5} | {'RESP':<5} | {'ACT':<5}")
    print("-" * 65)

    target_diseases = ['HEALTHY', 'Parvovirus', 'Pneumonia', 'Rabies', 'Anthrax', 'Swine Flu']
    
    for disease in target_diseases:
        if disease == 'HEALTHY':
            # Updated Healthy Logic (Midpoint of range)
            t, h, r, a = 38.3, 75, 22, 90
        else:
            # Deterministic Logic from UPDATED train_model.py
            dn = disease.upper()
            seed_val = int(hashlib.sha256(dn.encode('utf-8')).hexdigest(), 16) % (10**9)
            rng = np.random.RandomState(seed_val)
            
            # High Separation Replication
            if rng.rand() > 0.5:
                base_temp = 39.5 + (rng.rand() * 2.5) # Fever
            else:
                base_temp = 36.0 + (rng.rand() * 1.5) # Low
                
            base_act = rng.rand() * 60 
                
            base_hr = 40 + (rng.rand() * 140)
            base_resp = 10 + (rng.rand() * 80)
            
            t, h, r, a = base_temp, base_hr, base_resp, base_act

        print(f"{disease:<30} | {t:.1f}  | {int(h):<5} | {int(r):<5} | {int(a):<5}")

reveal_cheat_sheet()
