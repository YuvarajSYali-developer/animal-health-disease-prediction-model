import pandas as pd
import numpy as np
import random

def generate_enhanced_dataset(num_samples=8000):
    print("Generating Comprehensive Veterinary Dataset (v4 - Specialist Level)...")
    
    # --- EXPANDED SYMPTOM LIST (30 Clinical Signs) ---
    # This covers Systemic, GI, Neuro, Resp, Derm, and Urinary systems
    all_symptoms = [
        # GI
        'Vomiting', 'Diarrhea', 'Appetite_Loss', 'Bloat_Distension', 'Dehydration',
        # Respiratory
        'Coughing', 'Sneezing', 'Nasal_Discharge', 'Resp_Distress', 
        # Systemic/Pain
        'Lethargy', 'Fever_Chills', 'Weight_Loss', 'Pale_Gums', 'Jaundice',
        # Musculoskeletal/Derm
        'Lameness', 'Swelling', 'Stiff_Joints', 'Skin_Lesions', 'Hair_Loss', 'Blisters', 'Pustules',
        # Neurological
        'Seizures', 'Tremors', 'Uncoordinated', 'Aggression',
        # Head/Face
        'Eye_Discharge', 'Excess_Saliva', 'Swollen_Lymph_Nodes',
        # Urinary
        'Straining_Urinate', 'Red_Urine'
    ]

    # --- COMPREHENSIVE DISEASE DATABASE ---
    diseases = {
        # --- CANINE (DOG) ---
        'K9 Parvovirus': {
            'species': ['Dog', 'Fox'], # Added wild dog proxy
            'symptoms': {'Vomiting': 0.95, 'Diarrhea': 0.95, 'Appetite_Loss': 0.95, 'Lethargy': 0.95, 'Dehydration': 0.9, 'Red_Urine': 0.1}, # Bloody diarrhea proxy
            'vitals': {'temp': (39.5, 41.5), 'hr': (120, 180), 'resp': (30, 60), 'act': (0, 10)}
        },
        'K9 Distemper': {
            'species': ['Dog'],
            'symptoms': {'Coughing': 0.9, 'Eye_Discharge': 0.95, 'Nasal_Discharge': 0.95, 'Seizures': 0.4, 'Tremors': 0.5, 'Hard_Pads': 0.3},
            'vitals': {'temp': (39.5, 41.0), 'hr': (100, 150), 'resp': (30, 50), 'act': (10, 40)}
        },
        'Rabies': {
            'species': ['Dog', 'Cat', 'Cow', 'Horse', 'Fox'],
            'symptoms': {'Aggression': 0.9, 'Excess_Saliva': 0.95, 'Seizures': 0.7, 'Uncoordinated': 0.8, 'Appetite_Loss': 0.5},
            'vitals': {'temp': (39.5, 41.0), 'hr': (100, 160), 'resp': (40, 80), 'act': (80, 100)} # Agitated
        },
        'Heartworm Disease': {
            'species': ['Dog'],
            'symptoms': {'Coughing': 0.9, 'Resp_Distress': 0.8, 'Lethargy': 0.7, 'Weight_Loss': 0.6},
            'vitals': {'temp': 'NORMAL', 'hr': (100, 140), 'resp': (40, 70), 'act': (20, 50)}
        },
        'Lyme Disease': {
            'species': ['Dog', 'Horse'],
            'symptoms': {'Lameness': 0.9, 'Stiff_Joints': 0.95, 'Lethargy': 0.7, 'Fever_Chills': 0.8},
            'vitals': {'temp': (39.5, 40.5), 'hr': 'NORMAL', 'resp': 'NORMAL', 'act': (30, 60)}
        },
        
        # --- FELINE (CAT) ---
        'Feline Panleukopenia': {
            'species': ['Cat'],
            'symptoms': {'Vomiting': 0.95, 'Diarrhea': 0.9, 'Appetite_Loss': 0.95, 'Lethargy': 0.95, 'Dehydration': 0.8},
            'vitals': {'temp': (39.5, 41.5), 'hr': (160, 240), 'resp': (40, 80), 'act': (0, 20)}
        },
        'Feline FLUTD (Urinary)': {
            'species': ['Cat'],
            'symptoms': {'Straining_Urinate': 1.0, 'Red_Urine': 0.9, 'Aggression': 0.4, 'Lethargy': 0.3},
            'vitals': {'temp': 'NORMAL', 'hr': (140, 200), 'resp': 'NORMAL', 'act': (50, 80)} # Pain stress
        },
        'Cat Flu (URI)': { 
            'species': ['Cat'],
            'symptoms': {'Sneezing': 0.95, 'Eye_Discharge': 0.9, 'Nasal_Discharge': 0.9, 'Appetite_Loss': 0.5},
            'vitals': {'temp': (39.0, 40.0), 'hr': (140, 180), 'resp': (30, 50), 'act': (40, 70)}
        },

        # --- BOVINE (COW) ---
        'Bovine Mastitis': {
            'species': ['Cow', 'Goat', 'Sheep'],
            'symptoms': {'Swelling': 0.95, 'Appetite_Loss': 0.6, 'Lethargy': 0.6, 'Fever_Chills': 0.7},
            'vitals': {'temp': (39.5, 41.5), 'hr': (80, 110), 'resp': (30, 60), 'act': (20, 50)}
        },
        'Foot and Mouth': {
            'species': ['Cow', 'Pig', 'Sheep', 'Goat'],
            'symptoms': {'Lameness': 0.95, 'Excess_Saliva': 0.95, 'Blisters': 1.0, 'Appetite_Loss': 0.9},
            'vitals': {'temp': (40.0, 42.0), 'hr': (90, 120), 'resp': (40, 70), 'act': (10, 30)}
        },
        'Milk Fever (Hypocalcemia)': {
            'species': ['Cow'],
            'symptoms': {'Tremors': 0.8, 'Uncoordinated': 0.9, 'Lethargy': 1.0, 'Bloat_Distension': 0.3},
            'vitals': {'temp': (36.0, 37.8), 'hr': (80, 120), 'resp': (10, 30), 'act': (0, 10)} # LOW TEMP
        },
        'Bloat': {
            'species': ['Cow', 'Sheep', 'Goat'],
            'symptoms': {'Bloat_Distension': 1.0, 'Resp_Distress': 0.8, 'Restlessness': 0.7, 'Appetite_Loss': 1.0},
            'vitals': {'temp': 'NORMAL', 'hr': (100, 140), 'resp': (60, 90), 'act': (20, 40)}
        },

        # --- EQUINE (HORSE) ---
        'Strangles': {
            'species': ['Horse'],
            'symptoms': {'Coughing': 0.3, 'Nasal_Discharge': 0.95, 'Swollen_Lymph_Nodes': 1.0, 'Lethargy': 0.8},
            'vitals': {'temp': (39.5, 41.0), 'hr': (45, 65), 'resp': (20, 35), 'act': (20, 40)}
        },
        'Colic': {
            'species': ['Horse'],
            'symptoms': {'Rolling': 1.0, 'Sweating': 0.9, 'Appetite_Loss': 1.0, 'Restlessness': 1.0, 'Bloat_Distension': 0.4},
            'vitals': {'temp': (37.5, 39.0), 'hr': (50, 100), 'resp': (25, 60), 'act': (40, 60)} 
        },
        'Laminitis': {
            'species': ['Horse'],
            'symptoms': {'Lameness': 1.0, 'Stiff_Joints': 0.8, 'Sweating': 0.5, 'Restlessness': 0.6},
            'vitals': {'temp': (37.5, 39.0), 'hr': (50, 80), 'resp': (20, 50), 'act': (10, 30)}
        },
        'Tetanus': {
            'species': ['Horse', 'Dog', 'Sheep'],
            'symptoms': {'Stiff_Joints': 1.0, 'Tremors': 0.8, 'Excess_Saliva': 0.5, 'Resp_Distress': 0.6},
            'vitals': {'temp': (38.5, 40.0), 'hr': (80, 120), 'resp': (30, 60), 'act': (0, 10)} # Locked up
        },

        # --- PORCINE (PIG) ---
        'Swine Erysipelas': {
            'species': ['Pig'],
            'symptoms': {'Skin_Lesions': 1.0, 'Fever_Chills': 0.9, 'Lameness': 0.7, 'Lethargy': 0.8},
            'vitals': {'temp': (40.0, 42.0), 'hr': (90, 130), 'resp': (30, 60), 'act': (10, 30)}
        },
        'Gastroenteritis (General)': {
            'species': ['Dog', 'Cat', 'Pig'],
            'symptoms': {'Vomiting': 0.9, 'Diarrhea': 0.9, 'Appetite_Loss': 0.8},
            'vitals': {'temp': (38.5, 40.0), 'hr': (100, 160), 'resp': (20, 40), 'act': (30, 60)}
        },
        
        # --- GENERAL ---
        'Internal Parasites': {
            'species': ['Dog', 'Cat', 'Cow', 'Sheep', 'Goat', 'Horse'],
            'symptoms': {'Weight_Loss': 0.9, 'Pale_Gums': 0.8, 'Diarrhea': 0.4, 'Lethargy': 0.5},
            'vitals': 'NORMAL'
        },
        'Healthy': {
            'species': ['Dog', 'Cat', 'Cow', 'Horse', 'Pig', 'Sheep', 'Goat'],
            'symptoms': {}, 
            'vitals': 'SPECIES_SPECIFIC'
        }
    }

    # Species "Normal" Vitals (Temp, HR, Resp)
    species_vitals = {
        'Dog': ((38.0, 39.0), (60, 100), (10, 30)),
        'Cat': ((38.0, 39.2), (120, 180), (20, 40)),
        'Cow': ((38.0, 39.0), (48, 84), (26, 50)),
        'Horse': ((37.2, 38.3), (28, 44), (10, 24)),
        'Pig': ((38.7, 39.8), (60, 80), (10, 20)),
        'Sheep': ((39.0, 40.0), (70, 80), (15, 30)),
        'Goat': ((38.5, 39.7), (70, 80), (15, 30)),
        'Fox': ((38.5, 39.8), (80, 120), (20, 40)) # Added for robustness
    }

    data = []

    for _ in range(num_samples):
        # 1. Pick a disease
        keys = list(diseases.keys())
        weights = [1] * len(keys)
        # Healthy still needs to be baseline
        healthy_idx = keys.index('Healthy')
        weights[healthy_idx] = 4 # High baseline to avoid false positives
        
        disease_name = random.choices(keys, weights=weights, k=1)[0]
        info = diseases[disease_name]
        
        # 2. Pick a species
        species = random.choice(info['species'])
        
        row = {
            'Animal_Type': species,
            'Disease_Prediction': disease_name
        }

        # 3. Generate Vitals
        if species in species_vitals:
            sp_vitals = species_vitals[species]
        else:
            sp_vitals = species_vitals['Dog'] # Fallback
        
        # Early Stage Logic
        is_early = False
        if disease_name != 'Healthy' and random.random() < 0.20:
            is_early = True

        if disease_name == 'Healthy' or is_early or info.get('vitals') == 'NORMAL':
            # Use species normal
            t_range, h_range, r_range = sp_vitals
            row['Body_Temperature'] = round(random.uniform(*t_range), 1)
            row['Heart_Rate'] = int(random.uniform(*h_range))
            row['Respiratory_Rate'] = int(random.uniform(*r_range))
            
            if disease_name == 'Healthy':
                row['Activity_Level'] = int(random.uniform(90, 100))
            else:
                row['Activity_Level'] = int(random.uniform(70, 95)) 
        else:
            # Severe Case
            v = info['vitals']
            
            t_range = v.get('temp', sp_vitals[0])
            if t_range == 'NORMAL': t_range = sp_vitals[0]
            
            h_range = v.get('hr', sp_vitals[1])
            if h_range == 'NORMAL': h_range = sp_vitals[1]
            
            r_range = v.get('resp', sp_vitals[2])
            if r_range == 'NORMAL': r_range = sp_vitals[2]
            
            a_range = v.get('act', (20, 60))

            row['Body_Temperature'] = round(random.uniform(*t_range), 1)
            row['Heart_Rate'] = int(random.uniform(*h_range))
            row['Respiratory_Rate'] = int(random.uniform(*r_range))
            row['Activity_Level'] = int(random.uniform(*a_range))
        
        # 4. Generate Symptoms
        for sym in all_symptoms:
            row[sym] = 0
            
        disease_symptoms = info.get('symptoms', {})
        for sym, prob in disease_symptoms.items():
            if random.random() < prob:
                if sym in row: # Safety check
                    row[sym] = 1
        
        # 5. Logical Inconsistencies Fix (Sanity Check)
        # A dead animal (Act=0) shouldn't be "Healthy"
        if row['Activity_Level'] < 30 and disease_name == 'Healthy':
             row['Activity_Level'] = 80 # Force fix

        data.append(row)

    df = pd.DataFrame(data)
    
    # Save
    df.to_csv('enhanced_animal_disease.csv', index=False)
    print(f"Specialist dataset generated with {len(df)} samples.")
    print(f"Covering {len(keys)} distinct conditions.")

if __name__ == "__main__":
    generate_enhanced_dataset()
