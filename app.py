import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- LOAD ARTIFACTS ---
model = None
le = None
model_features = None

def load_system():
    global model, le, model_features
    try:
        print("[*] Loading Neural Network Weights...")
        model = joblib.load('animal_model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_features = joblib.load('model_features.pkl')
        print(f"[+] System Online. Features aligned: {len(model_features)}")
        return True
    except Exception as e:
        print(f"[!] CRITICAL ERROR: {e}")
        return False

# Initialize
success = load_system()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online', 
        'model_loaded': model is not None,
        'version': '4.1.0-Enhanced'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not load_system():
            return jsonify({'status': 'error', 'message': 'System Initialization Failed'}), 503

    try:
        data = request.json
        
        # User Input: 
        # {
        #   "species": "Dog",
        #   "temp": 39.5,
        #   "hr": 120,
        #   "resp": 30,
        #   "activity": 50,
        #   "symptoms": ["Vomiting", "Diarrhea"]
        # }
        
        # 1. Initialize empty feature vector with correct columns
        input_data = {col: 0 for col in model_features}
        
        # 2. Fill Vitals
        input_data['Body_Temperature'] = float(data.get('temp', 38.0))
        input_data['Heart_Rate'] = int(data.get('hr', 80))
        input_data['Respiratory_Rate'] = int(data.get('resp', 20))
        input_data['Activity_Level'] = int(data.get('activity', 100))
        
        # 3. Fill One-Hot Species
        species = data.get('species', 'Dog')
        species_col = f"Animal_Type_{species}"
        if species_col in input_data:
            input_data[species_col] = 1
        
        # 4. Fill Symptoms
        active_symptoms = data.get('symptoms', [])
        for sym in active_symptoms:
            if sym in input_data:
                input_data[sym] = 1
                
        # 5. Create DataFrame for Prediction
        df_input = pd.DataFrame([input_data])
        
        # 6. Predict
        pred_idx = model.predict(df_input)[0]
        confidence_array = model.predict_proba(df_input)[0]  # Get probability array
        
        # Get sorted predictions
        top_indices = np.argsort(confidence_array)[::-1]
        
        # Top 1
        top_pred_name = le.inverse_transform([top_indices[0]])[0]
        top_conf = confidence_array[top_indices[0]] * 100
        
        # Logic: Find the first NON-HEALTHY prediction if symptoms are present
        final_prediction = top_pred_name
        final_conf = top_conf
        note = ""
        
        if active_symptoms:
            # If Model thinks it's healthy but we see symptoms, dig deeper
            if "HEALTHY" in top_pred_name.upper():
                # Loop through top 3 predictions to find a disease
                found_disease = False
                for i in range(1, 4): # Check 2nd, 3rd, 4th
                    if i >= len(top_indices): break
                    
                    alt_name = le.inverse_transform([top_indices[i]])[0]
                    alt_conf = confidence_array[top_indices[i]] * 100
                    
                    if "HEALTHY" not in alt_name.upper() and alt_conf > 10.0:
                        final_prediction = alt_name
                        final_conf = alt_conf
                        note = " (Pattern Match)"
                        found_disease = True
                        break
                
                if not found_disease:
                    final_prediction = "Unknown Infection"
                    final_conf = 0.1 # Non-zero to show it exists but is low
                    note = " (Vitals check out, but symptoms persist)"

        # Normalize Confidence Display
        # If it's really low, don't say 0%, say the value but warn
        if final_conf < 30.0 and final_prediction != "Unknown Infection":
            note += " [Low Confidence]"

        # 7. Generate "Why" (Reasoning)
        reasoning = []
        if active_symptoms:
            reasoning.append(f"Symptoms: {', '.join(active_symptoms)}")
        else:
            reasoning.append("No active symptoms")
            
        if input_data['Body_Temperature'] > 40.0:
            reasoning.append("High Fever")
        
        why_text = " | ".join(reasoning) + note
        
        print(f"[>] Diagnosis: {final_prediction.upper()} ({final_conf:.1f}%)")

        return jsonify({
            'status': 'success',
            'prediction': str(final_prediction).upper(),
            'confidence': f"{final_conf:.1f}%",
            'reasoning': why_text
        })

    except Exception as e:
        print(f"[!] Inference Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("[+] System V4.1 Loaded... Starting Server...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
