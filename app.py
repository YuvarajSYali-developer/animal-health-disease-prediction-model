import json
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- LOAD ARTIFACTS ---
model = None
le = None
model_features = None
metrics = None

def load_system():
    global model, le, model_features, metrics
    try:
        print("[*] Loading Neural Network Weights...")
        model = joblib.load('animal_model.pkl')
        le = joblib.load('label_encoder.pkl')
        model_features = joblib.load('model_features.pkl')
        try:
            with open('training_metrics.json', 'r', encoding='utf-8') as handle:
                metrics = json.load(handle)
        except Exception:
            metrics = None
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
        'version': '4.2.0-Enhanced',
        'metrics': metrics or {}
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not load_system():
            return jsonify({'status': 'error', 'message': 'System Initialization Failed'}), 503

    try:
        data = request.json
        if data is None:
            return jsonify({'status': 'error', 'message': 'Missing JSON body'}), 400
        
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
        try:
            input_data['Body_Temperature'] = float(data.get('temp', 38.0))
            input_data['Heart_Rate'] = int(data.get('hr', 80))
            input_data['Respiratory_Rate'] = int(data.get('resp', 20))
            input_data['Activity_Level'] = int(data.get('activity', 100))
        except (TypeError, ValueError):
            return jsonify({'status': 'error', 'message': 'Vitals must be numeric values'}), 400
        
        # 3. Fill One-Hot Species
        species = data.get('species', 'Dog')
        if not isinstance(species, str):
            return jsonify({'status': 'error', 'message': 'Species must be a string'}), 400
        species_col = f"Animal_Type_{species}"
        if species_col in input_data:
            input_data[species_col] = 1
        
        # 4. Fill Symptoms
        active_symptoms = data.get('symptoms', [])
        if not isinstance(active_symptoms, list):
            return jsonify({'status': 'error', 'message': 'Symptoms must be a list'}), 400
        for sym in active_symptoms:
            if sym in input_data:
                input_data[sym] = 1
                
        # 5. Create DataFrame for Prediction (ensure correct column order)
        df_input = pd.DataFrame([input_data], columns=model_features)
        
        # 6. Predict (use DMatrix to guarantee feature names are present)
        dmatrix = xgb.DMatrix(df_input, feature_names=list(model_features))
        confidence_array = model.get_booster().predict(dmatrix)[0]
        pred_idx = int(np.argmax(confidence_array))
        
        # Get sorted predictions
        top_indices = np.argsort(confidence_array)[::-1]
        
        # Top 1
        top_pred_name = le.inverse_transform([top_indices[0]])[0]
        top_conf = confidence_array[top_indices[0]] * 100

        top_predictions = []
        for idx in top_indices[:3]:
            top_predictions.append({
                "disease": le.inverse_transform([idx])[0],
                "confidence": f"{confidence_array[idx] * 100:.1f}%"
            })
        
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
            'reasoning': why_text,
            'top_predictions': top_predictions
        })

    except Exception as e:
        print(f"[!] Inference Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("[+] System V4.1 Loaded... Starting Server...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
