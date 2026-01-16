from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys

# --- CINEMATIC BOOT SEQUENCE ---
print("=========================================")
print("   A-VITAL BIOLOGICAL INTELLIGENCE NODE  ")
print("=========================================")
print("Initializing Neural Core...")

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- CONFIGURATION & ASSETS ---
MODEL_PATH = 'animal_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'

model = None
le = None

def load_system():
    global model, le
    print(f"Loading Artifacts from {os.getcwd()}...")
    
    # Load Random Forest Model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ [CORE] Model Loaded Successfully.")
        except Exception as e:
            print(f"❌ [CORE] Model Load Failed: {e}")
    else:
        print(f"⚠️ [CORE] {MODEL_PATH} not found. Prediction will fail.")

    # Load Label Encoder
    if os.path.exists(ENCODER_PATH):
        try:
            le = joblib.load(ENCODER_PATH)
            print("✅ [DECODER] Label Encoder Active.")
        except Exception as e:
            print(f"❌ [DECODER] Encoder Load Failed: {e}")
    else:
        print(f"⚠️ [DECODER] {ENCODER_PATH} not found. Output will be raw class IDs.")

# Initialize Logic
load_system()

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the Cinematic Dashboard"""
    # Flask looks in the 'templates' folder for this file
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Bio-Digital Inference Endpoint
    Receives: { features: [Temp, HR, Resp, Activity] }
    Returns:  { prediction: "Disease Name", confidence: "99.9%" }
    """
    if not model or not le:
        return jsonify({'status': 'error', 'message': 'Neural Core Offline (Model Missing)'}), 503

    try:
        # 1. Parse Input
        data = request.json
        print(f"📥 Received Vitals: {data}")
        
        features = data.get('features')
        if not features or len(features) != 4:
            return jsonify({'status': 'error', 'message': 'Invalid Vector Dimensions'}), 400

        # 2. Preprocessing
        # Reshape to (1, 4) for single-sample prediction
        input_vector = np.array(features).reshape(1, -1)

        # 3. Neural Inference
        pred_idx = model.predict(input_vector)[0]
        confidence_array = model.predict_proba(input_vector)
        confidence_score = np.max(confidence_array) * 100

        # 4. Decoding (Index -> String)
        try:
            disease_name = le.inverse_transform([pred_idx])[0]
        except:
            disease_name = f"Unknown_Class_{pred_idx}"

        print(f"📤 Diagnosis: {disease_name.upper()} ({confidence_score:.1f}%)")

        return jsonify({
            'status': 'success',
            'prediction': str(disease_name).upper(),
            'confidence': f"{confidence_score:.1f}%"
        })

    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def check_status():
    return jsonify({
        'status': 'ONLINE',
        'model': 'READY' if model else 'MISSING'
    })

if __name__ == '__main__':
    print("\nSYSTEM READY. ACCESS DASHBOARD AT:")
    print(" >>> http://localhost:5000 <<< \n")
    # UPDATED: Dynamic Port for Cloud Hosting (Render/Heroku)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
