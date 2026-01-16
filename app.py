import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- LOAD MODEL (Robust) ---
model = None
le = None

try:
    print("[*] Loading Neural Network Weights...")
    model = joblib.load('animal_model.pkl')
    le = joblib.load('label_encoder.pkl')
    print("[+] System Online: Neural Core Active.")
except Exception as e:
    print(f"[!] CRITICAL ERROR: Could not load model. {e}")
    # Don't crash, just serve simplified mode
    model = None

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online', 
        'model_loaded': model is not None,
        'version': '4.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({'status': 'error', 'message': 'System offline. Train model first.'}), 503

    try:
        data = request.json
        features = data.get('features') # [Temp, HR, Resp, Activity]

        if not features or len(features) != 4:
            return jsonify({'status': 'error', 'message': 'Invalid Vector Dimensions'}), 400

        input_vector = np.array(features).reshape(1, -1)
        pred_idx = model.predict(input_vector)[0]
        confidence_array = model.predict_proba(input_vector)
        confidence_score = np.max(confidence_array) * 100

        try:
            disease_name = le.inverse_transform([pred_idx])[0]
        except:
            disease_name = f"Unknown_Class_{pred_idx}"

        # --- SMART INVALIDATION ---
        # If the model is unsure (< 35%) and it's not predicting 'HEALTHY',
        # it's better to admit uncertainty than to guess randomly.
        if confidence_score < 35.0 and str(disease_name).upper() != 'HEALTHY':
            disease_name = "INCONCLUSIVE (ABNORMAL VITALS)"
            confidence_score = 0.0

        print(f"[>] Diagnosis: {disease_name.upper()} ({confidence_score:.1f}%)")

        return jsonify({
            'status': 'success',
            'prediction': str(disease_name).upper(),
            'confidence': f"{confidence_score:.1f}%"
        })

    except Exception as e:
        print(f"[!] Inference Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print(r"""
       _               _ _        _ 
      / \   _ __  _ __(_) |______| |
     / _ \ | '_ \| '_ \ | |______| |
    / ___ \| |_) | |_) | |      | |
   /_/   \_\ .__/| .__/|_|      |_|
           |_|   |_|               
    """)
    print("   VETERINARY INTELLIGENCE NODE v4.0")
    print("   ---------------------------------")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
