#!/bin/bash

echo "=================================================="
echo "   A-VITAL SYSTEM LAUNCHER (Linux/Mac/GitBash)"
echo "=================================================="

echo "[1/3] Installing Dependencies..."
pip install flask flask-cors pandas numpy scikit-learn joblib

echo "[2/3] Training Neural Core..."
python train_model.py

echo "[3/3] Starting Application Node..."
echo ""
echo "   >>> OPEN YOUR BROWSER TO: http://localhost:5000 <<<"
echo ""
python app.py
