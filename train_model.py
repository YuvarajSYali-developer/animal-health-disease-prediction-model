import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def build_and_train():
    print("[1/5] Loading Enhanced Dataset (3000 samples)...")
    try:
        df = pd.read_csv('enhanced_animal_disease.csv')
    except FileNotFoundError:
        print("Error: 'enhanced_animal_disease.csv' not found. Run generate_data.py first.")
        return

    # --- PREPROCESSING ---
    print("[2/5] Preprocessing & Encoding...")
    
    # 1. Separate Target
    y = df['Disease_Prediction']
    
    # 2. Separate Features
    X_raw = df.drop('Disease_Prediction', axis=1)
    
    # One-Hot Encode 'Animal_Type'
    X = pd.get_dummies(X_raw, columns=['Animal_Type'], drop_first=False)
    
    # Boolean to Int (for XGBoost)
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    feature_names = X.columns.tolist()
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"   -> Features: {len(feature_names)}")
    
    # --- TRAINING (XGBOOST) ---
    print("[3/5] Training XGBoost Classifier...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
    
    # XGBoost Calculation
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # --- EVALUATION ---
    print("[4/5] Evaluating...")
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"   -> Model Accuracy: {accuracy * 100:.2f}%")
    
    # Feature Importance
    print("\n[!] Top 5 Key Predictors:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(5):
        print(f"    {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # --- EXPORT ---
    print("\n[5/5] Exporting Artifacts...")
    joblib.dump(model, 'animal_model.pkl', compress=3) 
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(feature_names, 'model_features.pkl')
    
    print("Build Complete.")

if __name__ == "__main__":
    build_and_train()
