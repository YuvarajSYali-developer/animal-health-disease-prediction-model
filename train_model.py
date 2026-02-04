import argparse
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DEFAULT_DATASET = "enhanced_animal_disease.csv"
COMBINED_DATASET = Path("data") / "combined_dataset.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the animal disease prediction model.")
    parser.add_argument(
        "--dataset",
        default=str(COMBINED_DATASET if COMBINED_DATASET.exists() else DEFAULT_DATASET),
        help="Path to the training dataset CSV.",
    )
    return parser.parse_args()

def build_and_train(dataset_path: str):
    print(f"[1/5] Loading Dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(
            f"Error: '{dataset_path}' not found. "
            "Run generate_data.py or data_ingest.py first."
        )
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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.15,
        random_state=42,
        stratify=y_encoded,
    )
    
    # XGBoost Calculation
    model = XGBClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )
    
    # --- EVALUATION ---
    print("[4/5] Evaluating...")
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    balanced = balanced_accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"   -> Accuracy: {accuracy * 100:.2f}%")
    print(f"   -> Balanced Accuracy: {balanced * 100:.2f}%")
    print(f"   -> Macro F1: {macro_f1 * 100:.2f}%")
    
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

    metrics = {
        "dataset": dataset_path,
        "rows": int(df.shape[0]),
        "features": int(len(feature_names)),
        "classes": int(len(le.classes_)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(macro_f1),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    with open("training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    
    print("Build Complete.")

if __name__ == "__main__":
    args = parse_args()
    build_and_train(args.dataset)
