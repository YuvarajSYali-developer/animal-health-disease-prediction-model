import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate_model():
    # 1. Load Dataset
    print("Loading dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum().sum()}")
    
    # 2. Prepare Data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Initialize and Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False) # block=False so it doesn't hang the script if run from terminal
    
    return model, scaler, data.feature_names

def predict_single_patient(model, scaler, feature_names):
    # Example: Simulating a new patient with mean values from the dataset
    # In a real app, you would get these from user input
    print("\n--- Simulating New Patient Prediction ---")
    
    # Let's take a random sample from the dataset to simulate a patient
    sample_data = load_breast_cancer()
    random_idx = np.random.randint(0, len(sample_data.target))
    patient_data = sample_data.data[random_idx]
    actual_diagnosis = sample_data.target[random_idx]
    
    print(f"Patient Features (Subset): {patient_data[:5]}...")
    
    # Preprocess
    patient_data_scaled = scaler.transform([patient_data])
    
    # Predict
    prediction = model.predict(patient_data_scaled)
    prediction_prob = model.predict_proba(patient_data_scaled)
    
    class_name = sample_data.target_names[prediction[0]]
    confidence = np.max(prediction_prob) * 100
    
    print(f"Predicted Diagnosis: {class_name.upper()} ({confidence:.2f}% confidence)")
    print(f"Actual Diagnosis:    {sample_data.target_names[actual_diagnosis].upper()}")

if __name__ == "__main__":
    try:
        trained_model, trained_scaler, feature_names = train_and_evaluate_model()
        predict_single_patient(trained_model, trained_scaler, feature_names)
        print("\nProcess completed successfully.")
    except ImportError as e:
        print(f"\nError: Missing library. {e}")
        print("Please install requirements: pip install pandas numpy scikit-learn matplotlib seaborn")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
