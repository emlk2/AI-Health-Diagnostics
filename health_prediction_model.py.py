import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_health_model():
    print("--- Medical AI: Disease Prediction Model ---")
    print("Initializing Healthcare Data Pipeline...\n")

    # Simulating a medical dataset (e.g., Blood pressure, Glucose, BMI, Age -> Outcome)
    # In a real scenario, this would be loaded via pandas: pd.read_csv('kaggle_health_data.csv')
    np.random.seed(42)
    sample_size = 500
    
    data = {
        'Glucose_Level': np.random.randint(70, 200, sample_size),
        'Blood_Pressure': np.random.randint(60, 120, sample_size),
        'BMI': np.random.uniform(18.5, 40.0, sample_size),
        'Age': np.random.randint(21, 80, sample_size),
        # 1 means High Risk / Disease Present, 0 means Healthy
        'Risk_Outcome': np.random.choice([0, 1], sample_size, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)
    
    # Feature engineering & splitting data
    X = df.drop('Risk_Outcome', axis=1)
    y = df['Risk_Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier on Medical Data...")
    # Using Random Forest as it's highly effective for medical tabular data
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Evaluating Model Accuracy...\n")
    y_pred = rf_model.predict(X_test)
    
    # Simulating a clinical report
    print("--- Clinical Decision Support System Report ---")
    print(f"Model Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Detailed Classification Report (Precision / Recall):")
    print(classification_report(y_test, y_pred))
    
    print("Note: This model is a prototype. Future integration planned via .NET PWA for mobile access.")

if __name__ == "__main__":
    train_health_model()