# AI Assignment - Patient Readmission Prediction System
# Author: Hargreaves Hardennes Gitonga
# Project Goal: Predict if a patient will be readmitted within 30 days after hospital discharge

# ------------------------
# 1. DATA COLLECTION
# ------------------------
import pandas as pd
import numpy as np

# Example: Load hospital dataset
# You would replace this with real hospital EHR data in practice
data = pd.read_csv('hospital_readmission.csv')

# ------------------------
# 2. DATA PREPROCESSING
# ------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[['age', 'num_lab_procedures']] = imputer.fit_transform(data[['age', 'num_lab_procedures']])

# Encode target variable
# 1 = readmitted within 30 days, 0 = otherwise
data['readmitted'] = data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Feature selection
features = ['age', 'num_lab_procedures', 'time_in_hospital', 'num_medications']
X = data[features]
y = data['readmitted']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ------------------------
# 3. MODEL DEVELOPMENT
# ------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Initialize model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ------------------------
# 4. EVALUATION
# ------------------------
from sklearn.metrics import confusion_matrix, classification_report

y_pred = best_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# ------------------------
# 5. DEPLOYMENT SIMULATION
# ------------------------
import joblib

# Save model
joblib.dump(best_model, 'readmission_predictor.pkl')

# Simulated API prediction (like hospital system would do)
def predict_risk(input_features):
    input_scaled = scaler.transform([input_features])
    prediction = best_model.predict(input_scaled)
    return "High Risk" if prediction[0] == 1 else "Low Risk"

# Example prediction
example_patient = [60, 12, 5, 10]  # age, labs, days, meds
print("\nPrediction for example patient:", predict_risk(example_patient))
