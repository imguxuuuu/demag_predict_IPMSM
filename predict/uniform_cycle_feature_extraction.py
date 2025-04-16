# uniform_cycle_feature_extraction.py
"""
Demagnetisation Fault Feature Extraction and Prediction for IPMSM

This script extracts statistical features cycle-by-cycle from motor current data
and predicts demagnetisation percentage using Gradient Boosting.

Author: Gururaghuraman
Date: 2024-12-18
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Constants
# -----------------------------

CYCLE_LENGTH = 46600
TOTAL_CYCLES = 11  # Considering 10.7 cycles as 11

# -----------------------------
# Feature Extraction Function
# -----------------------------

def extract_features(df):
    """Extract statistical features from one cycle of current data."""
    features = {}

    for phase in ["Current(PhaseA) [A]", "Current(PhaseB) [A]", "Current(PhaseC) [A]"]:
        features[f"{phase}_mean"] = df[phase].mean()
        features[f"{phase}_std"] = df[phase].std()
        features[f"{phase}_min"] = df[phase].min()
        features[f"{phase}_max"] = df[phase].max()
        features[f"{phase}_rms"] = np.sqrt(np.mean(df[phase] ** 2))

    max_current = df[["Current(PhaseA) [A]", "Current(PhaseB) [A]", "Current(PhaseC) [A]"]].max().max()
    min_current = df[["Current(PhaseA) [A]", "Current(PhaseB) [A]", "Current(PhaseC) [A]"]].min().min()
    features["phase_imbalance"] = max_current - min_current

    return features

# -----------------------------
# Load Training Data
# -----------------------------

dataset_files = {
    "healthy_data.csv": 0,
    "demag_5_data.csv": 5,
    "demag_10_data.csv": 10,
    "demag_15_data.csv": 15,
    "demag_20_data.csv": 20,
    "demag_25_data.csv": 25
}

feature_list = []
severity_labels = []

for file_name, severity in dataset_files.items():
    df = pd.read_csv(file_name)
    num_samples = len(df)
    num_cycles = min(TOTAL_CYCLES, num_samples // CYCLE_LENGTH)

    cycle_features_list = []
    for i in range(num_cycles):
        cycle_df = df.iloc[i * CYCLE_LENGTH:(i + 1) * CYCLE_LENGTH]
        features = extract_features(cycle_df)
        feature_list.append(features)
        severity_labels.append(severity)
        features["Cycle"] = i + 1
        features["Severity"] = severity
        cycle_features_list.append(features)

    cycle_features_df = pd.DataFrame(cycle_features_list)
    print(f"\nExtracted Features for {file_name} (Severity {severity}):")
    print(cycle_features_df.drop(columns=["Cycle"]).to_string(index=False))

# -----------------------------
# Train Model
# -----------------------------

train_data = pd.DataFrame(feature_list)
train_data["Severity"] = severity_labels

X = train_data.drop(columns=["Severity"])
y = train_data["Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Live Prediction Function
# -----------------------------

def predict_demagnetization_percentage(live_file):
    """Predict demagnetisation percentage from live IPMSM current data."""
    live_data = pd.read_csv(live_file)
    num_samples = len(live_data)
    num_cycles = min(TOTAL_CYCLES, num_samples // CYCLE_LENGTH)

    predictions = []
    cycle_features_list = []

    for i in range(num_cycles):
        cycle_df = live_data.iloc[i * CYCLE_LENGTH:(i + 1) * CYCLE_LENGTH]
        live_features = extract_features(cycle_df)
        live_features['Cycle'] = i + 1
        live_df = pd.DataFrame([live_features])
        cycle_features_list.append(live_features)

        live_scaled = scaler.transform(live_df)
        prediction = model.predict(live_scaled)[0]
        predictions.append(prediction)
        live_features["Predicted Demagnetization (%)"] = prediction

    cycle_features_df = pd.DataFrame(cycle_features_list)
    print("\nExtracted Features and Predictions for Live Machine Data:")
    print(cycle_features_df.to_string(index=False))

    avg_prediction = np.mean(predictions)
    print(f"\nFinal Estimated Demagnetization Percentage (Average of Cycles): {avg_prediction:.2f}%")

# -----------------------------
# Run Prediction
# -----------------------------

predict_demagnetization_percentage("live_machine_data.csv")
