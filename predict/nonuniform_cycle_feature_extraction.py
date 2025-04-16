# nonuniform_cycle_feature_extraction.py

"""
Nonuniform-Cycle Feature Extraction and Demagnetization Prediction
------------------------------------------------------------------

This script performs feature extraction on nonuniform-cycle current data 
from an IPMSM machine. It trains a Gradient Boosting Regressor model to 
predict the demagnetization severity percentage and evaluates it on live data.

Author: Gururaghuraman
Date: 2025-03-18
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# ----------------------- Configuration -----------------------

CYCLE_LENGTH = 41
TOTAL_CYCLES = 48

dataset_files = {
    "I5nu.csv": 0,
    "I10nu.csv": 5,
    "I15nu.csv": 10,
    "I20nu.csv": 20,
    "I25nu.csv": 25
}

# -------------------- Feature Extraction --------------------

def extract_features(df):
    """
    Extracts statistical and RMS features from the current phases.
    Also computes phase imbalance.
    """
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

# --------------------- Dataset Processing ---------------------

feature_list = []
severity_labels = []

for file_name, severity in dataset_files.items():
    df = pd.read_csv(file_name)
    num_samples = len(df)
    num_cycles = min(TOTAL_CYCLES, num_samples // CYCLE_LENGTH)

    if num_cycles == 0:
        print(f"Skipping {file_name} (Not enough data for a single cycle)")
        continue

    for i in range(num_cycles):
        cycle_df = df.iloc[i * CYCLE_LENGTH:(i + 1) * CYCLE_LENGTH]
        features = extract_features(cycle_df)
        feature_list.append(features)
        severity_labels.append(severity)

# ------------------------- Training ---------------------------

train_data = pd.DataFrame(feature_list)
train_data["Severity"] = severity_labels

print("\nExtracted Features for Training Data:")
print(train_data.to_string(index=False))

X = train_data.drop(columns=["Severity"])
y = train_data["Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.00001, max_depth=4)
model.fit(X_train_scaled, y_train)

# ---------------------- Live Prediction -----------------------

def predict_demagnetization_percentage(live_file):
    """
    Predicts demagnetization percentage for each cycle in the live data.
    Also prints the average demagnetization across all cycles.
    """
    live_data = pd.read_csv(live_file)
    num_samples = len(live_data)
    num_cycles = min(TOTAL_CYCLES, num_samples // CYCLE_LENGTH)

    predictions = []
    cycle_features_list = []

    for i in range(num_cycles):
        cycle_df = live_data.iloc[i * CYCLE_LENGTH:(i + 1) * CYCLE_LENGTH]
        live_features = extract_features(cycle_df)
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

# ----------------------- Run Prediction -----------------------

predict_demagnetization_percentage("I12nu.csv")
