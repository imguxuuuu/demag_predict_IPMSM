"""
classify_live_data.py

📌 Real-time classification of IPMSM health using current readings and a trained ML model.

- Classifies samples into Healthy or Faulty based on predicted demagnetization severity.
- Visualizes the classification results as a bar chart.

Author: Gururaghuraman
Date: 2025-01-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib  # For loading model and scaler

# --- CONFIGURATION ---

SEVERITY_THRESHOLD = 5  # Threshold above which samples are considered faulty
DATA_FILE = "live_machine_data.csv"  # Path to the live input CSV
MODEL_PATH = "model.pkl"             # Path to your trained ML model
SCALER_PATH = "scaler.pkl"           # Path to your saved scaler

# --- LOAD MODEL AND SCALER ---

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("❌ Please ensure model.pkl and scaler.pkl are in the correct path.")

# --- HEALTH CLASSIFICATION FUNCTION ---

def classify_health_status(live_data: pd.DataFrame):
    """
    Predicts severity using a trained model and classifies samples into Healthy or Faulty.
    Visualizes the result using a bar chart.
    """
    # Extract and scale relevant current features
    try:
        live_data_features = live_data[["Current(PhaseA) [A]", "Current(PhaseB) [A]", "Current(PhaseC) [A]"]]
    except KeyError as e:
        raise KeyError(f"Missing expected column in input data: {e}")

    live_data_scaled = scaler.transform(live_data_features)

    # Predict severity scores
    predictions = model.predict(live_data_scaled)

    # Apply threshold to classify
    healthy_samples = np.sum(predictions < SEVERITY_THRESHOLD)
    faulty_samples = len(predictions) - healthy_samples

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.bar(["Healthy", "Faulty"], [healthy_samples, faulty_samples], color=['green', 'red'], alpha=0.7)
    plt.xlabel("Sample Type", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(f"Sample Distribution: Healthy vs Faulty (Threshold = {SEVERITY_THRESHOLD})", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars
    counts = [healthy_samples, faulty_samples]
    for index, value in enumerate(counts):
        plt.text(index, value + max(counts) * 0.02, str(value), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    try:
        live_data = pd.read_csv(DATA_FILE)
        print("📂 Live data loaded successfully.")
        classify_health_status(live_data)
    except FileNotFoundError:
        print(f"❌ File '{DATA_FILE}' not found. Please check the path and filename.")
