"""
plot_feature_patterns.py

📊 Visualize how different statistical features of IPMSM phase currents vary with demagnetization severity.

This script helps to understand the behavior of key time-domain features (mean, std, min, max, rms)
for a selected motor phase (A, B, or C) against severity levels.

Author: Gururaghuraman
Date: 2025-01-25
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

DATA_FILE = "train_data.csv"  # Replace with the actual filename
SELECTED_PHASE = "A"          # Choose from "A", "B", or "C"
FEATURE_TYPES = ["mean", "std", "min", "max", "rms"]

# --- LOAD DATA ---

try:
    train_data = pd.read_csv(DATA_FILE)
    print("✅ Training data loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ '{DATA_FILE}' not found. Please ensure the file exists in your directory.")

# --- PLOT FEATURE PATTERNS FOR SELECTED PHASE ---

def plot_phase_features(phase: str):
    """
    Plots the variation of statistical features for the selected phase
    against demagnetization severity.
    """
    plt.figure(figsize=(8, 5))

    for feature in FEATURE_TYPES:
        feature_column = f"Current(Phase{phase}) [A]_{feature}"
        if feature_column not in train_data.columns:
            print(f"⚠️ Warning: Column '{feature_column}' not found in dataset.")
            continue
        plt.plot(train_data["Severity"], train_data[feature_column], marker='o', linestyle='-', label=feature)

    plt.xlabel("Demagnetization Severity (%)", fontsize=12)
    plt.ylabel("Feature Value", fontsize=12)
    plt.title(f"Feature Patterns for Phase {phase}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    plot_phase_features(SELECTED_PHASE)
