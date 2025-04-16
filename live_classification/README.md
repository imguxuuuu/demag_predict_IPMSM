# ⚙️ Live Health Classification: IPMSM Fault Detection

This module provides a real-time classification system that identifies **Healthy** vs **Faulty** samples in an Interior Permanent Magnet Synchronous Motor (IPMSM) based on live current input data.

It uses a pre-trained regression model to estimate demagnetization severity and then classifies each sample based on a user-defined threshold.

---

## 🎯 Objective

Classify live sensor data into:

- ✅ **Healthy**: If predicted severity < threshold
- ❌ **Faulty**: If predicted severity ≥ threshold

It also visualizes the count of healthy and faulty samples in a bar chart.

---

## 📁 Folder Structure


---

## 📥 Input

CSV file with the following columns:

| Column Name             | Description             |
|-------------------------|-------------------------|
| Current(PhaseA) [A]     | Current in Phase A      |
| Current(PhaseB) [A]     | Current in Phase B      |
| Current(PhaseC) [A]     | Current in Phase C      |


## 📊 Output
A bar chart showing the number of samples predicted as:

Healthy (green)

Faulty (red)

Also, the count of each category is printed on top of the bars.

## ⚙️ How It Works
Standardizes the input using a pre-fitted scaler.

Predicts severity using a pre-trained model.

Applies a threshold to classify samples.

Displays a bar chart of the results.

## 🚀 How to Run
🔧 Make sure to load the model and scaler objects (model, scaler) before calling the function.
```
# Load your model and scaler beforehand
import joblib
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Then call the script
python classify_live_data.py
```
## 🛠 Requirements
Install dependencies with:
```
pip install numpy pandas scikit-learn matplotlib
```
