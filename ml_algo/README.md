# 📊 ML Model Evaluation for IPMSM Demagnetization

This module evaluates multiple machine learning regression models to estimate **demagnetization severity** based on synthetic current signal data. It mimics the behavior of an Interior Permanent Magnet Synchronous Motor (IPMSM) under varying load/fault conditions.

---

## 🧠 Models Evaluated

The following models are trained and compared:

- ✅ Gradient Boosting Regressor
- ✅ Random Forest Regressor
- ✅ Linear Regression
- ✅ Support Vector Regressor (SVR)
- ✅ K-Nearest Neighbors Regressor

Each model is trained using **polynomial-transformed**, **standardized** input data for improved accuracy.

---

## 🧪 Dataset

The script uses a **synthetic dataset** generated on the fly with the following features:

| Feature               | Description                  |
|-----------------------|------------------------------|
| Current(PhaseA) [A]   | Phase A current              |
| Current(PhaseB) [A]   | Phase B current              |
| Current(PhaseC) [A]   | Phase C current              |

The **target (y)** is a severity value derived from the **sum of phase currents**, with added noise to simulate real-world conditions.

---

## 📈 Metrics Reported

After training, each model is evaluated using the following regression metrics:

- 📉 Mean Squared Error (MSE)
- 📉 Mean Absolute Error (MAE)
- 📈 R² Score (coefficient of determination)

Results are printed in a neatly formatted table using `prettytable`.

---

## 🚀 How to Run

1. Navigate to the `evaluation/` directory.
2. Run the script using:

```bash
python model_evaluation.py
