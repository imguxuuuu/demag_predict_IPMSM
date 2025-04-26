# 🚘 Demagnetization Fault Prediction in IPMSM using Machine Learning

This project focuses on early-stage detection and classification of **demagnetization faults in Interior Permanent Magnet Synchronous Motors (IPMSM)** using machine learning. It includes data preparation, model evaluation, live classification, and visualization modules.

---

## 🧠 Project Overview

**Goal**: Predict severity levels of demagnetization faults and classify motor health status based on phase current data. The approach is tested on simulated data and generalizes to unseen/livestream machine data.


---

## ⚙️ Modules

### 1. 📊 `eval_matrix.py`

Trains and evaluates 5 ML regressors on synthetic current data:

- Gradient Boosting Regressor
- Random Forest Regressor
- Linear Regression
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

Generates a **PrettyTable** showing each model's:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

📥 *Input*: Synthetic data with correlated severity  
📤 *Output*: Printed evaluation matrix



### 2. 🧪 `live_classify.py`

Classifies live data samples as **Healthy** or **Faulty** based on predicted severity levels.

- Uses pre-trained `model` and `scaler`
- Threshold-based classification (e.g., ≥ 5% severity → Faulty)
- Visual output: Bar chart of healthy vs faulty samples

📥 *Input*: `live_machine_data.csv`  
📤 *Output*: Classification plot



### 3. 📈 `plot_feature.py`

Visualizes statistical trends of current features (`mean`, `std`, `min`, `max`, `rms`) across severity levels.

- Helps identify which features are most sensitive to demagnetization
- Can be run for any current phase (A, B, or C)

📥 *Input*: `train_data.csv`  
📤 *Output*: Line plot for selected features

### 4. 🔮 `predict/`

Predicts the severity of demagnetization faults in IPMSM based on machine learning models trained on simulation data.

Uses multiple machine learning algorithms for fault prediction (e.g., Random Forest, SVM, Neural Networks).

Predicts fault severity based on various input features (e.g., current, voltage, speed, temperature).

Outputs predicted fault severity along with confidence scores.

📥 Input: new_machine_data.csv (current, voltage, speed, temperature)
📤 Output: Predicted fault severity and confidence scores in a CSV file



---

## 🧰 Requirements

- Python 3.7+
- Libraries:
  ```bash
  pip install numpy pandas matplotlib scikit-learn prettytable
  ```

---

## ▶️ How to Run
Prepare Datasets:

Place train_data.csv and live_machine_data.csv inside the data/ folder.

Run Each Module:
```
python evaluation_matrix.py
python classify_health_status.py
python plot_feature_patterns.py
```

Customize as Needed:
- Adjust severity thresholds
- Change phases in plots

---

## 🧠 Use Cases

- Predictive maintenance for electric motors
- Health monitoring in EVs and industrial drives
- Feature selection for fault classification
- Educational demo for ML in electrical systems

---

## 👨‍🔬 Contributors
The project was developed under the guidance of Dr. Praveenkumar N (Amrita School of Engineering, India)

By: 
1. Gururaghuraman S – BTech in Electrical and Electronics Engineering
2. Hari Venkatesh S – BTech in Electrical and Electronics Engineering
3. Manuprabha R – BTech in Electrical and Electronics Engineering
4. Renill H – BTech in Electrical and Electronics Engineering
