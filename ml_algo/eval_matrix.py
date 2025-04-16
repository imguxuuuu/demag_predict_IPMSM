"""
model_evaluation.py

This script evaluates multiple machine learning models using synthetic current data,
simulating demagnetization severity prediction for an IPMSM motor.

Author: Gururaghuraman
Repository: https://github.com/imguxuuuu/demag_predict_IPMSM
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable

# ---- 1. Generate synthetic dataset with correlated target ----
np.random.seed(42)
X = pd.DataFrame(np.random.rand(1200, 3), columns=["Current(PhaseA) [A]", "Current(PhaseB) [A]", "Current(PhaseC) [A]"])
y = X.sum(axis=1) * 10 + np.random.normal(0, 2, size=len(X))  # severity ~ sum of currents + noise

# ---- 2. Data splitting and scaling ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 3. Polynomial feature expansion ----
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# ---- 4. ML Models ----
models = {
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=4),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200),
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor (SVR)": SVR(kernel='rbf'),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5)
}

# ---- 5. Train & Evaluate ----
results = []
for name, model in models.items():
    print(f"Training model: {name}")
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    
    results.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R² Score": max(r2_score(y_test, y_pred), 0)
    })

# ---- 6. Display Results ----
table = PrettyTable(["Model", "MSE", "MAE", "R² Score"])
for res in results:
    table.add_row([res["Model"], f"{res['MSE']:.3f}", f"{res['MAE']:.3f}", f"{res['R² Score']:.3f}"])

print("\nEvaluation Metrics for ML Models:")
print(table)
