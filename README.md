# demag_predict_IPMSM
This repository contains the source code, simulation data, and machine learning models developed for early-stage demagnetisation fault detection in Interior Permanent Magnet Synchronous Motors (IPMSM). The project leverages ANSYS Twin Builder simulations to generate datasets, which are then used to train machine learning models for classifying and predicting fault severity levels.

### Project Overview
Objective: To detect and classify uniform demagnetisation faults in an IPMSM using machine learning algorithms.

### Approach:
- Simulate the motor under various demagnetisation conditions using ANSYS MotorCAD
- Extract relevant features from current, voltage, and speed waveforms.
- Train and evaluate ML models for early and accurate fault prediction.
- Fault Levels Simulated: 0% (Healthy), 5%, 10%, 15%, 20%, and 25% demagnetisation.

### Machine Learning Techniques
We explored several supervised ML algorithms:
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
Performance was evaluated using accuracy, precision, recall, and F1-score metrics.

### Tools & Technologies
- Simulation: ANSYS Twin Builder 2024 R1
- Programming Language: Python 3.x
- Libraries: Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- Environment: Jupyter Notebook, Google Colab

### Results
Our models achieved over 95% accuracy in classifying the different demagnetisation levels, showing strong potential for real-time fault diagnostics in electric vehicle applications and industrial motor systems.
