# 🔍 Feature Pattern Visualization for Demagnetization Severity

This project provides a Python script to visualize how different time-domain features of IPMSM (Interior Permanent Magnet Synchronous Motor) phase currents vary with **demagnetization severity**. It's designed to support early fault detection and pattern recognition in machine learning applications.

---

## 📁 Repository Structure


---

## 📈 What Does It Do?

The script reads a CSV file containing:
- Statistical features (`mean`, `std`, `min`, `max`, `rms`) for phase current signals (A, B, C)
- A `Severity` column indicating the level of demagnetization (e.g., 0%, 5%, ...)

It generates line plots showing how each feature changes across severity levels for a selected phase.

---


## ▶️ How to Run

1. **Place your CSV file** in the `data/` folder.
2. **Edit the script config** (optional):
   - Change `SELECTED_PHASE` to `"A"`, `"B"`, or `"C"`
   - Change the `DATA_FILE` path if needed

3. **Run the script**:

```bash
python plot_feature_patterns.py
```
## 📌 Requirements
Python 3.7+

Libraries:

- pandas

- matplotlib

Install dependencies using:
```
pip install pandas matplotlib
```
