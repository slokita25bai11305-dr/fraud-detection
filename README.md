# fraud-detection
# 🔍 Fraud Detection — AI & ML Project

A complete end-to-end machine learning pipeline that detects fraudulent financial transactions using supervised learning. The project covers synthetic data generation, exploratory data analysis (EDA), preprocessing, model training (4 algorithms), and visual evaluation — all from a single Python script.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo Output](#demo-output)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Results](#model-results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## Overview

Fraud detection is one of the most critical applications of machine learning in the financial industry. This project simulates a real-world fraud detection system by:

- Generating a realistic dataset of **12,000 bank transactions** (7% fraudulent, 93% legitimate)
- Performing **exploratory data analysis** with 12 visualizations
- Applying **preprocessing** techniques including log-transformation, oversampling, and feature scaling
- Training and comparing **4 ML models**: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- Evaluating each model using **ROC-AUC, Precision-Recall, Confusion Matrix**, and threshold analysis

> This project is ideal for students and developers learning applied machine learning, data science pipelines, or fraud analytics.

---

## Demo Output

The script automatically generates three chart files when run:

| File | Description |
|------|-------------|
| `eda.png` | 12-panel exploratory data analysis dashboard |
| `results.png` | Model evaluation dashboard (ROC curves, confusion matrices, feature importance) |
| `scores.png` | Fraud probability distribution and threshold analysis |

---

## Project Structure

```
fraud-detection/
│
├── fraud_detection.py      # Main script — run this
├── eda.png                 # Generated: EDA charts
├── results.png             # Generated: Model evaluation charts
├── scores.png              # Generated: Score distribution charts
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## Features

- **Synthetic data generation** — Realistic transaction data with engineered fraud signals (night-time activity, foreign transactions, high credit utilization, new devices)
- **EDA dashboard** — Distribution plots, correlation heatmap, boxplots, binary feature comparisons, and a dataset summary table
- **Class imbalance handling** — Oversampling the minority (fraud) class to 33% of training data using `sklearn.utils.resample`
- **4 trained models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Comprehensive evaluation** — ROC curves, Precision-Recall curves, confusion matrices, cross-validation scores
- **Threshold analysis** — Finds the optimal decision threshold that maximises F1-Score
- **Feature importance** — Random Forest feature importance bar chart
- **Dark-themed visualisations** — Professional publication-quality charts

---

## Prerequisites

Make sure you have the following installed before proceeding:

- **Python 3.9 or higher** — [Download here](https://www.python.org/downloads/)
- **pip** — comes bundled with Python
- **VS Code** (recommended) — [Download here](https://code.visualstudio.com/)
  - Install the **Python** extension by Microsoft inside VS Code

To verify Python is installed, open a terminal and run:

```bash
# On Windows
python --version

# On macOS / Linux
python3 --version
```

> If you see `command not found`, visit the [Python installation guide](https://www.python.org/downloads/) and ensure you check **"Add Python to PATH"** during setup on Windows.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

### Run the full pipeline

```bash
# Windows
python fraud_detection.py

# macOS / Linux
python3 fraud_detection.py
```

### Expected terminal output

```
============================================================
   FRAUD ANALYSIS PROJECT — AI & ML PIPELINE
============================================================

[1] Generating synthetic transaction dataset...
    Dataset shape : (12000, 11)
    Fraud cases   : 840 (7.0%)

[2] Running Exploratory Data Analysis...
  [✓] EDA saved → eda.png

[3] Preprocessing & balancing data...
    Train shape: (13392, 10)  |  Test shape: (2400, 10)

[4] Training models...
  [Logistic Regression]  ROC-AUC=1.0000  AP=0.9998  CV-AUC=1.0000
  [Decision Tree]        ROC-AUC=0.9851  AP=0.9723  CV-AUC=0.9996
  [Random Forest]        ROC-AUC=1.0000  AP=1.0000  CV-AUC=1.0000
  [Gradient Boosting]    ROC-AUC=0.9999  AP=0.9991  CV-AUC=1.0000

[5] Plotting evaluation dashboard...
  [✓] Results saved → results.png

[6] Plotting score distribution...
  [✓] Score distribution saved → scores.png

============================================================
   BEST MODEL : Random Forest
   ROC-AUC    : 1.0000
   CV-AUC     : 1.0000
============================================================
```

Three `.png` files will be saved in the same folder as the script.

---

## How It Works

### 1. Data Generation

Two separate populations of transactions are created using NumPy:

| Feature | Legitimate | Fraudulent |
|---------|-----------|------------|
| Transaction amount | Small–medium (log-normal) | Large, erratic |
| Transaction hour | 6 AM – 11 PM | Midnight – 6 AM |
| Distance from home | ~10 km avg | ~80 km avg |
| Foreign transaction | 5% rate | 60% rate |
| New device | 8% rate | 75% rate |
| Credit utilization | ~35% avg | ~85% avg |
| Account age | 30–3650 days | 1–400 days |
| Failed login attempts | ~0.2 avg | ~2.5 avg |

### 2. Preprocessing Pipeline

```
Raw data
   │
   ├── Log-transform skewed features (amount, distance, avg_txn_30d)
   │
   ├── Stratified Train/Test Split (80% / 20%)
   │
   ├── Oversample minority class (fraud → 33% of training set)
   │
   └── StandardScaler (zero mean, unit variance)
```

### 3. Model Training

Four models are trained on the balanced, scaled training data:

- **Logistic Regression** — Linear baseline, fast and interpretable
- **Decision Tree** — Rule-based splits, max_depth=8
- **Random Forest** — 100 trees with bootstrap sampling and class_weight="balanced"
- **Gradient Boosting** — Sequential error correction, learning_rate=0.1

### 4. Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| ROC-AUC | Ability to rank fraud above legitimate transactions |
| Average Precision | Precision-Recall curve summary (better for imbalanced data) |
| Cross-Val AUC | Generalisation estimate across 5 folds |
| Confusion Matrix | TP, TN, FP, FN breakdown |
| F1-Score | Harmonic mean of Precision and Recall |

---

## Model Results

| Model | ROC-AUC | Avg Precision | CV-AUC |
|-------|---------|--------------|--------|
| Random Forest | **1.000** | **1.000** | 1.000 |
| Logistic Regression | 1.000 | 0.9998 | 1.000 |
| Gradient Boosting | 0.9999 | 0.9991 | 1.000 |
| Decision Tree | 0.985 | 0.972 | 0.9996 |

> **Note:** Near-perfect scores are expected with synthetic data that has strong, clean signals. On real-world datasets (e.g. Kaggle Credit Card Fraud), Random Forest typically achieves **0.95–0.98 ROC-AUC**.

---

## Technologies Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Core language |
| NumPy | latest | Data generation, array math |
| Pandas | latest | DataFrames, train/test split |
| Matplotlib | latest | All visualisations |
| Seaborn | latest | Heatmap, styled plots |
| Scikit-learn | latest | ML models, metrics, preprocessing |

---

## Future Improvements

- [ ] Replace synthetic data with a real dataset (e.g. [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- [ ] Add SMOTE oversampling using `imbalanced-learn`
- [ ] Add XGBoost and LightGBM models
- [ ] Build a Flask/FastAPI REST API to serve predictions
- [ ] Add SHAP values for individual prediction explainability
- [ ] Convert to a Jupyter Notebook for interactive exploration
- [ ] Dockerise the project for portable deployment

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

SLOKITA GHOSH
25BAI11305
- GitHub: (https://github.com/slokita25bai11305-dr)
- LinkedIn: (https://www.linkedin.com/in/slokita-ghosh-a62666248?utm_source=share_via&utm_content=profile&utm_medium=member_ios)

---

> If you found this project helpful, please consider giving it a ⭐ on GitHub!
