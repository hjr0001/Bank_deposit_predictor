# Predictive Lead Targeting for Bank Telemarketing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-blueviolet.svg)](https://dash.plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning and operational intelligence system designed to optimize outbound call center campaigns for bank term deposit subscriptions. By prioritizing high-propensity leads and eliminating operational data leakage, the solution cuts outbound call volumes by over **90%** and maximizes net campaign profitability.

---

## Table of Contents

- [Business Overview & Problem Statement](#business-overview--problem-statement)
- [Key Results & Financial Impact](#key-results--financial-impact)
- [System Architecture & Methodology](#system-architecture--methodology)
- [Data Integrity & Leakage Prevention](#data-integrity--leakage-prevention)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [Model Explainability (SHAP Analysis)](#model-explainability-shap-analysis)
- [Interactive Dash Application](#interactive-dash-application)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [License](#license)

---

## Business Overview & Problem Statement

Financial institutions routinely conduct outbound telemarketing campaigns to cross-sell term deposits. Traditional marketing strategies typically rely on:
1. Contacting every customer in a broad contact registry (**Call Everyone**).
2. Heuristic, uncalibrated calling lists with poor conversion efficiency.

### Core Challenges:
- **High Operational Costs:** Agent talk-time and call infrastructure consume operational budget (\$5.00/call).
- **Severe Class Imbalance:** Term deposit conversion rates are inherently low (~11.7% in the UCI Bank Marketing dataset), meaning roughly 9 out of 10 calls yield no conversion.
- **Customer Fatigue:** Over-contacting unpromising leads damages customer relations and brand goodwill.

### Project Objective:
Formulate an operational decision pipeline that flags high-probability conversion candidates *before* a call is initiated, enabling data-driven lead dispatch and measurable net profit lift.

---

## Key Results & Financial Impact

Using the holdout test set ($N = 9,043$ leads, $1,058$ actual conversions) and standard unit economics:
- **Cost per Call:** $\$5.00$
- **Revenue per Conversion:** $\$50.00$

| Metric | Scenario A: Blanket Calling (Baseline) | Scenario B: Model-Targeted Strategy (0.70 Threshold) | Net Delta / Lift |
| :--- | :---: | :---: | :---: |
| **Calls Placed** | 9,043 | **798** | **-91.2% call reduction** |
| **Subscribers Acquired** | 1,058 | 409 | *Focused premium subset* |
| **Gross Campaign Costs** | \$45,215.00 | **\$3,990.00** | **\$41,225.00 saved** |
| **Gross Revenue** | \$52,900.00 | \$20,450.00 | — |
| **Net Campaign Profit** | \$7,685.00 | **\$16,460.00** | **+\$8,775.00 (+114.2% lift)** |

*By targeting only leads crossing a 70% confidence threshold, the bank achieves more than double the net profit while eliminating over 8,200 non-productive calls.*

---

## System Architecture & Methodology

```
UCI Repository (ID 222)
          │
          ▼
Data Sanitization ──────────► [CRITICAL] Drop `duration` feature
          │
          ▼
Preprocessing Pipeline
   ├── Numeric: Median Imputer + StandardScaler
   └── Categorical: Most-Frequent Imputer + OneHotEncoder
          │
          ▼
Class Imbalance Balancing ──► Dynamic `scale_pos_weight` calculation (~7.55)
          │
          ▼
Hyperparameter Tuning ─────► RandomizedSearchCV (Stratified 3-fold, scoring='average_precision')
          │
          ▼
Calibrated XGBoost Model
          │
          ├── Probabilistic Decision Thresholding (τ = 0.70)
          ├── SHAP TreeExplainer Global & Local Feature Insights
          └── Interactive Operational Dashboard (Plotly Dash)
```

---

## Data Integrity & Leakage Prevention

The UCI Bank Marketing dataset includes a `duration` attribute (call duration in seconds). In many published models, this attribute falsely inflates predictive performance (AUC > 0.93) because long calls naturally correlate with conversion.

> **Operational Safeguard:** 
> Call duration is **unknown before a call is made**. Retaining `duration` causes severe target leakage. In this pipeline, `duration` is programmatically removed prior to training to ensure real-world deployability.

---

## Model Performance & Evaluation

Because telemarketing data is heavily skewed, models were evaluated and tuned via **Average Precision (AUPRC)** rather than accuracy or standard ROC-AUC.

### Hyperparameters (Optimized):
- `n_estimators`: 300
- `learning_rate`: 0.05
- `max_depth`: 5
- `subsample`: 0.80
- `colsample_bytree`: 1.00
- `scale_pos_weight`: ~7.55

### Classification Metrics Comparison:

#### 1. Default Decision Threshold ($\tau = 0.50$)
```text
                 precision    recall  f1-score   support
No Subscription       0.94      0.87      0.90      7985
   Subscription       0.36      0.56      0.44      1058

       accuracy                           0.83      9043
      macro avg       0.65      0.71      0.67      9043
   weighted avg       0.87      0.83      0.85      9043
```

#### 2. Calibrated Lead Generation Threshold ($\tau = 0.70$)
```text
                 precision    recall  f1-score   support
No Subscription       0.92      0.95      0.94      7985
   Subscription       0.51      0.39      0.44      1058

       accuracy                           0.89      9043
      macro avg       0.72      0.67      0.69      9043
   weighted avg       0.87      0.89      0.88      9043
```

Raising the cut-off to 0.70 increases precision from 36% to **51%**, ensuring outbound sales agents spend their time almost exclusively with warm leads.

---

## Model Explainability (SHAP Analysis)

Global feature importance and directional attribution were generated via `shap.TreeExplainer`:

1. **Target Debt-Free Prospects:** Customers without an existing housing loan (`housing_no`) exhibit the single highest positive SHAP value for conversion.
2. **Prioritize Prior Campaign Successes:** Customers who agreed to past campaign offers (`poutcome_success`) have high recurring subscription probabilities.
3. **Control Campaign Frequency (`campaign`):** Increased contact counts during a single campaign cause a steep downward SHAP impact, demonstrating that over-solicitation creates diminishing returns.
4. **Account Balance & Age:** Higher account balances and older demographics (retirees) show positive lift towards term deposit conversion.

---

## Interactive Dash Application

An interactive web application (`app.py`) is included to simulate ROI scenarios and inspect feature attributions in real time.

### Key Features:
- **Threshold Slider ($\tau = 0.10 \to 0.90$):** Dynamically adjusts confidence cut-offs and recomputes confusion matrices and precision/recall.
- **Dynamic Cost/Revenue Simulator:** Configure variable cost per call and revenue per subscription to model diverse operational situations.
- **KPI Summary Cards:** Real-time visibility into net profit lift, call reductions, and conversions won/missed.
- **SHAP Drivers & Strategic Recommendations:** Inline operational guidance for sales leadership.

---

## Project Structure

```text
.
├── bank_marketing_targeting.ipynb  # Full Colab / Jupyter notebook pipeline
├── app.py                         # Interactive Plotly Dash ROI dashboard
├── model_test_predictions.csv      # (Optional) Exported test set true values & probabilities
├── shap_summary.png               # SHAP TreeExplainer feature importance plot
├── requirements.txt               # Environment dependencies
└── README.md                      # Project documentation
```

---

## Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bank-marketing-targeting.git
cd bank-marketing-targeting
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Or install core libraries manually:)*
```bash
pip install ucimlrepo xgboost scikit-learn pandas numpy shap dash plotly
```

### 4. Run the Pipeline & App
To run the notebook:
```bash
jupyter notebook bank_marketing_targeting.ipynb
```

To run the interactive Dash dashboard:
```bash
python app.py
```

---

