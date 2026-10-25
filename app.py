import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Bank Telemarketing ROI Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Predictive Lead Targeting & ROI Dashboard")
st.markdown("Optimize call center resources by adjusting the model confidence threshold and analyzing financial performance in real-time.")

# Load and train cached model & data
@st.cache_data
def load_and_train_model():
    bank_marketing = fetch_ucirepo(id=222)
    X = pd.DataFrame(bank_marketing.data.features)
    y = pd.DataFrame(bank_marketing.data.targets)

    if 'duration' in X.columns:
        X.drop(columns=['duration'], inplace=True)
    if isinstance(y, pd.DataFrame) and 'y' in y.columns:
        y = y['y'].map({'yes': 1, 'no': 0})

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    negative_cases = (y_train == 0).sum()
    positive_cases = (y_train == 1).sum()
    imbalance_ratio = negative_cases / positive_cases

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=imbalance_ratio,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    return X_test, y_test, y_proba

with st.spinner("Fetching data and training XGBoost model... Please wait."):
    X_test, y_test, y_proba = load_and_train_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Simulation Controls")
threshold = st.sidebar.slider("Conversion Probability Threshold", 0.0, 1.0, 0.70, 0.01)
cost_per_call = st.sidebar.number_input("Cost per Call ($)", value=5, min_value=1)
revenue_per_sub = st.sidebar.number_input("Revenue per Subscription ($)", value=50, min_value=1)

# --- BUSINESS SIMULATION LOGIC ---
y_pred_custom = (y_proba >= threshold).astype(int)

# Scenario A: Baseline (Call Everyone)
calls_made_baseline = len(y_test)
subs_gained_baseline = int(y_test.sum())
profit_baseline = (subs_gained_baseline * revenue_per_sub) - (calls_made_baseline * cost_per_call)

# Scenario B: Model Targeted
calls_made_model = int(y_pred_custom.sum())
subs_gained_model = int(((y_test == 1) & (y_pred_custom == 1)).sum())
profit_model = (subs_gained_model * revenue_per_sub) - (calls_made_model * cost_per_call)

call_reduction_pct = 100 - (calls_made_model / calls_made_baseline * 100) if calls_made_baseline > 0 else 0
net_financial_impact = profit_model - profit_baseline
conversion_rate = (subs_gained_model / calls_made_model * 100) if calls_made_model > 0 else 0

# --- MAIN DASHBOARD KPIS ---
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Profit (Model)", f"${profit_model:,.2f}", delta=f"${net_financial_impact:,.2f} vs Baseline")
with col2:
    st.metric("Calls Made", f"{calls_made_model:,}", delta=f"-{call_reduction_pct:.1f}% volume", delta_color="inverse")
with col3:
    st.metric("Subscriptions Gained", f"{subs_gained_model:,}", delta=f"out of {calls_made_model} calls")
with col4:
    st.metric("Targeted Conversion Rate", f"{conversion_rate:.1f}%")

st.markdown("---")

# --- COMPARISON TABLE ---
st.subheader("📋 Scenario Breakdown & Comparison")
comparison_data = {
    "Metric": [
        "Total Outbound Calls", 
        "Successful Subscriptions", 
        "Total Cost of Calls", 
        "Total Revenue Generated", 
        "Net Profit"
    ],
    "Scenario A (Call Everyone / Baseline)": [
        f"{calls_made_baseline:,}",
        f"{subs_gained_baseline:,}",
        f"${calls_made_baseline * cost_per_call:,.2f}",
        f"${subs_gained_baseline * revenue_per_sub:,.2f}",
        f"${profit_baseline:,.2f}"
    ],
    f"Scenario B (Model @ {int(threshold*100)}% Threshold)": [
        f"{calls_made_model:,}",
        f"{subs_gained_model:,}",
        f"${calls_made_model * cost_per_call:,.2f}",
        f"${subs_gained_model * revenue_per_sub:,.2f}",
        f"${profit_model:,.2f}"
    ]
}

st.table(pd.DataFrame(comparison_data))
