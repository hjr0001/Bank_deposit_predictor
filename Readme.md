# 🏦 Bank Term Deposit Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bankdepositpredictor-bfn6qdssnh25qr8q5bveyk.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

An end-to-end machine learning web application that predicts whether a bank client will subscribe to a term deposit based on demographic, financial, and marketing campaign data.

---

## 🚀 Live Demo

Explore the interactive dashboard and test real-time predictions here:  
👉 **[Bank Deposit Predictor — Streamlit Dashboard](https://bankdepositpredictor-bfn6qdssnh25qr8q5bveyk.streamlit.app/)**

---

## 📌 Project Overview

Direct marketing campaigns (phone calls) are a primary method used by banking institutions to offer term deposits. However, reaching out to every customer is resource-intensive. 

This project analyses historical client and campaign data to build a classification model capable of identifying high-probability subscribers, helping banks optimise their marketing efforts and improve conversion rates.

### Key Features
* **Interactive Web Interface:** Built with Streamlit to allow users to input client parameters and get instant subscription predictions.
* **Data Preprocessing Pipeline:** Handles categorical encoding, feature scaling, and class imbalance.
* **Machine Learning Modelling:** Evaluates classification algorithms to maximise predictive accuracy and recall on positive term deposit subscriptions.

---

## 📊 Dataset Description

The model is trained on bank marketing campaign data, incorporating the following key features:

* **Client Demographics:** Age, job type, marital status, education level.
* **Financial Profile:** Account balance, housing loan status, personal loan status, credit in default.
* **Campaign Interactions:** Contact communication type, last contact day/month, call duration, number of contacts performed during the campaign (`campaign`), days passed since last contact (`pdays`), and outcome of the previous marketing campaign (`poutcome`).
* **Target Variable (`y` / `deposit`):** Whether the client subscribed to a term deposit (`Yes` / `No`).

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualisation:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn

---

## 💻 Local Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hjr0001/Bank_deposit_predictor.git
   cd Bank_deposit_predictor
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   *(Note: Replace `app.py` with the main Python script name if yours differs).*

---

## 📁 Repository Structure

```text
Bank_deposit_predictor/
├── app.py                  # Main Streamlit web application script
├── requirements.txt        # Python dependencies required for deployment
├── model/                  # Saved/serialised machine learning model & scalers (.pkl / .joblib)
├── notebooks/              # Jupyter notebooks for EDA and model training
├── data/                   # Dataset files
└── README.md               # Project documentation
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/hjr0001/Bank_deposit_predictor/issues) or submit a pull request.

## 👤 Author

* **GitHub:** [@hjr0001](https://github.com/hjr0001)