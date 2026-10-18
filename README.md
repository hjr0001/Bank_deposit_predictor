# Bank_deposit_predictor
📈 Predictive Lead Targeting for Bank Marketing
An end-to-end machine learning pipeline and interactive dashboard designed to optimize call center operations for bank term-deposit campaigns. By calibrating an imbalanced XGBoost model to a custom high-confidence decision threshold, this system cuts outbound call volume by over 91% while delivering +$8,775 in net financial lift.
🎯 Business Problem
Financial telemarketing campaigns typically call customer lists indiscriminately or rely on default 0.50 probability cutoffs. This produces heavy agent fatigue, high calling costs, and low conversion rates.
	•	Objective: Predict which leads are most likely to subscribe to a term deposit before picking up the phone. 
	•	Resource Optimization: Prioritize premium leads to minimize wasted outbound agent hours and maximize campaign net profit. 
🛠️ Technical Pipeline & Architecture
	•	Dataset: UCI Bank Marketing Repository (ID: 222). 
	•	Data Leakage Prevention: The duration attribute was intentionally excluded prior to modeling because call duration cannot be known prior to placing a call. 
	•	Preprocessing: ColumnTransformer handling numerical imputation (median) and standardization (StandardScaler), alongside categorical imputation (most_frequent) and one-hot encoding (OneHotEncoder). 
	•	Class Imbalance: Handled by dynamically calculating and injecting scale_pos_weight = 7.55 directly into the XGBoost loss function. 
	•	Hyperparameter Tuning: Tuned with RandomizedSearchCV (3-fold cross-validation) optimizing for Average Precision (AUPRC) rather than ROC-AUC or standard accuracy. 
	•	Explainability: Model decisions audited globally using SHAP (TreeExplainer) summary plots. 
	•	Interactive Dashboard: Built with Plotly Dash to simulate call center ROI and conversion tradeoffs across decision thresholds dynamically. 
📊 Key Results & Business Impact
Targeting was calibrated to a 0.70 confidence threshold to isolate high-conviction leads:
Metric
Scenario A: Call Everyone (Baseline) IPYNB
Scenario B: Model-Targeted (0.70 Threshold) IPYNB
Net Difference / Lift IPYNB
Calls Placed
9,043
798
-91.2% call volume

Subscriptions Won
1,058
409
409 converted leads
Targeting Precision
11.7%
51.0%

+39.3% precision
Call Center Cost ($5/call)

$45,215.00
$3,990.00
$41,225.00 saved
Gross Revenue ($50/sub)

$52,900.00
$20,450.00
—
Net Campaign Profit
$7,685.00
$16,460.00

+$8,775.00 net lift

🔍 Model Explainability (SHAP Insights)
SHAP feature attribution surfaced three core recommendations for outbound sales operations:
	1	Target Debt-Free Leads: Customers without existing housing loans (housing_no) show the highest baseline probability of subscribing. 
	2	Capitalize on Past Campaign Success: Leads who converted in previous marketing outreach (poutcome_success) are statistically the strongest converters. 
	3	Prevent Call Fatigue: High values for campaign (too many contacts within a single campaign cycle) trigger an immediate, sharp drop-off in conversion probability. 
🖥️ Interactive Dashboard (app.py)
The included Dash application allows sales managers and stakeholders to:
	•	Adjust the conversion probability threshold via slider ($0.10$ to $0.90$). 
	•	Test variable economic assumptions (Cost per Call and Revenue per Subscription). 
	•	View real-time Confusion Matrices, Net Profit Lift bar charts, and KPI metric cards. 
