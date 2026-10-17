# Bank_deposit_predictor
# Predictive Lead Targeting for Bank Marketing

## Business Objective
The goal of this project is to optimize call center resources for a financial institution. By predicting which customers are most likely to convert and subscribe to a term deposit, the marketing team can prioritize high-probability leads, reducing wasted agent hours and increasing net profit.

## Technical Approach
* **Algorithm:** Built an XGBoost classification pipeline.
* **Data Integrity:** Explicitly removed the 'duration' feature to prevent data leakage (as call duration is unknown before a call is made).
* **Imbalance Handling:** Dynamically calculated and applied `scale_pos_weight` to handle the highly imbalanced dataset.
* **Optimization:** Utilized `RandomizedSearchCV` to tune hyperparameters (n_estimators, max_depth, learning_rate) optimizing for Average Precision (AUPRC).

## Business Impact & ROI
Rather than using a default 0.50 probability threshold, the model was calibrated to a **0.70 confidence threshold** to target premium leads. 

*Simulation Results:*
* Reduced total outbound call volume by **[Insert % from your script]**%.
* Maintained high conversion rates among targeted customers.
* Generated a net positive financial impact of **$[Insert amount from your script]** compared to a randomized calling strategy, assuming a $5 cost per call and $50 revenue per subscription.

## Key Drivers (SHAP Analysis)
![SHAP Summary Plot](shap_summary.png)

Based on the SHAP feature importance analysis, the top recommendations for the marketing team are:
1. **Target "Debt-Free" Customers:** Customers without existing housing loans (`housing_no`) have the highest probability of conversion.
2. **Prioritize Previous Successes:** Customers who converted in previous campaigns (`poutcome_success`) are highly likely to convert again.
3. **Avoid Call Fatigue:** The model shows a sharp drop-off in probability if a customer is contacted too many times during a single campaign.
