# 💳 Credit Card Consumption Prediction

This project predicts missing values of customer credit card consumption (`cc_cons`) using demographic and transactional behavior data. It is a complete case study in data preprocessing, feature engineering, model training, and evaluation.
## ✅ Problem Statement

We are given data for 20,000 customers:
- For 15,000 customers, the credit card consumption (`cc_cons`) is known.
- For 5,000 customers, this value is missing.

🎯 **Goal**: Train a model to accurately predict `cc_cons` for the missing 5,000 rows.

---

## 📊 Dataset Summary

- `CustomerDemographics.xlsx`: Age, gender, income level, banking tenure, etc.
- `CustomerBehaviorData.xlsx`: Credit/debit transaction data (3 months), card limits, loans, investments.
- `CreditConsumptionData.xlsx`: Known target variable `cc_cons` for 15,000 customers.

---

## 🔁 Workflow

1. **Data Preprocessing**
   - Merge 3 files on `Customer_ID`
   - Impute missing values
   - One-hot encode categorical variables

2. **Modeling**
   - Split data into training and validation sets
   - Train multiple models:
     - Linear Regression ✅ (Best performing)
     - Random Forest Regressor
     - Gradient Boosting Regressor

3. **Evaluation**
   - Metrics used: MAE, MSE, RMSE, R², RMSPE
   - Linear Regression achieved the best RMSPE and R² score

4. **Prediction**
   - Use the best model to predict `cc_cons` for the 5,000 unknowns
   - Save predictions to a CSV

---

## 🧪 Model Results Summary

| Model                     | MAE     | RMSE    | R²     | RMSPE  |
|--------------------------|---------|---------|--------|--------|
| **Linear Regression**     | 1857.26 | 2577.96 | 0.8660 | 0.2703 |
| Random Forest Regressor  | 1942.74 | 2875.87 | 0.8332 | 0.2908 |
| Gradient Boosting        | 1874.71 | 2674.19 | 0.8558 | 0.2852 |

✅ **Linear Regression** was chosen as the final model.
📁 Outputs
📄 predicted_credit_consumption_for_missing.csv: Contains predictions for 5,000 customers.

📝 results_report.md: Model comparison and metrics.

📊 Notebook visualizations: Correlation heatmaps, feature importance, etc.
📌 Notes
All missing values were imputed (mean/mode)

One-hot encoding was used for categorical features

Feature selection based on correlation and business intuition
