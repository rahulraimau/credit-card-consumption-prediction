# 📊 Credit Card Consumption Prediction — Model Evaluation Report

## ✅ Data Summary
- Total records: **20,000**
- Training set (with target): **15,000**
- Prediction set (missing target): **5,000**
- Final processed features: **47**

---

## 🧪 Model Performance Comparison

| Model                     | MAE     | MSE        | RMSE    | R²     | RMSPE  |
|--------------------------|---------|------------|---------|--------|--------|
| **Linear Regression**     | 1857.26 | 6.64M      | 2577.96 | 0.8660 | 0.2703 |
| Random Forest Regressor  | 1942.74 | 8.27M      | 2875.87 | 0.8332 | 0.2908 |
| Gradient Boosting Regressor | 1874.71 | 7.15M   | 2674.19 | 0.8558 | 0.2852 |

- 🔥 **Best Model**: Linear Regression (lowest RMSPE: `0.2703`)
- Target variable: `cc_cons` (credit card consumption)

---

## 🧠 Prediction Output

- 5,000 customers had missing `cc_cons` values
- Predictions saved to: `predicted_credit_consumption_for_missing.csv`

Sample predictions:
Customer_ID predicted_cc_cons
17591 3182.42
13541 6987.66
13431 2633.90
8687 9012.69
...

yaml
Copy
Edit

---

## 📌 Notes
- Missing values were imputed using mean (numerical) or mode (categorical)
- Categorical variables were one-hot encoded
- Data was scaled before modeling where necessary