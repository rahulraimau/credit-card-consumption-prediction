
# 🧠 Credit Card Consumption Prediction

This project aims to **predict credit card consumption** using customer demographics and behavior data. Built as part of a capstone case study, it walks through a full machine learning pipeline from data exploration to model evaluation.

## 📁 Project Structure

```
credit-card-consumption-prediction/
├── data/                   # Raw data files (Excel)
├── notebooks/              # Main Jupyter notebook
├── reports/                # PDF report & analysis
├── src/                    # Custom functions/scripts (optional)
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── .gitignore
```

## 📊 Dataset Description

- **CustomerDemographics.xlsx**: Age, gender, income, education, etc.
- **CustomerBehaviorData.xlsx**: Credit limit usage, transactions, etc.
- **CreditConsumptionData.xlsx**: Actual credit consumption (target variable)

## 🔍 Workflow Summary

1. **EDA**: Visualizations of income, credit limit, and correlation heatmaps.
2. **Preprocessing**: Missing value imputation, outlier removal.
3. **Feature Engineering**: Combining datasets, scaling, encoding.
4. **Modeling**:
   - Linear Regression
   - Lasso & Ridge Regression
   - Random Forest Regressor
5. **Evaluation**: MAE, MSE, R² comparison

## 📈 Results

The Random Forest model performed best with highest R² score, suggesting nonlinear relationships in features.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-card-consumption-prediction.git
cd credit-card-consumption-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook notebooks/predict_credit_card_consumption.ipynb
```

## 📄 License

MIT License
