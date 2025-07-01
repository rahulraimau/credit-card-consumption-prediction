import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
def load_data(demo_path, behavior_path, consumption_path):
    demo = pd.read_excel(demo_path)
    behavior = pd.read_excel(behavior_path)
    consumption = pd.read_excel(consumption_path)
    return demo, behavior, consumption

# Merge & Clean
def preprocess_data(demo, behavior, consumption):
    df = demo.merge(behavior, on='ID', how='inner').merge(consumption, on='ID', how='inner')
    df.dropna(inplace=True)
    return df

# EDA function (Optional — use in notebooks)
def basic_eda(df):
    print(df.describe())
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

# Train-test split
def split_data(df, target_col='Credit Consumption'):
    X = df.drop(columns=['ID', target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
def train_all_models(X_train, y_train):
    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(alpha=0.01),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model
    return models

# Evaluate models
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"📌 {name}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
        print(f"R2:  {r2_score(y_test, y_pred):.2f}")
        print("-" * 30)

# Full pipeline usage example
if __name__ == "__main__":
    demo_path = "data/CustomerDemographics.xlsx"
    behavior_path = "data/CustomerBehaviorData.xlsx"
    consumption_path = "data/CreditConsumptionData.xlsx"

    demo, behavior, consumption = load_data(demo_path, behavior_path, consumption_path)
    df = preprocess_data(demo, behavior, consumption)

    X_train, X_test, y_train, y_test = split_data(df)
    models = train_all_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)