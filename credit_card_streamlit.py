import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

import streamlit as st
from io import BytesIO

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Card Consumption Prediction", layout="wide")
st.title("🔍 Credit Card Consumption Prediction App")

uploaded_cc = st.file_uploader("Upload CreditConsumptionData.xlsx")
uploaded_demog = st.file_uploader("Upload CustomerDemographics.xlsx")
uploaded_behavior = st.file_uploader("Upload CustomerBehaviorData.xlsx")

@st.cache_data
def load_data(cc_file, demog_file, behavior_file):
    df_cc = pd.read_excel(cc_file)
    df_demog = pd.read_excel(demog_file)
    df_behavior = pd.read_excel(behavior_file)

    df_cc.rename(columns={"ID": "Customer_ID"}, inplace=True)
    df_demog.rename(columns={"ID": "Customer_ID"}, inplace=True)
    df_behavior.rename(columns={"ID": "Customer_ID"}, inplace=True)

    df = df_cc.merge(df_demog, on="Customer_ID", how="left")
    df = df.merge(df_behavior, on="Customer_ID", how="left")
    return df

def missing_imputation(df, column_name, var_type):
    if var_type == 'continuous':
        return df[column_name].fillna(df[column_name].mean())
    elif var_type == 'categorical':
        return df[column_name].fillna(df[column_name].mode()[0])
    return df[column_name]

def create_dummies(df, column_names):
    for col in column_names:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-8))))

if uploaded_cc and uploaded_demog and uploaded_behavior:
    df = load_data(uploaded_cc, uploaded_demog, uploaded_behavior)
    st.success("Data loaded and merged successfully!")

    df_train = df[df['cc_cons'].notna()].copy()
    df_predict = df[df['cc_cons'].isna()].copy()

    features = [col for col in df.columns if col not in ['Customer_ID', 'cc_cons']]
    num_features = df_train[features].select_dtypes(include=np.number).columns.tolist()
    cat_features = df_train[features].select_dtypes(include='object').columns.tolist()

    for col in num_features:
        df_train[col] = missing_imputation(df_train, col, 'continuous')
    for col in cat_features:
        df_train[col] = missing_imputation(df_train, col, 'categorical')

    df_train = create_dummies(df_train, cat_features)
    X = df_train.drop(['Customer_ID', 'cc_cons'], axis=1)
    y = df_train['cc_cons']

    selector = SelectKBest(score_func=f_regression, k=30)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    X_df = pd.DataFrame(X_selected, columns=selected_features)
    X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    st.subheader("📊 Model Performance")
    st.write("MAE:", mean_absolute_error(y_val, y_pred))
    st.write("MSE:", mean_squared_error(y_val, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
    st.write("R2:", r2_score(y_val, y_pred))
    st.write("RMSPE:", rmspe(y_val, y_pred))

    # --- Save Model ---
    joblib.dump(model, "linear_model.pkl")
    joblib.dump(selected_features, "selected_features.pkl")
    st.success("✅ Model saved as 'linear_model.pkl'")

    # --- Predict Missing ---
    for col in num_features:
        df_predict[col] = missing_imputation(df_predict, col, 'continuous')
    for col in cat_features:
        df_predict[col] = missing_imputation(df_predict, col, 'categorical')

    df_predict = create_dummies(df_predict, cat_features)

    for col in selected_features:
        if col not in df_predict.columns:
            df_predict[col] = 0
    df_predict = df_predict[selected_features]

    predictions = model.predict(df_predict)
    df_output = df[df['cc_cons'].isna()][['Customer_ID']].copy()
    df_output['predicted_cc_cons'] = predictions

    csv = df_output.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", data=csv, file_name="predicted_cc_cons.csv", mime="text/csv")

    st.success("✅ Predictions ready!")
else:
    st.warning("👆 Please upload all 3 Excel files to begin.")
