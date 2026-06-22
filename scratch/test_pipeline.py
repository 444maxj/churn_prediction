import sys, os
os.chdir(r'e:\churn_prediction')
sys.path.insert(0, '.')

import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split

print('Testing pipeline...')
df = pd.read_csv('data/telco_churn.csv')
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

encoders = joblib.load('models/encoders.pkl')
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
for col in binary_cols:
    if col in df.columns and df[col].dtype == object:
        df[col] = (df[col] == 'Yes').astype(int)

for col, le in encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col].astype(str))

df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=[0,1,2,3], include_lowest=True).astype(int)
svc = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df['TotalServices'] = df[svc].apply(lambda r: sum(1 for v in r if v > 0), axis=1)

feature_names = joblib.load('models/feature_names.pkl')
X = df[feature_names]
y = df['Churn']
scaler = joblib.load('models/scaler.pkl')
X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

from src.evaluation import compute_metrics
models = {
    'xgboost': joblib.load('models/xgb_model.pkl'),
    'catboost': joblib.load('models/cat_model.pkl')
}
for name, m in models.items():
    r = compute_metrics(m, X_test, y_test, name)
    print(f"  {name}: AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  Recall={r['recall']:.4f}")

print('\nTest prediction pipeline...')
from src.prediction import predict_churn, get_retention_recommendations
test_client = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 3, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 85.70, 'TotalCharges': 260.0
}
result = predict_churn(test_client, 'xgboost')
print(f"  Prediction: {result['label']}  Prob: {result['probability_churn']:.1%}")
recs = get_retention_recommendations(test_client, result['probability_churn'])
print(f"  Recs ({len(recs)}): {recs[0]}")

print('\nALL TESTS PASSED')
