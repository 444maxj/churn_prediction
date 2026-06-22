# src/prediction.py
import numpy as np
import pandas as pd
import joblib
from typing import Union


def load_artifacts(model_type: str = 'xgboost') -> tuple:
    """
    Charge le modèle, le scaler, les feature names et les encodeurs.
    model_type : 'xgboost' ou 'catboost'
    """
    feature_names = joblib.load('models/feature_names.pkl')
    scaler = joblib.load('models/scaler.pkl')
    try:
        encoders = joblib.load('models/encoders.pkl')
    except FileNotFoundError:
        encoders = {}

    if model_type == 'xgboost':
        model = joblib.load('models/xgb_model.pkl')
    elif model_type == 'catboost':
        model = joblib.load('models/cat_model.pkl')
    else:
        raise ValueError("model_type doit être 'xgboost' ou 'catboost'")

    return model, scaler, feature_names, encoders


def preprocess_single_client(client_data: dict,
                               feature_names: list,
                               scaler,
                               encoders: dict) -> pd.DataFrame:
    """
    Prépare les données d'un seul client pour la prédiction en utilisant les encodeurs.
    """
    processed = {}
    for key, value in client_data.items():
        processed[key] = value

    # Variables binaires (Yes/No)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        if col in processed and isinstance(processed[col], str):
            processed[col] = 1 if processed[col] == 'Yes' else 0

    # Variables catégorielles avec LabelEncoder
    for col, le in encoders.items():
        if col in processed:
            try:
                # On met dans une liste car transform attend un array-like
                processed[col] = le.transform([str(processed[col])])[0]
            except ValueError:
                # Si valeur inconnue, on prend la première classe par défaut
                processed[col] = 0

    # Feature engineering (mêmes transformations qu'à l'entraînement)
    tenure = processed.get('tenure', 0)
    monthly = processed.get('MonthlyCharges', 0)
    total = processed.get('TotalCharges', monthly * tenure)

    processed['TotalCharges'] = total
    processed['ChargesPerMonth'] = total / (tenure + 1)

    # TenureGroup
    if tenure <= 12:
        processed['TenureGroup'] = 0
    elif tenure <= 24:
        processed['TenureGroup'] = 1
    elif tenure <= 48:
        processed['TenureGroup'] = 2
    else:
        processed['TenureGroup'] = 3

    # TotalServices
    service_keys = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    processed['TotalServices'] = sum(
        1 for k in service_keys if processed.get(k, 0) > 0
    )

    # Créer le DataFrame avec les bonnes colonnes dans le bon ordre
    df = pd.DataFrame([processed])
    df = df.reindex(columns=feature_names, fill_value=0)

    # Normalisation avec le scaler entraîné
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_names
    )

    return df_scaled


def predict_churn(client_data: dict,
                   model_type: str = 'xgboost') -> dict:
    """
    Prédit le risque de churn pour un client.
    Retourne un dict avec la prédiction et la probabilité.
    """
    model, scaler, feature_names, encoders = load_artifacts(model_type)

    X = preprocess_single_client(client_data, feature_names, scaler, encoders)
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    result = {
        'prediction': int(prediction),
        'label': 'CHURN' if prediction == 1 else 'NO CHURN',
        'probability_churn': float(proba[1]),
        'probability_no_churn': float(proba[0]),
        'risk_level': get_risk_level(proba[1]),
        'model_used': model_type
    }

    return result


def get_risk_level(proba_churn: float) -> str:
    """Classe le risque en 3 niveaux."""
    if proba_churn >= 0.70:
        return 'Risque élevé'
    elif proba_churn >= 0.40:
        return 'Risque modéré'
    else:
        return 'Risque faible'


def get_retention_recommendations(client_data: dict,
                                   proba_churn: float) -> list:
    """
    Génère des recommandations de fidélisation basées sur le profil client.
    """
    recommendations = []

    if proba_churn < 0.4:
        return ["Client fidèle. Maintenir la qualité de service actuelle."]

    # Analyse du contrat
    contract = client_data.get('Contract', '')
    if contract == 'Month-to-month':
        recommendations.append(
            "Proposer un contrat annuel avec remise de 10-15%"
        )

    # Analyse des charges
    monthly = client_data.get('MonthlyCharges', 0)
    if monthly > 70:
        recommendations.append(
            "Offrir une réduction tarifaire ou un bundle avantageux"
        )

    # Analyse de la durée d'abonnement
    tenure = client_data.get('tenure', 0)
    if tenure < 12:
        recommendations.append(
            "Programme de fidélité pour les nouveaux clients (< 1 an)"
        )

    # Analyse des services
    tech_support = client_data.get('TechSupport', 'No')
    if tech_support in ['No', 0]:
        recommendations.append(
            "Proposer le service TechSupport (réduit les résiliations de 20%)"
        )

    online_security = client_data.get('OnlineSecurity', 'No')
    if online_security in ['No', 0]:
        recommendations.append(
            "Activer OnlineSecurity — service très valorisé par les clients"
        )

    payment = client_data.get('PaymentMethod', '')
    if payment == 'Electronic check':
        recommendations.append(
            "Encourager le paiement automatique (moins de frictions)"
        )

    if not recommendations:
        recommendations.append("Contacter le client pour un entretien de satisfaction")

    return recommendations


if __name__ == "__main__":
    # Test avec un client fictif
    test_client = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 3,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.70,
        'TotalCharges': 260.0
    }

    result = predict_churn(test_client, 'xgboost')
    print(f"\nPrédiction : {result['label']}")
    print(f"Probabilité de churn : {result['probability_churn']:.1%}")
    print(f"Niveau de risque : {result['risk_level']}")

    recs = get_retention_recommendations(test_client, result['probability_churn'])
    print("\nRecommandations :")
    for r in recs:
        print(f"   {r}")