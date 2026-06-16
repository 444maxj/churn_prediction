# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset Telco Churn."""
    df = pd.read_csv(filepath)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Affiche les infos de base sur le dataset."""
    print("\nAperçu du dataset :")
    print(df.head())
    print(f"\nDimensions : {df.shape}")
    print(f"\nTypes de données :\n{df.dtypes}")
    print(f"\nValeurs manquantes :\n{df.isnull().sum()}")
    print(f"\nDistribution de la cible :")
    print(df['Churn'].value_counts(normalize=True).round(3) * 100)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset."""
    df = df.copy()

    # Supprimer customerID (identifiant inutile pour le ML)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convertir TotalCharges en numérique (contient des espaces vides)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Imputer les valeurs manquantes de TotalCharges par la médiane
    median_total = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total)

    print(f"Données nettoyées : {df.shape[0]} lignes")
    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode les variables catégorielles.
    Retourne (df_encoded, encoders_dict)
    """
    df = df.copy()
    encoders = {}

    # Encoder la cible Churn (Yes=1, No=0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Variables binaires Yes/No → 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'SeniorCitizen'
    ]
    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col] == 'Yes').astype(int)

    # Variables catégorielles à encoder avec LabelEncoder
    cat_cols = [
        'gender', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    print(f"Encodage terminé. {len(cat_cols)} variables encodées.")
    return df, encoders


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée de nouvelles features."""
    df = df.copy()

    # Ratio charges totales / durée d'abonnement
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Catégorie de durée d'abonnement
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # Nombre de services souscrits
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    # Compter les services actifs (valeur > 0 après encodage)
    df['TotalServices'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v > 0), axis=1
    )

    print("Feature engineering terminé.")
    return df


def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline complet : chargement → nettoyage → encodage → split.
    Retourne (X_train, X_test, y_train, y_test, feature_names, encoders)
    """
    # Chargement et nettoyage
    df = load_data(filepath)
    df = clean_data(df)
    df, encoders = encode_features(df)
    df = feature_engineering(df)

    # Séparation features / cible
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    feature_names = list(X.columns)

    # Normalisation (utile pour certains modèles)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names
    )

    # Split train/test stratifié (pour préserver le ratio Churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nSplit des données :")
    print(f"   Train : {X_train.shape[0]} exemples")
    print(f"   Test  : {X_test.shape[0]} exemples")
    print(f"   Taux churn (train) : {y_train.mean():.1%}")
    print(f"   Taux churn (test)  : {y_test.mean():.1%}")

    # Sauvegarder les noms de features
    os.makedirs('models', exist_ok=True)
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_train, X_test, y_train, y_test, feature_names, encoders


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, encoders = prepare_data(
        'data/telco_churn.csv'
    )
    print(f"\n✅ Features : {features}")