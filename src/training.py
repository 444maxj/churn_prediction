# src/training.py
import joblib
import os
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train_xgboost(X_train, y_train, params: dict = None) -> XGBClassifier:
    """
    Entraîne un modèle XGBoost.
    Paramètres optimisés pour le churn (déséquilibre de classes).
    """
    # Calculer le poids des classes pour gérer le déséquilibre
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos

    default_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos,  # Gestion du déséquilibre
        'random_state': 42,
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'verbosity': 0
    }

    if params:
        default_params.update(params)

    print("Entraînement XGBoost...")
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    print("XGBoost entraîné.")
    return model


def train_catboost(X_train, y_train, params: dict = None) -> CatBoostClassifier:
    """
    Entraîne un modèle CatBoost.
    Avantage : gère nativement les variables catégorielles.
    """
    default_params = {
        'iterations': 300,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',  # Gestion automatique du déséquilibre
        'verbose': 0
    }

    if params:
        default_params.update(params)

    print("Entraînement CatBoost...")
    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)
    print("CatBoost entraîné.")
    return model


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Modèle baseline pour comparaison."""
    print("Entraînement Logistic Regression (baseline)...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Logistic Regression entraînée.")
    return model


def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    """
    Validation croisée stratifiée.
    Retourne les scores moyens sur plusieurs métriques.
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    metrics = {}
    for metric in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        scores = cross_val_score(model, X, y, cv=kf, scoring=metric)
        metrics[metric] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
        print(f"   {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

    return metrics


def save_model(model, path: str) -> None:
    """Sauvegarde un modèle avec joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path: str):
    """Charge un modèle sauvegardé."""
    model = joblib.load(path)
    print(f"Modèle chargé : {path}")
    return model


def train_all_models(X_train, y_train, X_test=None, y_test=None):
    """
    Entraîne tous les modèles et les sauvegarde.
    Retourne un dict {nom: modèle}.
    """
    models = {}

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, 'models/xgb_model.pkl')
    models['XGBoost'] = xgb_model

    # CatBoost
    cat_model = train_catboost(X_train, y_train)
    save_model(cat_model, 'models/cat_model.pkl')
    models['CatBoost'] = cat_model

    # Logistic Regression (baseline)
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, 'models/lr_model.pkl')
    models['LogisticRegression'] = lr_model

    print("\nTous les modèles ont été entraînés et sauvegardés.")
    return models


if __name__ == "__main__":
    from preprocessing import prepare_data

    X_train, X_test, y_train, y_test, features, _ = prepare_data(
        'data/telco_churn.csv'
    )

    print("\n--- Validation croisée XGBoost ---")
    xgb = train_xgboost(X_train, y_train)
    cross_validate_model(xgb, X_train, y_train)