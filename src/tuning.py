# src/tuning.py
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from preprocessing import prepare_data

def tune_xgboost():
    print("Préparation des données pour l'optimisation des hyperparamètres...")
    X_train, X_test, y_train, y_test, features, _ = prepare_data('data/telco_churn.csv')

    # Calculer scale_pos_weight pour gérer le déséquilibre
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos

    print("\nDéfinition de l'espace de recherche des hyperparamètres...")
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0]
    }

    # Modèle de base
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False,
        verbosity=0
    )

    # Validation croisée stratifiée
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Lancement du RandomizedSearchCV (validation croisée à 3 plis)...")
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=15,             # 15 combinaisons aléatoires
        scoring='roc_auc',      # Optimisation basée sur l'AUC-ROC
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1               # Utiliser tous les cœurs disponibles
    )

    random_search.fit(X_train, y_train)

    print("\nOptimisation terminée !")
    print(f"Meilleurs hyperparamètres trouvés : {random_search.best_params_}")
    print(f"Meilleur score de validation (AUC-ROC) : {random_search.best_score_:.4f}")

    # Évaluation sur le jeu de test
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"📈 Précision globale (Accuracy) sur le jeu de test : {test_score:.4f}")

    return random_search.best_params_

if __name__ == '__main__':
    tune_xgboost()
