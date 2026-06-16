"""
Trois stratégies d'optimisation :
  A) RandomizedSearchCV   — rapide, bonne couverture
  B) GridSearchCV         — exhaustif, lent, pour affinage final
  C) Optuna               — Bayésien, le plus efficace (BONUS++)
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────
# A) RANDOMIZED SEARCH
# ──────────────────────────────────────────

def tune_xgboost_random(X_train, y_train,
                         n_iter: int = 50,
                         cv: int = 5) -> XGBClassifier:
    """
    Optimisation rapide de XGBoost avec RandomizedSearchCV.
    n_iter=50 donne un bon compromis vitesse/qualité.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    param_dist = {
        'n_estimators':      randint(100, 500),
        'max_depth':         randint(3, 10),
        'learning_rate':     loguniform(0.01, 0.3),
        'subsample':         uniform(0.6, 0.4),
        'colsample_bytree':  uniform(0.6, 0.4),
        'min_child_weight':  randint(1, 10),
        'reg_alpha':         loguniform(0.001, 1.0),
        'reg_lambda':        loguniform(0.1, 10.0),
        'gamma':             uniform(0, 0.5),
    }

    base_model = XGBClassifier(
        scale_pos_weight=neg / pos,
        random_state=42, verbosity=0,
        eval_metric='auc'
    )

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=n_iter, cv=cv_strategy,
        scoring=scorer,
        n_jobs=-1, verbose=1,
        random_state=42
    )

    print(f"RandomizedSearch XGBoost ({n_iter} itérations, {cv} folds)...")
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Terminé en {elapsed:.0f}s")
    print(f"   Meilleur AUC (CV) : {search.best_score_:.4f}")
    print(f"   Meilleurs params  : {search.best_params_}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(search.best_estimator_, 'models/xgb_optimized.pkl')

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df = cv_df.sort_values('rank_test_score')
    os.makedirs('outputs', exist_ok=True)
    cv_df.to_csv('outputs/xgb_search_results.csv', index=False)
    print(f"Résultats sauvegardés : outputs/xgb_search_results.csv")

    return search.best_estimator_


def tune_catboost_random(X_train, y_train,
                          n_iter: int = 30,
                          cv: int = 5) -> CatBoostClassifier:
    """
    Optimisation CatBoost avec RandomizedSearchCV.
    CatBoost est plus lent, donc n_iter plus faible.
    """
    param_dist = {
        'iterations':    [200, 300, 500],
        'depth':         randint(4, 10),
        'learning_rate': loguniform(0.01, 0.2),
        'l2_leaf_reg':   randint(1, 10),
        'bagging_temperature': uniform(0, 1),
        'random_strength':     uniform(0, 3),
        'border_count':  [32, 64, 128, 254],
    }

    base_model = CatBoostClassifier(
        auto_class_weights='Balanced',
        random_seed=42, verbose=0
    )

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=n_iter, cv=cv_strategy,
        scoring=scorer,
        n_jobs=1,
        verbose=0,
        random_state=42
    )

    print(f"RandomizedSearch CatBoost ({n_iter} itérations)...")
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Terminé en {elapsed:.0f}s")
    print(f"   Meilleur AUC (CV) : {search.best_score_:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(search.best_estimator_, 'models/cat_optimized.pkl')
    return search.best_estimator_


# ──────────────────────────────────────────
# B) GRID SEARCH (affinage final)
# ──────────────────────────────────────────

def fine_tune_xgboost(X_train, y_train,
                       best_params: dict = None) -> XGBClassifier:
    """
    GridSearchCV sur un espace réduit autour des meilleurs paramètres.
    À utiliser APRÈS RandomizedSearch pour affiner.
    """
    best_depth = best_params.get('max_depth', 6) if best_params else 6
    best_lr = best_params.get('learning_rate', 0.05) if best_params else 0.05

    param_grid = {
        'max_depth':     [max(3, best_depth - 1), best_depth, best_depth + 1],
        'learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
        'min_child_weight': [1, 3, 5],
    }

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    base_model = XGBClassifier(
        n_estimators=300,
        scale_pos_weight=neg / pos,
        random_state=42, verbosity=0,
        eval_metric='auc',
        subsample=best_params.get('subsample', 0.8) if best_params else 0.8,
        colsample_bytree=best_params.get('colsample_bytree', 0.8) if best_params else 0.8,
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    grid = GridSearchCV(
        base_model, param_grid,
        cv=cv_strategy, scoring=scorer,
        n_jobs=-1, verbose=1
    )

    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"GridSearch XGBoost ({n_combinations} combinaisons × 5 folds)...")
    grid.fit(X_train, y_train)

    print(f"Meilleur AUC (CV) : {grid.best_score_:.4f}")
    print(f"   Params finaux     : {grid.best_params_}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(grid.best_estimator_, 'models/xgb_final.pkl')
    return grid.best_estimator_


# ──────────────────────────────────────────
# C) OPTUNA — Optimisation Bayésienne
# ──────────────────────────────────────────

def tune_with_optuna(X_train, y_train,
                     model_type: str = 'xgboost',
                     n_trials: int = 100) -> object:
    """
    Optimisation Bayésienne avec Optuna.
    Beaucoup plus efficace que Grid/Random Search.
    pip install optuna
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
print("Optuna non installé. Exécutez : pip install optuna")
        return None

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective_xgb(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'gamma':            trial.suggest_float('gamma', 0, 1.0),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=neg / pos,
            random_state=42, verbosity=0, eval_metric='auc'
        )
        scores = cross_val_score(model, X_train, y_train,
                                  cv=kf, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    def objective_cat(trial):
        params = {
            'iterations':          trial.suggest_int('iterations', 100, 600),
            'depth':               trial.suggest_int('depth', 4, 10),
            'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength':     trial.suggest_float('random_strength', 0, 3),
            'border_count':        trial.suggest_categorical('border_count', [32, 64, 128, 254]),
        }
        model = CatBoostClassifier(
            **params,
            auto_class_weights='Balanced',
            random_seed=42, verbose=0
        )
        scores = cross_val_score(model, X_train, y_train,
                                  cv=kf, scoring='roc_auc', n_jobs=1)
        return scores.mean()

    objective = objective_xgb if model_type == 'xgboost' else objective_cat
    model_tag = 'XGBoost' if model_type == 'xgboost' else 'CatBoost'

    print(f"\nOptuna — Optimisation {model_tag} ({n_trials} essais)...")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nMeilleur AUC (CV) : {study.best_value:.4f}")
    print(f"   Meilleurs params  :\n{study.best_params}")

    if model_type == 'xgboost':
        best_model = XGBClassifier(
            **study.best_params,
            scale_pos_weight=neg / pos,
            random_state=42, verbosity=0
        )
    else:
        best_model = CatBoostClassifier(
            **study.best_params,
            auto_class_weights='Balanced',
            random_seed=42, verbose=0
        )

    best_model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/{model_type}_optuna.pkl')

    _plot_optuna_results(study, model_type)

    return best_model


def _plot_optuna_results(study, model_type: str):
    """Génère les visualisations Optuna."""
    try:
        import matplotlib.pyplot as plt

        os.makedirs('outputs', exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        trials_df = study.trials_dataframe()
        axes[0].plot(trials_df['number'], trials_df['value'], '.', alpha=0.5)
        axes[0].plot(trials_df['number'],
                     trials_df['value'].cummax(), 'r-', linewidth=2, label='Meilleur')
        axes[0].set_xlabel('Essai')
        axes[0].set_ylabel('ROC-AUC')
        axes[0].set_title(f'Convergence Optuna — {model_type.upper()}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(trials_df['value'].dropna(), bins=20,
                     color='#2196F3', edgecolor='white', alpha=0.8)
        axes[1].axvline(study.best_value, color='red', linestyle='--',
                        label=f'Meilleur : {study.best_value:.4f}')
        axes[1].set_xlabel('ROC-AUC')
        axes[1].set_ylabel('Fréquence')
        axes[1].set_title('Distribution des scores')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'outputs/{model_type}_optuna_history.png', dpi=120, bbox_inches='tight')
        print(f"Graphique Optuna sauvegardé : outputs/{model_type}_optuna_history.png")

    except Exception as e:
        print(f"Impossible de générer les graphiques Optuna : {e}")


# ──────────────────────────────────────────
# COMPARAISON AVANT/APRÈS OPTIMISATION
# ──────────────────────────────────────────

def compare_before_after(models_dict: dict,
                          X_test, y_test) -> pd.DataFrame:
    """
    Compare les performances avant et après optimisation.
    models_dict = {'XGBoost (base)': m1, 'XGBoost (Optuna)': m2, ...}
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    )

    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append({
            'Modèle':    name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1':        f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_test, y_proba),
        })

    df = pd.DataFrame(results).set_index('Modèle').round(4)
    print("\nAvant vs Après Optimisation :")
    print(df.to_string())
    return df


if __name__ == "__main__":
    from src.preprocessing import prepare_data

    X_train, X_test, y_train, y_test, features, _ = prepare_data(
        'data/telco_churn.csv'
    )

    xgb_random = tune_xgboost_random(X_train, y_train, n_iter=50)
    best_params = joblib.load('models/xgb_optimized.pkl').get_params()
    xgb_grid = fine_tune_xgboost(X_train, y_train, best_params)
    xgb_optuna = tune_with_optuna(X_train, y_train, 'xgboost', n_trials=100)
    cat_optuna = tune_with_optuna(X_train, y_train, 'catboost', n_trials=50)

    base_xgb = joblib.load('models/xgb_model.pkl')
    compare_before_after({
        'XGBoost (base)': base_xgb,
        'XGBoost (RandomSearch)': xgb_random,
        'XGBoost (GridSearch)': xgb_grid,
        'XGBoost (Optuna)': xgb_optuna,
    }, X_test, y_test)
