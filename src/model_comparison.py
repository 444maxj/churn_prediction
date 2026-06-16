"""
Comparaison de 7 modèles de classification sur le dataset Churn.
Génère un rapport complet avec métriques, temps d'entraînement et visualisations.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate


# ──────────────────────────────────────────
# DÉFINITION DES MODÈLES
# ──────────────────────────────────────────

def get_all_models(scale_pos_weight: float = 3.0) -> dict:
    """
    Retourne un dictionnaire de modèles prêts à l'entraînement.
    scale_pos_weight = ratio neg/pos pour gérer le déséquilibre.
    """
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1,
            random_state=42
        ),
        'SVC': SVC(
            probability=True,
            kernel='rbf', class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, scale_pos_weight=scale_pos_weight,
            random_state=42, verbosity=0, eval_metric='auc'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, class_weight='balanced',
            random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=6,
            learning_rate=0.05, auto_class_weights='Balanced',
            random_seed=42, verbose=0
        ),
    }


# ──────────────────────────────────────────
# ÉVALUATION COMPLÈTE
# ──────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test,
                   model_name: str) -> dict:
    """
    Entraîne et évalue un modèle. Mesure aussi le temps d'entraînement.
    """
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    start = time.time()
    _ = model.predict(X_test)
    infer_time = (time.time() - start) * 1000

    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'train_time_s': round(train_time, 2),
        'infer_time_ms': round(infer_time, 1),
    }


def run_cross_validation(models: dict, X_train, y_train, cv: int = 5) -> pd.DataFrame:
    """
    Validation croisée stratifiée pour tous les modèles.
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    results = []

    for name, model in models.items():
        print(f"  → CV {name}...", end=" ")
        cv_results = cross_validate(model, X_train, y_train,
                                    cv=kf, scoring=scoring, n_jobs=-1)
        row = {'model': name}
        for metric in scoring:
            scores = cv_results[f'test_{metric}']
            row[f'{metric}_mean'] = scores.mean()
            row[f'{metric}_std'] = scores.std()
        results.append(row)
        print(f"AUC={row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}")

    return pd.DataFrame(results).set_index('model').round(4)


def compare_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Pipeline complet : entraîne, évalue et compare tous les modèles.
    Retourne (df_metrics, trained_models_dict)
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    models = get_all_models(scale_pos_weight=neg / pos)

    print("\nEntraînement et évaluation de tous les modèles...\n")
    results = []
    trained = {}

    for name, model in models.items():
        print(f"  → {name}...", end=" ")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        trained[name] = model
        print(f"   AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f}"
              f" | {metrics['train_time_s']}s")

    df = pd.DataFrame(results).set_index('model').round(4)
    df = df.sort_values('roc_auc', ascending=False)

    print("\nClassement final :")
    print(df[['accuracy', 'f1', 'roc_auc', 'train_time_s']].to_string())

    os.makedirs('models', exist_ok=True)
    for name, model in trained.items():
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, f'models/{safe_name}.pkl')

    return df, trained


def plot_model_comparison(df_metrics: pd.DataFrame,
                          save_path: str = None) -> plt.Figure:
    """
    Radar chart + bar chart de comparaison des modèles.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    x = np.arange(len(df_metrics))
    width = 0.13
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df_metrics[metric],
               width=width, label=metric.upper(),
               color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_metrics.index, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Comparaison des Métriques par Modèle', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('Score')

    ax2 = axes[1]
    heatmap_data = df_metrics[metrics + ['avg_precision']].T
    sns.heatmap(
        heatmap_data, ax=ax2,
        annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0.5, vmax=1.0,
        linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    ax2.set_title('Heatmap des Performances', fontsize=13)
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=35)

    plt.suptitle('Comparaison Complète des Modèles ML — Churn Prediction',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    return fig


def plot_tradeoff_scatter(df_metrics: pd.DataFrame,
                          save_path: str = None) -> plt.Figure:
    """
    Scatter : Recall vs Precision, taille = ROC-AUC, couleur = F1.
    Idéal pour visualiser les trade-offs métier (coût des FP vs FN).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        df_metrics['recall'],
        df_metrics['precision'],
        s=df_metrics['roc_auc'] * 800,
        c=df_metrics['f1'],
        cmap='RdYlGn', vmin=0.5, vmax=0.9,
        alpha=0.85, edgecolors='black', linewidth=0.7
    )
    plt.colorbar(scatter, ax=ax, label='F1-Score')

    for model_name, row in df_metrics.iterrows():
        ax.annotate(
            model_name,
            (row['recall'], row['precision']),
            xytext=(8, 4), textcoords='offset points',
            fontsize=9
        )

    ax.set_xlabel('Recall (détection des churners)', fontsize=12)
    ax.set_ylabel('Precision (fiabilité des alertes)', fontsize=12)
    ax.set_title('Trade-off Precision/Recall\n(taille = ROC-AUC, couleur = F1)',
                 fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    return fig


def plot_speed_vs_performance(df_metrics: pd.DataFrame,
                               save_path: str = None) -> plt.Figure:
    """
    Scatter : Temps d'entraînement vs ROC-AUC.
    Utile pour choisir le meilleur ratio perf/coût.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_metrics)))

    for i, (name, row) in enumerate(df_metrics.iterrows()):
        ax.scatter(row['train_time_s'], row['roc_auc'],
                   s=200, color=colors[i], zorder=5, edgecolors='black',
                   linewidth=0.7)
        ax.annotate(name, (row['train_time_s'], row['roc_auc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel("Temps d'entraînement (secondes)", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Performance vs Vitesse d'entraînement", fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    from src.preprocessing import prepare_data

    X_train, X_test, y_train, y_test, features, _ = prepare_data(
        'data/telco_churn.csv'
    )

    df_metrics, trained_models = compare_all_models(
        X_train, X_test, y_train, y_test
    )

    plot_model_comparison(df_metrics, save_path='outputs/model_comparison.png')
    plot_tradeoff_scatter(df_metrics, save_path='outputs/tradeoff_scatter.png')
    plot_speed_vs_performance(df_metrics, save_path='outputs/speed_vs_perf.png')
