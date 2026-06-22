# src/evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve
)
import os


def compute_metrics(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Calcule toutes les métriques d'évaluation.
    Retourne un dict avec les scores.
    """
    y_true = np.array(y_test).astype(int).ravel()
    y_pred = np.array(model.predict(X_test)).astype(int).ravel()
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }

    print(f"\nMétriques — {model_name}")
    print(f"   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    print(f"   F1-Score  : {metrics['f1']:.4f}")
    print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test,
                          model_name: str = "Model",
                          save_path: str = None) -> plt.Figure:
    """Affiche et sauvegarde la matrice de confusion."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Churn', 'Churn']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Matrice de Confusion — {model_name}', fontsize=13, pad=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_roc_curve(models: dict, X_test, y_test,
                   save_path: str = None) -> plt.Figure:
    """
    Trace la courbe ROC pour plusieurs modèles.
    models = {'NomModele': model_object}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ['#111827', '#4b5563', '#6b7280', '#9ca3af']
    for i, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr,
                label=f"{name} (AUC = {auc:.3f})",
                color=colors[i % len(colors)],
                linewidth=2)

    # Ligne de référence (classifieur aléatoire)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.500)')

    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=11)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=11)
    ax.set_title('Courbes ROC — Comparaison des modèles', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_feature_importance(model, feature_names: list,
                             model_name: str = "XGBoost",
                             top_n: int = 15,
                             save_path: str = None) -> plt.Figure:
    """Visualise les features les plus importantes."""
    # Récupérer l'importance selon le modèle
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Ce modèle n'a pas de feature_importances_")
        return None

    # Créer un DataFrame trié
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(fi_df)))
    bars = ax.barh(fi_df['feature'], fi_df['importance'],
                   color=colors, edgecolor='white', linewidth=0.5)

    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, fi_df['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=9)

    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'Top {top_n} Features — {model_name}', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_precision_recall_curve(models: dict, X_test, y_test,
                                save_path: str = None) -> plt.Figure:
    """Trace la courbe Precision-Recall pour plusieurs modèles."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#111827', '#4b5563', '#6b7280', '#9ca3af']
    
    for i, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        # Calculate Average Precision
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, y_proba)
        
        ax.plot(recall, precision,
                label=f"{name} (AP = {ap:.3f})",
                color=colors[i % len(colors)],
                linewidth=2)

    # Ligne de base (ratio de churners dans l'échantillon test)
    baseline = sum(y_test) / len(y_test)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
               label=f'Aléatoire (AP = {baseline:.3f})')

    ax.set_xlabel('Recall (Taux de Vrais Positifs)', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Courbes Precision-Recall', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_gain_curve(models: dict, X_test, y_test,
                    save_path: str = None) -> plt.Figure:
    """Trace la courbe de Gain Cumulé pour plusieurs modèles."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#111827', '#4b5563', '#6b7280', '#9ca3af']
    
    for i, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculer le gain cumulé
        percentages, gains = _compute_cumulative_gain(y_test, y_proba)
        
        ax.plot(percentages, gains,
                label=f"{name}",
                color=colors[i % len(colors)],
                linewidth=2)

    # Classifieur idéal (Perfect model)
    percentages_perfect, gains_perfect = _compute_cumulative_gain(y_test, y_test)
    ax.plot(percentages_perfect, gains_perfect, 
            label='Idéal', color='green', linestyle=':', linewidth=2)

    # Ligne de base (Random model)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire')

    ax.set_xlabel("% d'échantillon ciblé", fontsize=11)
    ax.set_ylabel('% de Churners capturés', fontsize=11)
    ax.set_title('Courbes de Gain Cumulé', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig

def _compute_cumulative_gain(y_true, y_proba):
    """Calcule les points pour la courbe de gain cumulé."""
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    
    total_positives = df['y_true'].sum()
    df['cumulative_positives'] = df['y_true'].cumsum()
    df['gain'] = df['cumulative_positives'] / total_positives
    
    percentages = np.linspace(0, 1, len(df))
    return percentages, df['gain'].values


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compare plusieurs modèles et retourne un DataFrame récapitulatif.
    """
    results = []
    for name, model in models.items():
        metrics = compute_metrics(model, X_test, y_test, name)
        results.append(metrics)

    df = pd.DataFrame(results).set_index('model')
    df = df.round(4)

    print("\n📊 Comparaison des modèles :")
    print(df.to_string())
    return df


if __name__ == "__main__":
    import joblib
    from preprocessing import prepare_data

    X_train, X_test, y_train, y_test, features, _ = prepare_data(
        'data/telco_churn.csv'
    )

    xgb = joblib.load('models/xgb_model.pkl')
    cat = joblib.load('models/cat_model.pkl')

    models = {'XGBoost': xgb, 'CatBoost': cat}
    compare_models(models, X_test, y_test)
    plot_roc_curve(models, X_test, y_test, save_path='outputs/roc_curve.png')
    plot_feature_importance(xgb, features, save_path='outputs/feature_importance.png')