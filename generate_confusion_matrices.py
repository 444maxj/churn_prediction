"""
generate_confusion_matrices.py
──────────────────────────────
Génère une figure PNG contenant les matrices de confusion
de tous les modèles (XGBoost, CatBoost, Logistic Regression, Random Forest).

Usage:
    python generate_confusion_matrices.py
Sortie:
    all_confusion_matrices.png  (dans le dossier racine du projet)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')   # backend non-interactif — pas de fenêtre GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Palette (assortie à l'interface) ────────────────────────────────
COLORS = {
    'XGBoost':             '#1d4ed8',   # bleu électrique
    'CatBoost':            '#0f0f0f',   # noir encre
    'Logistic Regression': '#6366f1',   # indigo
    'Random Forest':       '#0891b2',   # cyan
}
BG      = '#f5f4f0'   # fond canvas
SURFACE = '#ffffff'   # fond carte
BORDER  = '#e5e7eb'


# ── 1. Chargement et préparation des données ────────────────────────
def load_data():
    df = pd.read_csv('data/telco_churn.csv')
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

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
    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3], include_lowest=True
    ).astype(int)
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['TotalServices'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v > 0), axis=1
    )

    feature_names = joblib.load('models/feature_names.pkl')
    X = df[feature_names]
    y = df['Churn']

    scaler = joblib.load('models/scaler.pkl')
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


# ── 2. Chargement des modèles ────────────────────────────────────────
def load_models(X_train, y_train):
    models = {}
    models['XGBoost']  = joblib.load('models/xgb_model.pkl')
    models['CatBoost'] = joblib.load('models/cat_model.pkl')

    y_tr = np.array(y_train).astype(int).ravel()

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_tr)
    models['Logistic Regression'] = lr

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_tr)
    models['Random Forest'] = rf

    return models


# ── 3. Génération de la figure ───────────────────────────────────────
def plot_all_confusion_matrices(models, X_test, y_test, output_path='all_confusion_matrices.png'):
    labels = ['No Churn', 'Churn']
    n = len(models)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, 5.5 * nrows),
        facecolor=BG,
    )
    axes = np.array(axes).reshape(nrows, ncols)   # garanti 2-D

    # Titre global
    fig.suptitle(
        "Matrices de Confusion — Comparaison des Modèles",
        fontsize=18, fontweight='bold', color='#0f0f0f',
        y=1.01, fontfamily='DejaVu Sans'
    )

    for idx, (name, model) in enumerate(models.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        accent = COLORS.get(name, '#1d4ed8')

        # Matrice
        y_pred = np.array(model.predict(X_test)).astype(int).ravel()
        y_true = np.array(y_test).astype(int).ravel()
        cm = confusion_matrix(y_true, y_pred)

        # Colormap personnalisée : blanc → couleur accent
        cmap = LinearSegmentedColormap.from_list(
            f'cmap_{name}', ['#f0f4ff', accent, '#0f0f0f'], N=256
        )

        im = ax.imshow(cm, cmap=cmap, aspect='auto')

        # Annotations dans chaque cellule
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / cm.sum() * 100
                color = 'white' if val > thresh else '#0f0f0f'
                ax.text(
                    j, i,
                    f"{val:,}\n({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=15, fontweight='bold', color=color,
                )

        # Axes
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=11, color='#374151')
        ax.set_yticklabels(labels, fontsize=11, color='#374151', rotation=90, va='center')
        ax.set_xlabel('Prédit', fontsize=12, color='#374151', labelpad=8)
        ax.set_ylabel('Réel',   fontsize=12, color='#374151', labelpad=8)

        # Titre du sous-graphe avec badge coloré
        ax.set_title(name, fontsize=14, fontweight='bold', color=accent, pad=12)

        # Contour
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1.2)

        # Fond blanc de la zone de plot
        ax.set_facecolor(SURFACE)

    # Masquer les cellules vides si n est impair
    if n % 2 != 0:
        axes[-1, -1].axis('off')

    fig.patch.set_facecolor(BG)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"[OK] Figure sauvegardee : {output_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Chargement des données…")
    X_train, X_test, y_train, y_test = load_data()

    print("Chargement des modèles…")
    models = load_models(X_train, y_train)

    print("Génération de la figure…")
    plot_all_confusion_matrices(models, X_test, y_test)
