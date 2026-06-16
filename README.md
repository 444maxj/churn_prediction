# Système de Prédiction du Churn Client

**Projet 4IASD | XGBoost & CatBoost | Streamlit**

Ce projet implémente un système complet de prédiction du départ client (churn) pour un opérateur télécom. Il intègre une interface utilisateur moderne de style **SaaS Fintech Glassmorphism** construite avec Streamlit, et deux modèles de Machine Learning performants (XGBoost et CatBoost).

---

## Structure du Projet

```text
churn-prediction/
|-- README.md
|-- app.py                  # Point d'entrée de l'application Streamlit
|-- requirements.txt        # Dépendances du projet
|-- data/
|   └── telco_churn.csv     # Dataset brut (Telco Customer Churn)
|-- models/                 # Modèles et scalers entraînés
|   |-- xgb_model.pkl       # Modèle XGBoost
|   |-- cat_model.pkl       # Modèle CatBoost
|   |-- lr_model.pkl        # Modèle Logistic Regression (Baseline)
|   |-- scaler.pkl          # Scaler StandardScaler
|   └── feature_names.pkl   # Noms des features d'entraînement
|-- outputs/                # Graphiques générés par l'évaluation hors-ligne
|   |-- roc_curve.png
|   └── feature_importance.png
└── src/                    # Code source des modules
    |-- preprocessing.py    # Pipeline de nettoyage et d'encodage
    |-- training.py         # Script d'entraînement des modèles
    |-- evaluation.py       # Fonctions d'évaluation et de tracés
    |-- prediction.py       # Inférence sur nouveaux profils clients
    └── ui_components.py    # Composants graphiques et styles CSS
```

---

## Étape 1 : Setup de l'environnement

### 1. Activer l'environnement virtuel (Windows)
```bash
venv\Scripts\activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Étape 2 : Entraînement et Évaluation

Vous pouvez exécuter les scripts de préparation des données et d'entraînement directement depuis le dossier racine.

### Entraîner les modèles
```bash
python src/training.py
```
Cela va nettoyer le dataset, effectuer le feature engineering, puis entraîner et sauvegarder les modèles (XGBoost, CatBoost et Régression Logistique) dans le dossier `models/`.

### Évaluer les modèles
```bash
python src/evaluation.py
```
Cela calcule les métriques d'évaluation sur le jeu de test et exporte les graphiques de courbe ROC et d'importance des features dans le dossier `outputs/`.

---

## Étape 3 : Lancer l'application web

Démarrez l'application interactive Streamlit :
```bash
streamlit run app.py
```
L'interface s'ouvrira automatiquement à l'adresse [http://localhost:8501](http://localhost:8501).

---

## Métriques de performance obtenues

Les modèles ont été optimisés pour faire face au déséquilibre des classes (environ 26.5% de churn dans les données).

| Modèle | Accuracy | Précision | Recall (Rappel) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 75.9% | 53.2% | **77.8%** | **63.2%** | **84.1%** |
| **XGBoost** | **76.2%** | **53.9%** | 72.7% | 61.9% | 83.3% |

> **Rappel** : Le *Recall* (Rappel) est privilégié dans ce cas d'usage car il est crucial de détecter un maximum de clients sur le point de partir, quitte à générer quelques faux positifs.

---

## Design & Expérience utilisateur

L'application web Streamlit applique un thème **Fintech Glassmorphism** haut de gamme :
*   Arrière-plan épuré avec dégradés radiaux doux.
*   Composants de formulaire encadrés par des cartes à effet verre dépoli (`.glass-card`).
*   Transitions au survol et micro-animations animées en SVG pour les indicateurs visuels.
*   Recommandations personnalisées et dynamiques en fonction des caractéristiques du client (contrat mensuel, options souscrites, factures élevées).

---

## Bonus : Fonctionnalités avancées

L'application inclut désormais :
* Optimisation Hyperparamètres — RandomizedSearch et Optuna pour XGBoost / CatBoost
* Prédiction par Lot (CSV) — upload CSV et prédictions en masse
* Comparaison des Modèles — évaluation de plusieurs algorithmes et visualisation des trade-offs
* Dashboard Analytique — analyses interactives des segments à risque et des services

### Lancer le dashboard et les modules bonus
Après avoir entraîné les modèles et installé les dépendances :
```bash
streamlit run app.py
```

### Exécution en Docker
```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

Puis ouvrir :
```text
http://localhost:8501
```

### API REST avec FastAPI
L'API est disponible via `api.py`.
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints utiles :
* `GET /health`
* `POST /predict`
* `POST /batch_predict`

Example de test via `curl` :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.70,"TotalCharges":260.0,"model_type":"xgboost"}'
```

### Fichiers ajoutés
* `src/model_comparison.py`
* `src/hyperparameter_tuning.py`
* `src/dashboard.py`
* `api.py`
* `Dockerfile`
* `.dockerignore`
* `.streamlit/config.toml`
