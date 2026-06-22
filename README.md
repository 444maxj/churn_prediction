# Système de Prédiction du Churn Client

**Projet ML | XGBoost & CatBoost | Streamlit**

Ce projet implémente un système complet de prédiction du départ client (churn) pour un opérateur télécom. Il intègre une interface utilisateur moderne de style **SaaS Fintech Glassmorphism** construite avec Streamlit, et plusieurs modèles de Machine Learning performants (XGBoost, CatBoost, Random Forest, Régression Logistique).

---

## 📂 Structure du Projet

```text
churn_prediction/
|-- README.md               # Documentation du projet
|-- app.py                  # Point d'entrée de l'application Streamlit (Interface 5 onglets)
|-- requirements.txt        # Dépendances du projet
|-- data/                   # Données d'entraînement et de test
|-- models/                 # Modèles entraînés et scalers (.pkl)
|-- notebooks/              # Notebooks Jupyter (Exploration de données, tests)
└── src/                    # Code source des modules métier
    |-- dashboard.py        # Logique des graphiques interactifs Plotly
    |-- evaluation.py       # Fonctions d'évaluation et de métriques
    |-- hyperparameter_tuning.py # Logique d'optimisation (GridSearch/RandomSearch)
    |-- prediction.py       # Inférence et génération de recommandations
    |-- preprocessing.py    # Pipeline de nettoyage et d'encodage
    └── ui_components.py    # Composants graphiques et styles CSS personnalisés
```

---

## 🚀 Lancer l'application en local

L'application est conçue pour être lancée très facilement avec Streamlit.

### 1. Activer l'environnement virtuel (Optionnel mais recommandé)
```bash
# Sur Windows
venv\Scripts\activate
# Sur Mac/Linux
source venv/bin/activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Démarrer l'interface
```bash
streamlit run app.py
```
L'interface s'ouvrira automatiquement dans votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501).

---

## 📊 Fonctionnalités de l'Application

L'application Web interactive propose 5 onglets principaux :

1. **Prédiction** : Saisissez le profil d'un client (contrat, factures, options internet) et obtenez une prédiction instantanée de son risque de départ, accompagnée de recommandations d'actions de fidélisation.
2. **Analytics** : Explorez les données clients interactives via des graphiques Plotly (répartition des contrats, impact des services souscrits, facturation).
3. **Insights** : Découvrez quelles sont les variables qui ont le plus d'impact sur le churn (Feature Importance).
4. **Optimisation** : Comparez en temps réel les performances de 4 modèles (Régression Logistique, Random Forest, XGBoost, CatBoost) via des matrices de confusion et des courbes ROC interactives.
5. **Documentation** : Informations et architecture du projet.

---

## ☁️ Déploiement

Le projet est conçu pour être déployé gratuitement et sans configuration complexe sur **Streamlit Community Cloud**.

1. Poussez votre code sur GitHub.
2. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io/).
3. Connectez votre dépôt GitHub.
4. Spécifiez `app.py` comme fichier principal.
5. Cliquez sur "Deploy" !

Les modèles entraînés (`.pkl`) se trouvant dans le dossier `models/`, l'application est prête à l'emploi dès le premier lancement.
