# Conventions du Projet et Organisation

## Partage des Responsabilités

Le projet est divisé entre deux étudiants pour assurer une progression efficace et spécialisée :

### Étudiant 1 : Data Scientist / Data Analyst
- **Exploration des Données (EDA)** : Création des visualisations, analyse des corrélations, détection des outliers.
- **Modélisation** : Entraînement des modèles (XGBoost, CatBoost), comparaison des algorithmes.
- **Évaluation** : Analyse des métriques (ROC-AUC, F1-Score, Recall) et tuning des hyperparamètres.

### Étudiant 2 : Machine Learning Engineer / MLOps
- **Préparation des Données** : Nettoyage, traitement des valeurs manquantes, encodage et standardisation.
- **Développement Application** : Création de l'application Streamlit (dashboard, formulaires, visualisation).
- **Industrialisation** : Mise en place du pipeline scikit-learn, conteneurisation Docker, création de l'API (FastAPI).

---

## Conventions de Nommage

Pour maintenir un code propre et lisible, nous suivons ces conventions :

- **Fichiers Python & Dossiers** : `snake_case` (ex: `preprocessing.py`, `ui_components.py`)
- **Classes Python** : `PascalCase` (ex: `ChurnPredictor`, `DataLoader`)
- **Fonctions et Variables** : `snake_case` (ex: `train_model()`, `user_data`)
- **Notebooks Jupyter** : Numérotation séquentielle suivie d'une description en `snake_case` (ex: `01_EDA.ipynb`, `02_preprocessing.ipynb`)
- **Constantes** : `UPPER_SNAKE_CASE` (ex: `RANDOM_STATE = 42`)

---

## Conventions pour les Messages de Commit

Nous utilisons la convention **Conventional Commits** pour standardiser l'historique de notre dépôt Git. Le format d'un commit doit être le suivant :

`<type>(<portée optionnelle>): <description courte>`

### Types autorisés :
- `feat:` : Ajout d'une nouvelle fonctionnalité (ex: `feat: ajout de l'interface Streamlit`)
- `fix:` : Correction d'un bug (ex: `fix: gestion des valeurs nulles dans TotalCharges`)
- `docs:` : Modification de la documentation (ex: `docs: mise à jour du README`)
- `style:` : Changement de formatage sans impact sur le code (ex: `style: formatage avec black`)
- `refactor:` : Refactorisation de code sans ajout de feature ni correction (ex: `refactor: simplification de la fonction clean_data`)
- `test:` : Ajout ou modification de tests
- `chore:` : Mise à jour de configurations, dépendances, etc. (ex: `chore: mise à jour de requirements.txt`)

### Bonnes pratiques :
- Commencer la description par un verbe à l'infinitif ou un nom.
- Ne pas mettre de majuscule à la première lettre de la description.
- Ne pas terminer la description par un point.
