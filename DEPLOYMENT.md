# Guide de Déploiement : Churn Predictor

Ce document couvre les étapes 32 et 33 du projet pour déployer l'application Streamlit et l'API FastAPI en production.

---

## 1. Déploiement de l'Application Streamlit (Streamlit Cloud)

Streamlit Cloud est la méthode la plus simple pour déployer l'interface utilisateur gratuitement.

### Étapes à suivre :
1. **Préparer le Repository GitHub** :
   - Assurez-vous que votre code est pushé sur GitHub avec le fichier `app.py` et `requirements.txt` à la racine.
   - Les dossiers `data/` et `models/` doivent être présents avec les modèles pré-entraînés (car Streamlit Cloud n'exécutera pas l'entraînement).
2. **Connexion à Streamlit Cloud** :
   - Allez sur [share.streamlit.io](https://share.streamlit.io/).
   - Connectez-vous avec votre compte GitHub.
3. **Créer une Nouvelle Application** :
   - Cliquez sur **New app**.
   - Sélectionnez le repository de votre projet (ex: `votre-pseudo/telco-churn`).
   - Branche : `main` ou `master`.
   - Main file path : `app.py`.
4. **Configuration Avancée (Secrets)** :
   - Si votre application a besoin de mots de passe ou clés d'API (non requis pour ce projet), configurez-les dans `Advanced settings > Secrets`.
5. **Lancement** :
   - Cliquez sur **Deploy**.
   - Votre application sera en ligne en quelques minutes à une URL publique !

---

## 2. Déploiement de l'API FastAPI (Heroku / Render / AWS)

Pour déployer l'API REST FastAPI de manière autonome, il est recommandé d'utiliser des conteneurs Docker ou des plateformes PaaS comme Render ou Heroku.

### Option A : Déploiement sur Render (Gratuit / Facile)
Render gère très bien les web services basés sur Python et Docker.

1. Connectez-vous sur [Render.com](https://render.com) avec GitHub.
2. Cliquez sur **New > Web Service**.
3. Choisissez **Build and deploy from a Git repository**.
4. Dans la configuration :
   - **Environment** : `Docker`
   - **Dockerfile path** : `./Dockerfile.api` (Important : utilisez bien le Dockerfile dédié à l'API).
5. Cliquez sur **Create Web Service**. 
6. Render va construire l'image Docker et exposer l'API. Vous pourrez tester les requêtes POST sur `https://votre-app.onrender.com/predict`.

### Option B : Déploiement sur Heroku (Via Heroku Container Registry)
Si vous préférez Heroku, vous pouvez utiliser Docker.

```bash
# Se connecter à Heroku
heroku login
heroku container:login

# Créer une nouvelle application Heroku
heroku create churn-api-fastapi

# Pousser l'image Docker de l'API (utiliser Dockerfile.api)
docker build -t churn-api -f Dockerfile.api .
docker tag churn-api registry.heroku.com/churn-api-fastapi/web
docker push registry.heroku.com/churn-api-fastapi/web

# Release et ouvrir l'application
heroku container:release web -a churn-api-fastapi
heroku open -a churn-api-fastapi
```

---

## 3. Utilisation de l'API en Production

Une fois l'API déployée, vous pouvez y accéder via n'importe quel client HTTP. L'interface Swagger de test est disponible sur la route `/docs`.

**Exemple de requête (cURL) :**
```bash
curl -X POST "https://VOTRE_URL_API/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "gender": "Female",
         "SeniorCitizen": 0,
         "Partner": "No",
         "Dependents": "No",
         "tenure": 3,
         "PhoneService": "Yes",
         "MultipleLines": "No",
         "InternetService": "Fiber optic",
         "OnlineSecurity": "No",
         "OnlineBackup": "No",
         "DeviceProtection": "No",
         "TechSupport": "No",
         "StreamingTV": "Yes",
         "StreamingMovies": "Yes",
         "Contract": "Month-to-month",
         "PaperlessBilling": "Yes",
         "PaymentMethod": "Electronic check",
         "MonthlyCharges": 85.70,
         "TotalCharges": 260.0,
         "model_type": "catboost"
     }'
```
