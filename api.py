"""
API REST pour servir le modèle de prédiction.
Lancement : uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

from src.prediction import predict_churn, get_retention_recommendations

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du risque de churn client — 4IASD",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClientInput(BaseModel):
    gender:          Literal['Male', 'Female']
    SeniorCitizen:   Literal[0, 1]
    Partner:         Literal['Yes', 'No']
    Dependents:      Literal['Yes', 'No']
    tenure:          int   = Field(ge=0, le=72)
    PhoneService:    Literal['Yes', 'No']
    MultipleLines:   Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity:  Literal['Yes', 'No', 'No internet service']
    OnlineBackup:    Literal['Yes', 'No', 'No internet service']
    DeviceProtection:Literal['Yes', 'No', 'No internet service']
    TechSupport:     Literal['Yes', 'No', 'No internet service']
    StreamingTV:     Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract:        Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling:Literal['Yes', 'No']
    PaymentMethod:   Literal[
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ]
    MonthlyCharges:  float = Field(ge=0, le=200)
    TotalCharges:    float = Field(ge=0)
    model_type:      Literal['xgboost', 'catboost'] = 'xgboost'

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0,
                "Partner": "No", "Dependents": "No",
                "tenure": 3, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.70, "TotalCharges": 260.0,
                "model_type": "xgboost"
            }
        }


class PredictionOutput(BaseModel):
    prediction:           int
    label:                str
    probability_churn:    float
    probability_no_churn: float
    risk_level:           str
    model_used:           str
    recommendations:      list[str]


@app.get("/")
def root():
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/batch_predict", "/health", "/docs"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "models": ["xgboost", "catboost"]}


@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput):
    try:
        client_dict = client.model_dump(exclude={'model_type'})
        result = predict_churn(client_dict, client.model_type)
        recs = get_retention_recommendations(client_dict,
                                              result['probability_churn'])
        result['recommendations'] = recs
        return result

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modèles non trouvés. Entraînez d'abord les modèles."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
def batch_predict(clients: list[ClientInput]):
    results = []
    for client in clients:
        client_dict = client.model_dump(exclude={'model_type'})
        result = predict_churn(client_dict, client.model_type)
        result['recommendations'] = get_retention_recommendations(
            client_dict, result['probability_churn']
        )
        results.append(result)
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
