from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from xgboost import XGBClassifier

# the model file lives in fraudshield/models/, one folder above api/
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgb_fraud.json"

app = FastAPI(title="Fraud API")

# load the model ONCE, when the server starts (not on every request)
model = XGBClassifier()
model.load_model(str(MODEL_PATH))


@app.get("/")
def home():
    return {"message": "hello, fraud API is alive"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": int(model.n_features_in_),
    }

from pydantic import BaseModel

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

@app.post("/predict")
def predict(tx: Transaction):
    row = pd.DataFrame([tx.model_dump()])
    score = float(model.predict_proba(row)[:, 1][0])
    return {"fraud_score": round(score, 4)}