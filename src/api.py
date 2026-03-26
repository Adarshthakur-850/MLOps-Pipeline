from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="MLOps Inference API")

# Set MLflow tracking URI
# Default to local ./mlruns if not in Docker
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

class PredictionRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    model_name = "DiabetesRFModel"
    
    try:
        # Note: 'latest' alias might not work with file store in some versions nicely without stages
        # but let's try. If it parses "models:/...", it checks registry.
        # File-based registry is supported in newer MLflow versions.
        print(f"Loading model {model_name} from {tracking_uri}...")
        
        # Try loading the latest version
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Warning: API starting without model. Predictions will fail.")

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    data = pd.DataFrame([request.dict()])
    try:
        prediction = model.predict(data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
