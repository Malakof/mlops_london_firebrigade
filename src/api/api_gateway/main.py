import os
import warnings
import sys
from typing import Dict, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
import httpx

from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['api']
logging.info("api Logger loaded")

# Generate a warning to test
warnings.warn("This is a api TEST warning", UserWarning)

app = FastAPI(title="London Fire Brigade MLOPS API GATEWAY",
              description="API for London Fire Brigade incident prediction model",
              version="1.0.0")
security = HTTPBasic()

# Dummy users for authentication
users = {
    "admin": "fireforce",
    "user": "london123"
}

# Ajouter le chemin du dossier racine au sys.path
current_directory = os.getcwd()
print("Répertoire courant :", current_directory)

# Pydantic models for request/response
class DataProcessingRequest(BaseModel):
    data_types: List[str] = Field(..., description="List of data types to process: 'incident' or 'mobilisation'")
    convert_to_pickle: bool = Field(False, description="Whether to convert data to pickle format")


class TrainModelRequest(BaseModel):
    data_path: str = Field(..., description="Path to the dataset CSV file")
    ml_model_path: str = Field(..., description="Path to save the trained model")
    encoder_path: str = Field(..., description="Path to save the encoder")


class PredictionResponse(BaseModel):
    predicted_attendance_time: float = Field(..., description="Predicted attendance time in seconds")

# Authentication function
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if users.get(username) != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


@app.get("/process_data",
         summary="Process incident or mobilisation data",
         response_model=Dict[str, str])
async def process_data(
        background_tasks: BackgroundTasks,
        incident: bool = Query(False, description="Whether to process incident data"),
        mobilisation: bool = Query(False, description="Whether to process mobilisation data"),
        convert_to_pickle: bool = Query(False, description="Whether to convert processed data to pickle format"),
        username: str = Depends(authenticate_user)
):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://process_data_service:8001/process_data", params={
            "incident": incident,
            "mobilisation": mobilisation,
            "convert_to_pickle": convert_to_pickle
        })
    return response.json()


@app.get("/build_features",
         summary="Build features from processed data",
         response_model=Dict[str, str])
async def build_features(username: str = Depends(authenticate_user)):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get("http://build_features_service:8002/build_features")
    return response.json()


@app.post("/train_model",
          summary="Train the prediction model",
          response_model=Dict[str, float])
async def train_model(request: dict, username: str = Depends(authenticate_user)):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post("http://train_model_service:8003/train_model", json=request)
    return response.json()


@app.get("/predict",
         summary="Make a prediction using the trained model",
         response_model=PredictionResponse)
async def predict(distance: float = Query(..., description="Distance to the incident in kilometers"),
                  station: str = Query(..., description="Departing station name")):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://predict_service:8004/predict",
                                    params={"distance": distance, "station": station})
    return response.json()


@app.get("/health",
         summary="Check the health of the model",
         response_model=Dict[str, str])
async def health_check():
    try:
        # Check if necessary files exist
        required_files = [
            os.path.join(cfg.chemin_data, cfg.fichier_incident),
            os.path.join(cfg.chemin_data, cfg.fichier_mobilisation),
            os.path.join(cfg.chemin_model, cfg.fichier_model)
        ]
        for file in required_files:
            if not os.path.exists(file):
                return {"status": "warning", "message": f"Required file not found: {file}"}

        return {"status": "healthy", "message": "All systems operational"}
    except Exception as e:
        logging.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
