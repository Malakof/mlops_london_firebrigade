import os
import warnings
from typing import Dict, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from src.data import data_preprocessing
from src.features import build_features
from src.model import train_model, predict_model
from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['api']
logging.info("api Logger loaded")

# Generate a warning to test
warnings.warn("This is a api TEST warning", UserWarning)

app = FastAPI(title="London Fire Brigade MLOPS API",
              description="API for London Fire Brigade incident prediction model",
              version="1.0.0")
security = HTTPBasic()

# Dummy users for authentication
users = {
    "admin": "fireforce",
    "user": "london123"
}


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
    try:
        processing_errors = []

        # Process 'incident' data if set to true
        if incident:
            try:
                background_tasks.add_task(data_preprocessing.process_data, 'incident')
                logging.info("Started processing incident data.")
            except Exception as e:
                processing_errors.append(f"Failed to process incident: {str(e)}")
                logging.error(f"Error processing incident: {str(e)}")

        # Process 'mobilisation' data if set to true
        if mobilisation:
            try:
                background_tasks.add_task(data_preprocessing.process_data, 'mobilisation')
                logging.info("Started processing mobilisation data.")
            except Exception as e:
                processing_errors.append(f"Failed to process mobilisation: {str(e)}")
                logging.error(f"Error processing mobilisation: {str(e)}")

        # Convert to pickle if requested and after processing
        if convert_to_pickle:
            file_paths = []
            if incident:
                file_paths.append(os.path.join(cfg.chemin_data, cfg.fichier_incident))
            if mobilisation:
                file_paths.append(os.path.join(cfg.chemin_data, cfg.fichier_mobilisation))

            output_dir = cfg.chemin_data_ref
            try:
                pickle_errors = data_preprocessing.convert_to_pickle(file_paths, output_dir)
                if pickle_errors:
                    processing_errors.extend(pickle_errors)
            except Exception as e:
                processing_errors.append(f"Failed to convert data to pickle: {str(e)}")
                logging.error(f"Error in pickle conversion: {str(e)}")

        # Final response based on success or errors
        if processing_errors:
            return {"status": "partial success",
                    "message": f"Processing completed with errors: {processing_errors}"}
        else:
            return {"status": "success",
                    "message": "All data jobs submitted successfully"}

    except Exception as e:
        logging.error(f"Error in data processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/build_features",
         summary="Build features from processed data",
         response_model=Dict[str, str])
async def build_features_endpoint(username: str = Depends(authenticate_user)):
    try:
        build_features.build_features()
        return {"status": "success", "message": "Features built successfully"}
    except Exception as e:
        logging.error(f"Error in feature building: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_model",
          summary="Train the prediction model",
          response_model=Dict[str, float])
async def train_model_endpoint(request: TrainModelRequest, username: str = Depends(authenticate_user)):
    try:
        metrics = train_model.train_pipeline(request.data_path, request.ml_model_path, request.encoder_path)
        return metrics
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict",
         summary="Make a prediction using the trained model",
         response_model=PredictionResponse)
async def predict(distance: float = Query(..., description="Distance to the incident in kilometers"),
                  station: str = Query(..., description="Departing station name")):
    try:
        if not (distance and station):
            raise ValueError("Invalid data_type. Distance and/or station missing")
        prediction = predict_model.make_predict(distance, station)
        logging.info(f"Predicted attendance time: {prediction} for {distance} km from {station}")
        return PredictionResponse(predicted_attendance_time=prediction.iloc[0, 0])
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
