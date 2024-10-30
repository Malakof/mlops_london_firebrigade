import os
import warnings
from typing import Dict, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field

from src.model import train_model
from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['api']
logging.info("api Logger loaded")

# Generate a warning to test
warnings.warn("This is a api TEST warning", UserWarning)


app = FastAPI(title="London Fire Brigade MLOPS API /train",
              description="API for London Fire Brigade incident prediction model",
              version="1.0.0")


# Pydantic models for request/response

class TrainModelRequest(BaseModel):
    data_path: str = Field(..., description="Path to the dataset CSV file")
    ml_model_path: str = Field(..., description="Path to save the trained model")
    encoder_path: str = Field(..., description="Path to save the encoder")



@app.post("/train_model",
          summary="Train the prediction model",
          response_model=Dict[str, float])
async def train_model_endpoint(request: TrainModelRequest):
    try:
        metrics = train_model.train_pipeline(request.data_path, request.ml_model_path, request.encoder_path)
        return metrics
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)