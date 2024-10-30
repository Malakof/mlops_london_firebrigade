import os
import warnings
from typing import Dict, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field

from src.model import  predict_model
from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['api']
logging.info("api Logger loaded")

# Generate a warning to test
warnings.warn("This is a api TEST warning", UserWarning)


app = FastAPI(title="London Fire Brigade MLOPS API /predict",
              description="API for London Fire Brigade incident prediction model",
              version="1.0.0")


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    predicted_attendance_time: float = Field(..., description="Predicted attendance time in seconds")




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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)