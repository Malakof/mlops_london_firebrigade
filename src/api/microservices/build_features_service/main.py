import os
import warnings
from typing import Dict, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.features import build_features
from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['api']
logging.info("api Logger loaded")

# Generate a warning to test
warnings.warn("This is a api TEST warning", UserWarning)


app = FastAPI(title="London Fire Brigade MLOPS API /build_features",
              description="API for London Fire Brigade incident prediction model",
              version="1.0.0")

@app.get("/build_features",
          summary="Build features from processed data",
          response_model=Dict[str, str])
async def build_features_endpoint():
    try:
        build_features.build_features()
        return {"status": "success", "message": "Features built successfully"}
    except Exception as e:
        logging.error(f"Error in feature building: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)