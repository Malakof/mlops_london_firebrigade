import os
import warnings
from typing import Dict, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field


from src.data import data_preprocessing
from src.utils.config import LoggingMetricsManager
from src.utils import config as cfg
import os

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['api']
logging.info("api Logger loaded")

# Generate a warning to test
warnings.warn("This is a api TEST warning", UserWarning)

app = FastAPI(title="London Fire Brigade MLOPS API /process_data",
              description="API for London Fire Brigade incident prediction model",
              version="1.0.0")

# Pydantic models for request/response
class DataProcessingRequest(BaseModel):
    data_types: List[str] = Field(..., description="List of data types to process: 'incident' or 'mobilisation'")
    convert_to_pickle: bool = Field(False, description="Whether to convert data to pickle format")

@app.get("/process_data",
         summary="Process incident or mobilisation data",
         response_model=Dict[str, str])
async def process_data(
        background_tasks: BackgroundTasks,
        incident: bool = Query(False, description="Whether to process incident data"),
        mobilisation: bool = Query(False, description="Whether to process mobilisation data"),
        convert_to_pickle: bool = Query(False, description="Whether to convert processed data to pickle format"),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
