import os
import sys
import argparse
import pandas as pd
import joblib
from src.utils import config as cfg
from src.utils.config import logger_predict as logging


def _load_model_and_encoder():
    """
    Load the machine learning model and encoder from disk.

    Returns:
        tuple: Loaded model and encoder.
    """
    try:
        model = joblib.load(os.path.join(cfg.chemin_model, cfg.fichier_model))
        encoder = joblib.load(os.path.join(cfg.chemin_model, 'onehot_encoder.pkl'))
        logging.info("Model and encoder loaded successfully.")
        return model, encoder
    except Exception as e:
        logging.error("Failed to load model and encoder: " + str(e))
        raise


def _prepare_features(data, encoder):
    """
    Prepare features for prediction by encoding categorical variables and handling numeric features.

    Args:
        data (DataFrame): Data containing the input features.
        encoder (OneHotEncoder): Encoder to transform categorical features.

    Returns:
        DataFrame: Features ready for prediction.
    """
    try:
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.values
        categorical_features = data.select_dtypes(exclude=['int64', 'float64']).columns.values

        encoded_categorical = encoder.transform(data[categorical_features]).toarray()
        encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

        prepared_data = pd.concat([data[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)],
                                  axis=1)
        logging.debug("Features prepared for prediction.")
        return prepared_data
    except Exception as e:
        logging.error("Failed to prepare features: " + str(e))
        raise


def make_predict(distance=1.3, station_de_depart='Acton'):
    """
    Conducts the entire prediction process: loads the model and encoder, prepares features, and makes predictions.

    Args:
        distance (float): Distance for prediction input.
        station_de_depart (str): Deployed station name for prediction input.

    Returns:
        DataFrame: Predictions as a DataFrame.
    """
    try:
        model, encoder = _load_model_and_encoder()
        new_data = pd.DataFrame({
            'distance': [distance],
            'DeployedFromStation_Name': [station_de_depart]
        })

        prepared_features = _prepare_features(new_data, encoder)
        predictions = model.predict(prepared_features)
        predictions_df = pd.DataFrame(predictions, columns=['Predicted AttendanceTimeSeconds'])

        logging.debug("Prediction completed successfully.")
        return predictions_df
    except Exception as e:
        logging.error("Failed to make predictions: " + str(e))
        raise


def main():
    """
    Main function to handle command-line arguments and initiate the prediction process.
    """
    parser = argparse.ArgumentParser(description="Predict Attendance Time based on the given input parameters.")
    parser.add_argument('--distance', type=float, default=1.3, help='Input distance for prediction.')
    parser.add_argument('--station', type=str, default='Acton', help='Deployed station name for prediction.')

    args = parser.parse_args()
    try:
        prediction_results = make_predict(args.distance, args.station)
        logging.info(f"Predictions: \n{prediction_results}")
    except Exception as e:
        logging.error(f"An error occurred during the prediction process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()