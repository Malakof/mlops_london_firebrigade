import argparse
import sys

import pandas as pd


from src.utils.config import LoggingMetricsManager, load_model_and_encoder

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['predict_model']
logging.info("predict_model Logger loaded")
# TESTING
# logging.error("TEST ERROR")
# logging.warning("TEST ERROR")
# logging.info("TEST INFO")
# logging.debug("TEST DEBUG")
# logging.critical("TEST CRITICAL")

# Generate a warning to test
# warnings.warn("This is a predict_model TEST warning", UserWarning)

# Metric constants
NUM_FEATURES_METRIC = 'num_features'
NUM_PREDICTIONS_METRIC = 'prediction_result'
SUCCESS_METRIC = 'success'




def ascii_happy_dog_face():
    happy_dog_face_lines = [
        "  / \\__",
        " (    @\\___",
        " /         O",
        "/   (_____/",
        "/_____/   U"
    ]
    for line in happy_dog_face_lines:
        logging.info(line)



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
        # Identify numeric and categorical feature columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.values
        categorical_features = data.select_dtypes(exclude=['int64', 'float64']).columns.values

        # Encode categorical features using the OneHotEncoder
        encoded_categorical = encoder.transform(data[categorical_features]).toarray()

        # Create a DataFrame from the encoded categorical features
        encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

        # Concatenate numeric and encoded categorical features into a single DataFrame
        prepared_data = pd.concat([data[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)],
                                  axis=1)

        # Log the completion of feature preparation with the number of features as a metric
        logging.info("Features prepared for prediction.", metrics={
            NUM_FEATURES_METRIC: len(numeric_features) + len(categorical_features)
        })
        return prepared_data
    except Exception as e:
        # Log any errors that occur during feature preparation
        logging.error("Failed to prepare features: " + str(e))
        raise


def make_predict(distance=1.3, station_de_depart='Acton'):
    """
    Conducts the entire prediction process: loads the model and encoder tagged as 'prod',
    prepares features, and makes predictions.

    Args:
        distance (float): Distance for prediction input.
        station_de_depart (str): Deployed station name for prediction input.

    Returns:
        DataFrame: Predictions as a DataFrame.
    """
    try:
        # Load the model and encoder tagged as 'prod'
        model, encoder = load_model_and_encoder('prod')

        # Create a DataFrame for new data with distance and station name
        new_data = pd.DataFrame({
            'distance': [distance],
            'DeployedFromStation_Name': [station_de_depart]
        })

        # Prepare features for prediction using the encoder
        prepared_features = _prepare_features(new_data, encoder)

        # Make predictions using the model
        predictions = model.predict(prepared_features)

        # Convert predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions, columns=['Predicted AttendanceTimeSeconds'])

        logging.info("Prediction completed successfully.")
        logging.debug(f"Predictions: {predictions}")
        logging.info("TEST METRIC", metrics={'XXXXXX_DOG_METRIC_XXXXXX': 6666})
        ascii_happy_dog_face()

    except Exception as e:
        # Log any errors that occur during prediction
        logging.error("Failed to make predictions: " + str(e))
        raise

    return predictions_df

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
        logging.info(f"Predictions: \n{prediction_results}", metrics={SUCCESS_METRIC: True})
    except Exception as e:
        logging.error(f"An error occurred during the prediction process: {e} with args: {args}",
                      metrics={SUCCESS_METRIC: False})
        sys.exit(1)


if __name__ == "__main__":
    main()
