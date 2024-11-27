import argparse
import os
from time import time

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager, log_model, start_mlflow_session
from src.utils.config import MLFLOW_ENABLED

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['train_model']
logging.info("train_model Logger loaded")

# FOR TESTING
logging.error("TEST ERROR")
logging.warning("TEST ERROR")
logging.info("TEST INFO")
logging.debug("TEST DEBUG")
logging.critical("TEST CRITICAL")

# Generate a warning to test
# warnings.warn("This is a train_model TEST warning", UserWarning)
MSE_METRIC = 'mse'
R2_SCORE_METRIC = 'r2_score'
MAE_METRIC = 'mae'
MAX_ERROR_METRIC = 'max_error'
DATA_SIZE_METRIC = 'original_data_size'
PROCESSED_DATA_SIZE_METRIC = 'processed_data_size'


def _load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        DataFrame: The loaded dataset.
    """
    start_time = time()
    df = pd.read_csv(filepath)
    logging.info("Loading data from file.", metrics={DATA_SIZE_METRIC: df.memory_usage(deep=True).sum()})
    return df


def _preprocess_data(df):
    """
    Preprocesses the DataFrame by encoding categorical features and extracting numeric features.

    Args:
        df (DataFrame): The dataset to preprocess.

    Returns:
        DataFrame: The DataFrame with preprocessed features.
        OneHotEncoder: Fitted OneHotEncoder instance.
    """
    logging.info("Preprocessing data.")  # Log the start of the data preprocessing step

    # Identify numeric and categorical feature columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.values
    categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.values

    # Initialize the OneHotEncoder for categorical features
    encoder = OneHotEncoder()

    # Fit the encoder and transform the categorical features into one-hot encoded arrays
    encoded_categorical = encoder.fit_transform(df[categorical_features]).toarray()

    # Create a DataFrame from the encoded arrays with appropriate column names
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

    # Concatenate numeric and encoded categorical features into a single DataFrame
    processed_data = pd.concat([df[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Log the completion of data preprocessing with the processed data size metric
    logging.info("Data preprocessing completed.",
                 metrics={PROCESSED_DATA_SIZE_METRIC: processed_data.memory_usage(deep=True).sum()})

    # Return the processed data and the fitted encoder
    return processed_data, encoder


def _train_model(features, target):
    """
    Trains a linear regression model.

    Args:
        features (DataFrame): The features for training.
        target (Series): The target variable.

    Returns:
        model: Trained model.
    """
    # Train a linear regression model
    logging.info("Training model.")
    # Initialize a LinearRegression model
    model = LinearRegression()
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Return the trained model and the split data
    return model, X_train, X_test, y_train, y_test


def _evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using the test dataset.

    Args:
        model: The trained model.
        X_test (DataFrame): Test features.
        y_test (Series): True values for test features.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    logging.info("Evaluating model.")
    y_pred = model.predict(X_test)
    metrics = {
        MSE_METRIC: mean_squared_error(y_test, y_pred),
        R2_SCORE_METRIC: r2_score(y_test, y_pred),
        MAE_METRIC: mean_absolute_error(y_test, y_pred),
        MAX_ERROR_METRIC: max_error(y_test, y_pred)
    }
    return metrics


def _save_model(model, encoder, model_path, encoder_path):
    """
    Saves the trained model and encoder to disk.

    Args:
        model: Trained model.
        encoder: Fitted OneHotEncoder.
        model_path (str): Path to save the model.
        encoder_path (str): Path to save the encoder.
    """
    logging.info("Saving model and encoder.")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)


def train_pipeline(data_path, model_path, encoder_path):
    """
    Full pipeline for training a linear regression model from data loading to saving the model.

    Args:
        data_path (str): Path to the CSV file containing training data.
        model_path (str): Path to save the trained model.
        encoder_path (str): Path to save the trained encoder.
    """
    try:
        # Load data
        df_base = _load_data(data_path)
        # Split into features and target
        feats, target = df_base.drop('AttendanceTimeSeconds', axis=1), df_base['AttendanceTimeSeconds']
        # Preprocess features
        processed_feats, encoder = _preprocess_data(feats)
        # Split into training and test sets
        model, X_train, X_test, y_train, y_test = _train_model(processed_feats, target)
        # Evaluate model
        metrics = _evaluate_model(model, X_test, y_test)
        logging.info(f"Model Evaluation Metrics: {metrics}", metrics=metrics)
        # Log model and metrics in MLflow if enabled
        if MLFLOW_ENABLED:
            start_mlflow_session(cfg.MLFLOW_EXPERIMENT_NAME, cfg.MLFLOW_TRACKING_URI)
            log_model(model, encoder, "LFB_MLOPS_Model", model_path, encoder_path, metrics)

        # Save model and encoder locally
        _save_model(model, encoder, model_path, encoder_path)
    except Exception as e:
        # Catch any exceptions and log them
        error_msg = f"An error occurred during the pipeline execution: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    return metrics


def main():
    """Main function to handle command-line arguments and initiate the ML pipeline."""
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model with given dataset.")
    parser.add_argument('--data_path', type=str, help='Path to the dataset CSV file',
                        default=os.path.join(cfg.chemin_data, cfg.fichier_global))
    parser.add_argument('--ml_model_path', type=str, help='Path to save the trained model',
                        default=os.path.join(cfg.chemin_model, cfg.fichier_model))
    parser.add_argument('--encoder_path', type=str, help='Path to save the encoder',
                        default=os.path.join(cfg.chemin_model, 'onehot_encoder.pkl'))

    args = parser.parse_args()
    train_pipeline(args.data_path, args.ml_model_path, args.encoder_path)


if __name__ == "__main__":
    main()
    logging.info("Model training and evaluation complete.")
