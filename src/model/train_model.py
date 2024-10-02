import argparse
import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils import config as cfg
from src.utils.config import logger_train as logging


def _load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        DataFrame: The loaded dataset.
    """
    logging.info("Loading data from file.")
    return pd.read_csv(filepath)


def _preprocess_data(df):
    """
    Preprocesses the DataFrame by encoding categorical features and extracting numeric features.

    Args:
        df (DataFrame): The dataset to preprocess.

    Returns:
        DataFrame: The DataFrame with preprocessed features.
        OneHotEncoder: Fitted OneHotEncoder instance.
    """
    logging.info("Preprocessing data.")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.values
    categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.values

    encoder = OneHotEncoder()
    encoded_categorical = encoder.fit_transform(df[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

    return pd.concat([df[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1), encoder


def _train_model(features, target):
    """
    Trains a linear regression model.

    Args:
        features (DataFrame): The features for training.
        target (Series): The target variable.

    Returns:
        model: Trained model.
    """
    logging.info("Training model.")
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
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
        'MSE': mean_squared_error(y_test, y_pred),
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Max Error': max_error(y_test, y_pred)
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
    Orchestrates the machine learning pipeline from data loading to model saving.
    Returns evaluation metrics or raises an exception with a detailed error message.

    Args:
        data_path (str): Path to the dataset CSV file.
        model_path (str): Path where the trained model will be saved.
        encoder_path (str): Path where the encoder will be saved.

    Returns:
        dict: A dictionary containing various evaluation metrics if successful.

    Raises:
        Exception: An exception with a detailed error message if the pipeline fails.
    """
    try:
        df_base = _load_data(data_path)
        feats, target = df_base.drop('AttendanceTimeSeconds', axis=1), df_base['AttendanceTimeSeconds']
        processed_feats, encoder = _preprocess_data(feats)
        model, X_train, X_test, y_train, y_test = _train_model(processed_feats, target)
        metrics = _evaluate_model(model, X_test, y_test)
        logging.info(f"Model Evaluation Metrics: {metrics}")
        _save_model(model, encoder, model_path, encoder_path)
        return metrics
    except Exception as e:
        error_msg = f"An error occurred during the pipeline execution: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

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
    train_pipeline(args.data_path, args.model_path, args.encoder_path)


if __name__ == "__main__":
    main()
    logging.info("Model training and evaluation complete.")
