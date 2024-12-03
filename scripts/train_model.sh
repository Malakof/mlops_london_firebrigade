#!/bin/bash
# export PYTHONPATH="/Users/richard/Code/mlops_london_firebrigade:$PYTHONPATH"
# Fichier shell pour tester l'API FastAPI

test_health() {
  echo "--> Health check..."
  curl -X GET "http://127.0.0.1:8000/health"
  echo -e "\n"
}

test_predict() {
  echo "--> Predict..."
  curl 'http://localhost:8000/predict?distance=2.5&station=Acton'
  echo -e "\n"
}

test_data_preprocessing() {
  echo "--> Data download and processing full..."
  curl -X 'GET' \
  'http://127.0.0.1:8000/process_data?incident=true&mobilisation=true&convert_to_pickle=false' \
  -u 'admin:fireforce' \
  -H 'accept: application/json'
  echo -e "\n"
}

test_build_features() {
  echo "--> Build Features..."

  curl -X 'GET' \
    'http://127.0.0.1:8000/build_features' \
    -u 'admin:fireforce' \
    -H 'accept: application/json' \
    -d ''
  echo -e "\n"
}

test_train_model_mlflow_docker() {
  echo "--> Train with prod model from mlflow..."

  curl -X 'POST' \
    'http://127.0.0.1:8000/train_model' \
    -u 'admin:fireforce' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "data_path": "./data/global_data.csv",
    "ml_model_path": "./models/linear_regression_model.pkl",
    "encoder_path": "./models/onehot_encoder.pkl"
  }'
  echo -e "\n"
}

#TODO: Fonction pour tester les endpoint avec une mauvaise authentification
#TODO: Fonction pour tester avec des paramètres invalides


# Run all tests
set -x
test_data_preprocessing
sleep 60
test_build_features
test_train_model_mlflow_docker
test_predict
test_health
