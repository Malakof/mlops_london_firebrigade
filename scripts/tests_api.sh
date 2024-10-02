#!/bin/bash

# Fichier shell pour tester l'API FastAPI

test_health() {
  echo "Test Health check..."
  curl -X GET "http://127.0.0.1:8000/health"
  echo -e "\n"
}

test_predict() {
  echo "Test Predict..."
  curl 'http://localhost:8000/predict?distance=2.5&station=Acton'
}

test_process_full() {
  echo "Test process full..."
  curl -X 'GET' \
  'http://127.0.0.1:8000/process_data?incident=false&mobilisation=false&convert_to_pickle=true' \
  -u 'admin:fireforce' \
  -H 'accept: application/json'
  echo -e "\n"
}

test_build_features() {
  echo "Test process full..."

  curl -X 'POST' \
    'http://127.0.0.1:8000/build_features' \
    -u 'admin:fireforce' \
    -H 'accept: application/json' \
    -d ''
  echo -e "\n"
}

test_train_model() {
  echo "Test process full..."


  curl -X 'POST' \
    'http://127.0.0.1:8000/train_model' \
    -u 'admin:fireforce' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "data_path": "../../data/global_data.csv",
    "ml_model_path": "../../models/linear_regression_model.pkl",
    "encoder_path": "../../models/onehot_encoder.pkl"
  }'
  echo -e "\n"
}

#TODO: Fonction pour tester les endpoint avec une mauvaise authentification
#TODO: Fonction pour tester avec des paramètres invalides


# Run all tests
test_health
test_process_full
test_build_features
test_train_model
test_predict
