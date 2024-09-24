#!/bin/bash

# Fichier shell pour tester l'API FastAPI

# Fonction pour tester l'endpoint /verify
test_verify() {
  echo "Testing /verify endpoint..."
  curl -X GET "http://127.0.0.1:8000/verify"
  echo -e "\n"
}

# Fonction pour tester l'endpoint /generate_quiz avec succès
test_generate_quiz_success() {
  echo "Testing /generate_quiz endpoint (success case)..."
  curl -X POST "http://127.0.0.1:8000/generate_quiz" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "test_type": "Test de positionnement",
          "categories": ["BDD", "Docker"],
          "number_of_questions": 5
        }'
  echo -e "\n"
}

# Fonction pour tester l'endpoint /generate_quiz avec une mauvaise authentification
test_generate_quiz_auth_failure() {
  echo "Testing /generate_quiz endpoint (authentication failure)..."
  curl -X POST "http://127.0.0.1:8000/generate_quiz" \
    -H "Authorization: Basic ZmFrZXVzZXJuYW1lOmZha2VwYXNzd29yZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "test_type": "multiple_choice",
          "categories": ["geography", "history"],
          "number_of_questions": 5
        }'
  echo -e "\n"
}

# Fonction pour tester l'endpoint /generate_quiz avec des paramètres invalides
test_generate_quiz_invalid_params() {
  echo "Testing /generate_quiz endpoint (invalid parameters)..."
  curl -X POST "http://127.0.0.1:8000/generate_quiz" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "test_type": "invalid_type",
          "categories": ["unknown_category"],
          "number_of_questions": 100
        }'
  echo -e "\n"
}

# Fonction pour tester l'endpoint /create_question avec succès
test_create_question_success() {
  echo "Testing /create_question endpoint (success case)..."
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "question": "Quelle est la capitale de l Allemagne? ",
          "subject": "geography",
          "correct": ["Berlin"],
          "use": "multiple_choice",
          "responseA": "Munich",
          "responseB": "Berlin",
          "responseC": "Hamburg",
          "responseD": "Frankfurt"
        }'
  echo -e "\n"
}

# Fonction pour tester l'endpoint /create_question avec une mauvaise authentification
test_create_question_auth_failure() {
  echo "Testing /create_question endpoint (authentication failure)..."
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Authorization: Basic ZmFrZXVzZXJuYW1lOmZha2VwYXNzd29yZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "question": "Quelle est la capitale de l Italie ?",
          "subject": "geography",
          "correct": ["Rome"],
          "use": "multiple_choice",
          "responseA": "Naples",
          "responseB": "Venice",
          "responseC": "Rome",
          "responseD": "Milan"
        }'
  echo -e "\n"
}

# Fonction pour tester l'endpoint /create_question avec des paramètres invalides
test_create_question_invalid_params() {
  echo "Testing /create_question endpoint (invalid parameters)..."
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "question": "",
          "subject": "",
          "correct": [],
          "use": "",
          "responseA": "",
          "responseB": "",
          "responseC": "",
          "responseD": ""
        }'
  echo -e "\n"
}
# Function to test /create_question with partially empty fields
test_create_question_partial_empty() {
  echo "Testing /create_question endpoint (partially empty fields)..."
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: application/json" \
    -d '{
          "question": "What is the largest planet?",
          "subject": "astronomy",
          "correct": [],
          "use": "quiz",
          "responseA": "Jupiter",
          "responseB": "",
          "responseC": "Saturn",
          "responseD": "Mars"
        }'
  echo -e "\n"
}

# Function to test missing Authorization header
test_missing_auth_header() {
  echo "Testing /create_question endpoint (missing Authorization header)..."
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Content-Type: application/json" \
    -d '{
          "question": "What is the smallest planet?",
          "subject": "astronomy",
          "correct": ["Mercury"],
          "use": "test",
          "responseA": "Venus",
          "responseB": "Earth",
          "responseC": "Mars",
          "responseD": "Mercury"
        }'
  echo -e "\n"
}

# Function to test invalid Content-Type header
test_invalid_content_type() {
  echo "Testing /create_question endpoint (invalid Content-Type)..."
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: text/plain" \
    -d '{
          "question": "Which planet is known as the Red Planet?",
          "subject": "astronomy",
          "correct": ["Mars"],
          "use": "test",
          "responseA": "Venus",
          "responseB": "Earth",
          "responseC": "Mars",
          "responseD": "Jupiter"
        }'
  echo -e "\n"
}

# Function to test excessively large input values
test_large_input_values() {
  echo "Testing /create_question endpoint (large input values)..."
  large_string=$(printf 'A%.0s' {1..10000})  # Generates a string of 10,000 'A's
  curl -X POST "http://127.0.0.1:8000/create_question" \
    -H "Authorization: Basic YWxpY2U6d29uZGVybGFuZA==" \
    -H "Content-Type: application/json" \
    -d "{
          \"question\": \"$large_string\",
          \"subject\": \"$large_string\",
          \"correct\": [\"$large_string\"],
          \"use\": \"test\",
          \"responseA\": \"$large_string\",
          \"responseB\": \"$large_string\",
          \"responseC\": \"$large_string\",
          \"responseD\": \"$large_string\"
        }"
  echo -e "\n"
}

# Run all tests
test_verify
test_generate_quiz_success
test_generate_quiz_auth_failure
test_generate_quiz_invalid_params
test_create_question_success
test_create_question_auth_failure
test_create_question_invalid_params
test_create_question_partial_empty
test_missing_auth_header
test_invalid_content_type
test_large_input_values