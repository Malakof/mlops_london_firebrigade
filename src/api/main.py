import csv
import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

api = FastAPI()
security = HTTPBasic()

# Dictionnaire des utilisateurs pour l'authentification basique
users = {
    "alice": "wonderland",
    "bob": "builder",
    "clementine": "mandarine"
}

# Modèle Pydantic pour le payload de la requête pour générer un quiz
class QuizRequest(BaseModel):
    test_type: str = Field(..., description="Le type de test souhaité, par exemple 'multiple_choice'")
    categories: List[str] = Field(..., description="Une liste des catégories des questions désirées")
    number_of_questions: int = Field(..., description="Le nombre de questions à inclure dans le quiz")

# Modèle Pydantic pour le payload de la requête pour créer une question
class CreateQuestionRequest(BaseModel):
    question: str = Field(..., description="Le texte de la question à ajouter")
    subject: str = Field(..., description="Le sujet de la question, par exemple 'geography'")
    correct: List[str] = Field(..., description="Une liste contenant les réponses correctes")
    use: str = Field(..., description="Le contexte d'utilisation de la question, par exemple 'exam'")
    responseA: str = Field(..., description="Texte de la réponse A")
    responseB: str = Field(..., description="Texte de la réponse B")
    responseC: str = Field(..., description="Texte de la réponse C")
    responseD: str = Field(..., description="Texte de la réponse D")

# Modèle Pydantic pour les questions renvoyées
class Question(BaseModel):
    question: str = Field(..., description="Le texte de la question posée dans le quiz.")
    subject: str = Field(..., description="Le sujet ou la catégorie de la question, telle que 'mathématiques' ou 'histoire'.")
    correct: List[str] = Field(..., description="Liste contenant les réponses correctes à la question.")
    use: str = Field(..., description="Le contexte d'utilisation de la question, comme 'examen' ou 'entraînement'.")
    responseA: str = Field(..., description="Première option de réponse multiple.")
    responseB: str = Field(..., description="Deuxième option de réponse multiple.")
    responseC: str = Field(..., description="Troisième option de réponse multiple.")
    responseD: Optional[str] = Field(None, description="Quatrième option de réponse multiple, si applicable.")

def extract_user(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Extraire le nom d'utilisateur à partir des informations d'identification fournies.

    Args:
        credentials (HTTPBasicCredentials): L'objet contenant le nom d'utilisateur et le mot de passe.

    Raise:
        HTTPException: Si les informations d'identification ne sont pas valides.

    Returns:
        str: Le nom d'utilisateur extrait.
    """
    username = credentials.username
    password = credentials.password

    if users.get(username) != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non autorisé")

    return username

@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.post("/generate_quiz", response_model=List[Question],
          summary="Génère un quiz basé sur des critères spécifiques",
          description="Crée un quiz aléatoire basé sur le type, les catégories et le nombre de questions spécifiés.",
          response_description="Une liste de questions générées aléatoirement selon les critères fournis.")
async def generate_quiz(request: QuizRequest, username: str = Depends(extract_user)):
    try:
        data = pd.read_csv("questions.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichier de questions requis non trouvé.")

    filtered_data = data[(data['use'] == request.test_type) & (data['subject'].isin(request.categories))]
    if len(filtered_data) < request.number_of_questions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pas assez de questions dispos")

    selected_questions = filtered_data.sample(n=request.number_of_questions)
    selected_questions['correct'] = selected_questions['correct'].apply(lambda x: x.split(','))  # Convertit la chaîne en liste
    selected_questions['responseD'] = selected_questions['responseD'].where(pd.notnull(selected_questions['responseD']), None)
    selected_questions = selected_questions.to_dict(orient='records')
    return selected_questions

@api.post("/create_question", response_model=dict, status_code=status.HTTP_201_CREATED,
          summary="Ajoute une nouvelle question au système",
          description="Permet à un utilisateur authentifié ayant des privilèges administratifs de créer une nouvelle question.",
          response_description="Un message de confirmation indiquant le succès de la création de la question.")
async def create_question(request: CreateQuestionRequest, username: str = Depends(extract_user)):
    # Vérifie si tous les champs requis sont remplis (hormis responseD)
    if not all([request.question, request.subject, request.correct, request.use, request.responseA, request.responseB, request.responseC]):
        return {"message": "question vide non ajoutée"}
    # Prépare les données à être écrites dans le CSV
    new_question = {
        "question": request.question,
        "subject": request.subject,
        "correct": request.correct,  # Ceci devrait déjà être une liste
        "use": request.use,
        "responseA": request.responseA,
        "responseB": request.responseB,
        "responseC": request.responseC,
        "responseD": request.responseD or "",
    }

    # Ajoute la nouvelle question au fichier CSV
    try:
        with open("questions.csv", "a", newline='') as file:
            writer = csv.writer(file)
            # Prépare les données sous forme de liste pour correspondre aux colonnes CSV
            row = [
                new_question['question'],
                new_question['subject'],
                new_question['use'],
                ','.join(new_question['correct']),  # Joint la liste en une chaîne séparée par des virgules
                new_question['responseA'],
                new_question['responseB'],
                new_question['responseC'],
                new_question['responseD'],
            ]
            writer.writerow(row)
        return {"message": "Question créée avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question non sauvegardée, erreur: {str(e)}")

@api.get("/verify", summary="Vérifie l'état de l'API",
         description="Vérifie si l'API est opérationnelle et si les fichiers de configuration nécessaires sont présents.",
         response_description="Un message indiquant si l'API est fonctionnelle ou non.")
async def verify():
    required_file_path = "questions.csv"
    if not os.path.exists(required_file_path):
        raise HTTPException(status_code=404, detail="Fichier de questions requis non trouvé.")

    return {"message": "L'API est fonctionnelle, fichier de questions trouvé."}
