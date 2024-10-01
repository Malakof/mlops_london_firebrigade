#utilisation du model

import sys
import os
from src.utils import config as cfg
import pandas as pd
import joblib
from src.utils.config  import logger_predict as logging
logging.info("Logger loaded")

# Charger le modèle et l'encodeur
model = joblib.load(os.path.join(cfg.chemin_model, cfg.fichier_model))
encoder = joblib.load(os.path.join(cfg.chemin_model, 'onehot_encoder.pkl'))
# Données d'entrée
distance = 1.3  # Exemple de distance
station_de_depart = 'Acton'  # Exemple de station de départ

# Création d'un DataFrame avec les données d'entrée
new_data = pd.DataFrame({
    'distance': [distance],
    'DeployedFromStation_Name': [station_de_depart]
})

# Séparer les caractéristiques
feats_new = new_data

# Variables numériques
numeric_features = feats_new.select_dtypes(include=['int64', 'float64']).columns.values

# Variables catégorielles
categorical_features = feats_new.select_dtypes(exclude=['int64', 'float64']).columns.values

# Prétraitement des colonnes catégorielles
encoded_categorical = encoder.transform(feats_new[categorical_features]).toarray()

# Création d'un DataFrame pour les caractéristiques prétraitées
encoded_df_new = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
feats_new = pd.concat([feats_new[numeric_features].reset_index(drop=True), encoded_df_new.reset_index(drop=True)], axis=1)

# Prédiction
predictions = model.predict(feats_new)

# Affichage des résultats
predictions_df = pd.DataFrame(predictions, columns=['Predicted AttendanceTimeSeconds'])
logging.info(predictions_df)
