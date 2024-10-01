import sys
import os

from src.utils import config as cfg
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error

import joblib
from src.utils.config  import logger_train as logging
logging.info("Logger loaded")

df_base = pd.read_csv(os.path.join(str(cfg.chemin_data), str(cfg.fichier_global)))

# Split the data into features and target
feats=df_base.drop('AttendanceTimeSeconds', axis=1)
target=df_base['AttendanceTimeSeconds']

# Variables numériques
numeric_features = feats.select_dtypes(include=['int64','float64']).columns.values
# numeric_features = ['distance']

# Variables catégorielles
categorical_features = feats.select_dtypes(exclude=['int64','float64']).columns.values
# categorical_features = ['DeployedFromStation_Name']

# Prétraitement des colonnes catégorielles
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(feats[categorical_features]).toarray()

# Création d'un DataFrame pour les caractéristiques prétraitées
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
feats = pd.concat([feats[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Définition des modèles d'apprentissage
model_name = 'Linear Regression'
model=LinearRegression()

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, os.path.join(cfg.chemin_model, cfg.fichier_model))
joblib.dump(encoder, os.path.join(cfg.chemin_model, 'onehot_encoder.pkl'))
# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation de la performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
max_err = max_error(y_test, y_pred)

logging.info(f'{model_name}')
logging.info(f'Mean Squared Error: {mse:.4f}')
logging.info(f'R2 Score: {r2:.4f}')
logging.info(f'Mean Absolute Error: {mae:.4f}')
logging.info(f'Max Error: {max_err:.4f}')