# Get the data from internet
# valide datafile
#ATTENTION : Chargement long

import sys
import os
scr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(scr_dir + '/class')

import config as cfg
import pandas as pd

print("=============================================")
print("ATTENTION : Chargement long (fichier de plus de 100 Mo)")
print("=============================================")

url_incident=cfg.url_incident
url_mobilisation=cfg.url_mobilisation
years = cfg.years

def import_incident_data(url_incident, years=None):
    incident_data = pd.read_excel(url_incident)
    if years is not None:
        incident_data = incident_data[incident_data[cfg.CalYear_incident].isin(years)]
    return incident_data

def import_mobilisation_data(url_mobilisation, years=None):
    mobilisation_data = pd.read_excel(url_mobilisation)
    if years is not None:
        mobilisation_data = mobilisation_data[mobilisation_data[cfg.CalYear_mobilisation].isin(years)]
    return mobilisation_data


import_incident_data(url_incident, years).to_csv(cfg.chemin_data + cfg.fichier_incident, index=False)
import_mobilisation_data(url_mobilisation, years).to_csv(cfg.chemin_data + cfg.fichier_mobilisation, index=False)


def validate_file(file_path, expected_columns):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return False
    
    if os.path.getsize(file_path) == 0:
        print(f"Error: {file_path} is empty.")
        return False
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: Could not read {file_path}. Exception: {e}")
        return False
    
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        print(f"Error: {file_path} is missing columns: {', '.join(missing_columns)}")
        return False
    
    print(f"{file_path} is valid.")
    return True




validate_file(cfg.chemin_data + cfg.fichier_incident, cfg.incident_expected_columns)
validate_file(cfg.chemin_data + cfg.fichier_mobilisation, cfg.mobilisation_expected_columns)


