# Take the data file, drop unused columns merge data and build features
import sys
import os
scr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(scr_dir + '/class')

import config as cfg
from DistCalculation import DistCalculation
from DataFrameTransformer import DataFrameTransformer
import pandas as pd
from pyproj import Transformer, transform



# Load the incident and mobilisation data
df_incident = pd.read_csv(cfg.chemin_data + cfg.fichier_incident)
df_mobilisation = pd.read_csv(cfg.chemin_data + cfg.fichier_mobilisation)
df_stations =pd.read_csv(cfg.chemin_data_ref  + cfg.fichier_stations)
df_stations.columns = ['name_station', 'address_station', 'borough_station', 'latitude_station', 'longitude_station']

##############################################
#Cleanning the data mobilisation
##############################################
# Drop the columns that are not needed
df_mobilisation = df_mobilisation.drop(['CalYear','HourOfCall','ResourceMobilisationId','PerformanceReporting',
                                  'DateAndTimeMobile','DateAndTimeArrived','DateAndTimeLeft',
                                  'DateAndTimeReturned','PumpOrder','PlusCode_Code',
                                  'PlusCode_Description','DelayCodeId','DelayCode_Description','TurnoutTimeSeconds','TravelTimeSeconds',
                                  'DeployedFromLocation',
                                  'BoroughName','WardName' ],axis=1)


# Add Coord GPS From Station  to mobilisation data
#AFusion avec la table df_stati des stations
df_mobilisation = pd.merge(df_mobilisation, df_stations , left_on='DeployedFromStation_Name', right_on='name_station', how='left')

##### Supression des colonnes inutiles ####
df_mobilisation = df_mobilisation.drop(['name_station','address_station','borough_station'],axis=1)

##### Supression des lignes dont la latitude n'a pas été trouvé
df_mobilisation = df_mobilisation.dropna(subset=['latitude_station'])

##############################################
#Cleanning the data incident
##############################################
##### Supression des colonnes inutiles ####
df_incident = df_incident.drop(['DateOfCall','TimeOfCall','HourOfCall','CalYear','IncidentGroup','StopCodeDescription','SpecialServiceType',
                                  'PropertyCategory','ProperCase','PropertyType','AddressQualifier','Postcode_full',
                                  'IncGeo_WardNameNew','Postcode_district','UPRN','USRN','IncGeo_BoroughCode',
                                  'IncGeo_BoroughName','IncGeo_WardCode','IncGeo_WardName','Easting_m',
                                  'Northing_m','FRS','FirstPumpArriving_AttendanceTime',
                                  'FirstPumpArriving_DeployedFromStation','SecondPumpArriving_AttendanceTime',
                                  'SecondPumpArriving_AttendanceTime','SecondPumpArriving_DeployedFromStation',
                                  'NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount',
                                  'PumpMinutesRounded','Notional Cost (£)','NumCalls','IncidentStationGround'],axis=1)

#### complétude des latitude longitude à partir de Easting_Rounded et Northing_rounded
# récupération des colonnes Easting_rounded et Northing_rounded (coordonnées au format britanique)
eastings = df_incident['Easting_rounded']
northings = df_incident['Northing_rounded']

#convertion en longitude et latitude au format GPS (wgs84)
bng = 'epsg:27700'
wgs84 ='epsg:7406'
transformer = Transformer.from_crs(bng, wgs84)
res_list_en = transformer.transform(eastings, northings)

# ajout au dataframe des résultats
df_incident['latitude_rounded']=  res_list_en[0]
df_incident['longitude_rounded']=  res_list_en[1]

# complétude des valeurs manquantes des colonnes Latitude et Longitude par latitude_rounded et longitude_rounded
df_incident['Latitude'] = df_incident['Latitude'].fillna(df_incident['latitude_rounded'])
df_incident['Longitude'] = df_incident['Longitude'].fillna(df_incident['longitude_rounded'])


##### Supression de la colonne 'latitude_rounded', 'longitude_rounded', 'Easting_rounded' et 'Northing_rounded'  ####
df_incident = df_incident.drop(['latitude_rounded'],axis=1)
df_incident = df_incident.drop(['longitude_rounded'],axis=1)
df_incident = df_incident.drop(['Easting_rounded'],axis=1)
df_incident = df_incident.drop(['Northing_rounded'],axis=1)

#changement de nom de colonne
df_incident.columns = ['IncidentNumber', 'latitude_incident', 'longitude_incident']

##### Supression des lignes dont la latitude = 0  (603 lignes) ####
df_incident = df_incident[df_incident['latitude_incident']!=0]


# Merge the dataframes on a common column, for example 'incident_id'
df_merged = pd.merge(df_incident, df_mobilisation, left_on='IncidentNumber', right_on='IncidentNumber', how='left')

# Suppression des lignes avec des données manquantes
df_merged = df_merged.dropna()

# Calcul des distance entre la station et le lieu de l'incident
df_merged['distance'] = df_merged.apply(lambda row: 
                                                DistCalculation.haversine(
                                                    row.latitude_station, 
                                                    row.longitude_station, 
                                                    row.latitude_incident, 
                                                    row.longitude_incident)
                                                , axis=1)
# calcul de la vitesse moyenne
transformer = DataFrameTransformer(df_merged)
df_merged = transformer.average_speed('distance', 'AttendanceTimeSeconds', 'VitesseMoy')

#ecrétage
df_merged=df_merged[(df_merged['AttendanceTimeSeconds']>=cfg.BandWidth_AttendanceTimeSeconds_min) & (df_merged['AttendanceTimeSeconds']<=cfg.BandWidth_AttendanceTimeSeconds_max)]
df_merged=df_merged[(df_merged['VitesseMoy']>=cfg.BandWidth_speed_min) & (df_merged['VitesseMoy']<=cfg.BandWidth_speed_max)]


# Drop the columns that are not needed
df_merged = df_merged.drop(['IncidentNumber','DateAndTimeMobilised','DeployedFromStation_Code',
                                  'latitude_station','longitude_station','latitude_incident','longitude_incident',
                                  'Resource_Code',
                                  'VitesseMoy'
                                  ],axis=1)


# Save the merged dataframe to a new CSV file
df_merged.to_csv(cfg.chemin_data + cfg.fichier_global, index=False)