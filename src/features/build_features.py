import os

import pandas as pd
from pyproj import Transformer
from src.utils import config as cfg
from haversine import haversine, Unit

def load_data():
    df_incident = pd.read_csv(str(os.path.join(cfg.chemin_data, cfg.fichier_incident)))
    df_mobilisation = pd.read_csv(str(os.path.join(cfg.chemin_data, cfg.fichier_mobilisation)))
    df_stations = pd.read_csv(str(os.path.join(cfg.chemin_data_ref, cfg.fichier_stations)))
    return df_incident, df_mobilisation, df_stations


def clean_data(df_incident, df_mobilisation, df_stations):
    # Rename and drop columns in station data
    df_stations.rename(columns={'name_station': 'DeployedFromStation_Name'}, inplace=True)
    df_stations.columns = ['name_station', 'address_station', 'borough_station', 'latitude_station', 'longitude_station']

    # Clean mobilisation data
    drop_cols_mobilisation = ['CalYear', 'HourOfCall', 'ResourceMobilisationId', 'PerformanceReporting',
                              'DateAndTimeMobile', 'DateAndTimeArrived', 'DateAndTimeLeft', 'DateAndTimeReturned',
                              'PumpOrder', 'PlusCode_Code', 'PlusCode_Description', 'DelayCodeId',
                              'DelayCode_Description',
                              'TurnoutTimeSeconds', 'TravelTimeSeconds', 'DeployedFromLocation', 'BoroughName',
                              'WardName']

    df_mobilisation.drop(drop_cols_mobilisation, axis=1, inplace=True)
    df_mobilisation = pd.merge(df_mobilisation, df_stations, left_on='DeployedFromStation_Name', right_on='name_station', how='left')
    df_mobilisation = df_mobilisation.drop(['name_station', 'address_station', 'borough_station'], axis=1)
    df_mobilisation.dropna(subset=['latitude_station'], inplace=True)

    # Clean incident data
    drop_cols_incident = ['DateOfCall', 'TimeOfCall', 'HourOfCall', 'CalYear', 'IncidentGroup', 'StopCodeDescription',
                          'SpecialServiceType', 'PropertyCategory', 'ProperCase', 'PropertyType', 'AddressQualifier',
                          'Postcode_full', 'IncGeo_WardNameNew', 'Postcode_district', 'UPRN', 'USRN',
                          'IncGeo_BoroughCode',
                          'IncGeo_BoroughName', 'IncGeo_WardCode', 'IncGeo_WardName', 'Easting_m', 'Northing_m', 'FRS',
                          'FirstPumpArriving_AttendanceTime', 'FirstPumpArriving_DeployedFromStation',
                          'SecondPumpArriving_AttendanceTime', 'SecondPumpArriving_DeployedFromStation',
                          'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount', 'PumpMinutesRounded',
                          'Notional Cost (£)', 'NumCalls', 'IncidentStationGround']
                          #changement de nom de colonne
    df_incident.drop(drop_cols_incident, axis=1, inplace=True)
    df_incident = df_incident[df_incident['Latitude'] != 0]

    # Convert British National Grid to WGS84
    bng = 'epsg:27700'
    wgs84 = 'epsg:4326'
    transformer = Transformer.from_crs(bng, wgs84, always_xy=False)
    df_incident[['Latitude', 'Longitude']] = df_incident.apply(
        lambda row: transformer.transform(row['Easting_rounded'], row['Northing_rounded']), axis=1,
        result_type='expand')
    df_incident.drop(['Easting_rounded', 'Northing_rounded'], axis=1, inplace=True)

    return df_incident, df_mobilisation


def merge_datasets(df_incident, df_mobilisation):
    df_merged = pd.merge(df_incident, df_mobilisation, on='IncidentNumber', how='left')
    df_merged.dropna(inplace=True)
    df_merged['distance'] = df_merged.apply(
         lambda row: haversine((row.latitude_station, row.longitude_station),
                               (row.Latitude, row.Longitude), unit=Unit.KILOMETERS), axis=1)
    df_merged['VitesseMoy'] = df_merged['distance'] / (df_merged['AttendanceTimeSeconds'] / 3600)
    df_merged = df_merged[(df_merged['AttendanceTimeSeconds'] >= cfg.BandWidth_AttendanceTimeSeconds_min) & (
                df_merged['AttendanceTimeSeconds'] <= cfg.BandWidth_AttendanceTimeSeconds_max)]
    df_merged = df_merged[
        (df_merged['VitesseMoy'] >= cfg.BandWidth_speed_min) & (df_merged['VitesseMoy'] <= cfg.BandWidth_speed_max)]

    df_merged = df_merged.drop(['IncidentNumber', 'DateAndTimeMobilised', 'DeployedFromStation_Code',
                                'latitude_station', 'longitude_station', 'Latitude', 'Longitude',
                                'Resource_Code',
                                'VitesseMoy'
                                ], axis=1)
    return df_merged

def save_data(df_merged):
    df_merged.to_csv(str(os.path.join(cfg.chemin_data, cfg.fichier_global)), index=False)


def main():
    df_incident, df_mobilisation, df_stations = load_data()
    df_incident, df_mobilisation = clean_data(df_incident, df_mobilisation, df_stations)
    df_merged = merge_datasets(df_incident, df_mobilisation)
    save_data(df_merged)


if __name__ == "__main__":
    main()