import os
import warnings

import pandas as pd
from haversine import haversine, Unit
from pyproj import Transformer

from src.utils import config as cfg
# Initialize logging
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['build_features']
logging.info("build_features Logger loaded")
# TESTING
# logging.error("TEST ERROR")
# logging.warning("TEST ERROR")
# logging.info("TEST INFO")
# logging.debug("TEST DEBUG")
# logging.critical("TEST CRITICAL")

# Generate a warning to test
# warnings.warn("This is a build_features TEST warning", UserWarning)

INCIDENT_ROWS_METRIC = "incident_rows"
MOBILISATION_ROWS_METRIC = "mobilisation_rows"
INCIDENT_CLEANED_ROWS_METRIC = "incident_cleaned_rows"
MOBILISATION_CLEANED_ROWS_METRIC = "mobilisation_cleaned_rows"
STATION_ROWS_METRIC = "station_rows"
SAVED_FILE_SIZE_METRIC = "saved_file_size"
SUCCESS_METRIC = "success"

def _load_data():
    """
    Loads data from CSV files into pandas DataFrames.
    Returns:
        tuple: Tuple containing DataFrame for incidents, mobilisations, and stations.
    """
    try:
        df_incident = pd.read_csv(str(os.path.join(cfg.chemin_data, cfg.fichier_incident)))
        df_mobilisation = pd.read_csv(str(os.path.join(cfg.chemin_data, cfg.fichier_mobilisation)))
        df_stations = pd.read_csv(str(os.path.join(cfg.chemin_data_ref, cfg.fichier_stations)))
        logging.info("Data loaded successfully.", metrics={
            INCIDENT_ROWS_METRIC: len(df_incident),
            MOBILISATION_ROWS_METRIC: len(df_mobilisation),
            STATION_ROWS_METRIC: len(df_stations)
        })
        return df_incident, df_mobilisation, df_stations
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def _clean_data(df_incident, df_mobilisation, df_stations):
    """
    Cleans data by renaming, dropping unnecessary columns and merging DataFrames.
    Args:
        df_incident (DataFrame): Incident data.
        df_mobilisation (DataFrame): Mobilisation data.
        df_stations (DataFrame): Station data.
    Returns:
        tuple: Tuple containing cleaned DataFrames for incidents and mobilisations.
    """
    try:
        # Rename and redefine columns in station data
        df_stations.rename(columns={'name_station': 'DeployedFromStation_Name'}, inplace=True)
        df_stations.columns = ['name_station', 'address_station', 'borough_station', 'latitude_station',
                               'longitude_station']

        # Drop unnecessary columns from mobilisation data and merge with station data
        drop_cols_mobilisation = ['CalYear', 'HourOfCall', 'ResourceMobilisationId', 'PerformanceReporting',
                                  'DateAndTimeMobile', 'DateAndTimeArrived', 'DateAndTimeLeft', 'DateAndTimeReturned',
                                  'PumpOrder', 'PlusCode_Code', 'PlusCode_Description', 'DelayCodeId',
                                  'DelayCode_Description', 'TurnoutTimeSeconds', 'TravelTimeSeconds',
                                  'DeployedFromLocation', 'BoroughName', 'WardName']
        df_mobilisation.drop(drop_cols_mobilisation, axis=1, inplace=True)
        df_mobilisation = pd.merge(df_mobilisation, df_stations, left_on='DeployedFromStation_Name',
                                   right_on='name_station', how='left')
        df_mobilisation.drop(['name_station', 'address_station', 'borough_station'], axis=1, inplace=True)
        df_mobilisation.dropna(subset=['latitude_station'], inplace=True)

        # Clean incident data by dropping unnecessary columns and filtering
        drop_cols_incident = ['DateOfCall', 'TimeOfCall', 'HourOfCall', 'CalYear', 'IncidentGroup',
                              'StopCodeDescription',
                              'SpecialServiceType', 'PropertyCategory', 'ProperCase', 'PropertyType',
                              'AddressQualifier',
                              'Postcode_full', 'IncGeo_WardNameNew', 'Postcode_district', 'UPRN', 'USRN',
                              'IncGeo_BoroughCode', 'IncGeo_BoroughName', 'IncGeo_WardCode', 'IncGeo_WardName',
                              'Easting_m', 'Northing_m', 'FRS',
                              'FirstPumpArriving_AttendanceTime', 'FirstPumpArriving_DeployedFromStation',
                              'SecondPumpArriving_AttendanceTime', 'SecondPumpArriving_DeployedFromStation',
                              'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount', 'PumpMinutesRounded',
                              'Notional Cost (£)', 'NumCalls', 'IncidentStationGround']
        df_incident.drop(drop_cols_incident, axis=1, inplace=True)
        df_incident = df_incident[df_incident['Latitude'] != 0]

        # Convert British National Grid to WGS84 coordinates
        bng = 'epsg:27700'
        wgs84 = 'epsg:4326'
        transformer = Transformer.from_crs(bng, wgs84, always_xy=False)
        df_incident[['Latitude', 'Longitude']] = df_incident.apply(
            lambda row: transformer.transform(row['Easting_rounded'], row['Northing_rounded']), axis=1,
            result_type='expand')
        df_incident.drop(['Easting_rounded', 'Northing_rounded'], axis=1, inplace=True)

        logging.info("Data cleaning completed successfully.", metrics={
            INCIDENT_CLEANED_ROWS_METRIC: len(df_incident),
            MOBILISATION_CLEANED_ROWS_METRIC: len(df_mobilisation)
        })
        return df_incident, df_mobilisation
    except Exception as e:
        logging.error(f"Data cleaning failed: {e}")
        raise


def _merge_datasets(df_incident, df_mobilisation):
    """
    Merges incident and mobilisation datasets, calculates distances and filters data based on defined criteria.
    Args:
        df_incident (DataFrame): Incident data.
        df_mobilisation (DataFrame): Mobilisation data.
    Returns:
        DataFrame: Merged and filtered dataset.
    """
    try:
        df_merged = pd.merge(df_incident, df_mobilisation, on='IncidentNumber', how='left')
        df_merged.dropna(inplace=True)
        df_merged['distance'] = df_merged.apply(
            lambda row: haversine((row['latitude_station'], row['longitude_station']),
                                  (row['Latitude'], row['Longitude']), unit=Unit.KILOMETERS), axis=1)
        df_merged['VitesseMoy'] = df_merged['distance'] / (df_merged['AttendanceTimeSeconds'] / 3600)
        df_merged = df_merged[(df_merged['AttendanceTimeSeconds'] >= cfg.BandWidth_AttendanceTimeSeconds_min) &
                              (df_merged['AttendanceTimeSeconds'] <= cfg.BandWidth_AttendanceTimeSeconds_max)]
        df_merged = df_merged[
            (df_merged['VitesseMoy'] >= cfg.BandWidth_speed_min) & (df_merged['VitesseMoy'] <= cfg.BandWidth_speed_max)]

        df_merged = df_merged.drop(['IncidentNumber', 'DateAndTimeMobilised', 'DeployedFromStation_Code',
                                    'latitude_station', 'longitude_station', 'Latitude', 'Longitude',
                                    'Resource_Code', 'VitesseMoy'], axis=1)
        logging.info("Dataset merge and filter completed.", metrics={
            'merged_rows': len(df_merged)
        })
        return df_merged
    except Exception as e:
        logging.error(f"Failed to merge datasets: {e}")
        raise


def _save_data(df_merged):
    """
    Saves the merged DataFrame to a CSV file.
    Args:
        df_merged (DataFrame): The final merged and cleaned dataset.
    """
    try:
        df_merged.to_csv(os.path.join(cfg.chemin_data, cfg.fichier_global), index=False)
        logging.info("Data saved to CSV successfully.", metrics={
            SAVED_FILE_SIZE_METRIC: os.path.getsize(os.path.join(cfg.chemin_data, cfg.fichier_global))
        })
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise


def build_features():
    """
    Orchestrates the process to load, clean, merge, and save datasets.
    This function handles the entire feature building process which includes
    loading data, cleaning it, merging it, and saving the final result.

    Raises:
        Exception: Raises an exception with a detailed error message if any step fails.
    """
    try:
        # Load data
        df_incident, df_mobilisation, df_stations = _load_data()
        # Clean data
        df_incident, df_mobilisation = _clean_data(df_incident, df_mobilisation, df_stations)
        # Merge datasets
        df_merged = _merge_datasets(df_incident, df_mobilisation)
        # Save merged data
        _save_data(df_merged)

        logging.info("Feature building process completed successfully.", metrics={SUCCESS_METRIC :True})

    except Exception as e:
        error_msg = f"An error occurred in the feature building process: {e}"
        logging.error(error_msg, metrics={'success':False})
        raise Exception(error_msg)


def main():
    build_features()


if __name__ == "__main__":
    main()
