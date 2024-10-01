import os
import logging
from logging.handlers import RotatingFileHandler

# Définition des chemins

chemin_data = '../../data'
chemin_data_ref = '../../data/ref'
chemin_model = '../../models'
# Define the log directory and make sure it exists
log_directory = '../../logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

#définition des urls
url_incident="https://data.london.gov.uk/download/london-fire-brigade-incident-records/f5066d66-c7a3-415f-9629-026fbda61822/LFB%20Incident%20data%20from%202018%20onwards.csv.xlsx"
url_mobilisation="https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%202021%20-%202024.xlsx"


# Définition des nom de fichier
fichier_incident = "incident_data.csv"
fichier_mobilisation = "mobilisation_data.csv"
fichier_stations = "stations.csv"
fichier_calendrier = "calendrier.csv"
fichier_vehicle = "vehicle.csv"
fichier_global = "global_data.csv"
fichier_model = "linear_regression_model.pkl"

# définition de la période d'analyse
years = [2022,2023]

#définition des dimensions

incident_expected_columns = ['IncidentNumber','DateOfCall','CalYear','TimeOfCall','HourOfCall','IncidentGroup','StopCodeDescription',
                            'SpecialServiceType','PropertyCategory','PropertyType','AddressQualifier','Postcode_full','Postcode_district',
                            'UPRN','USRN','IncGeo_BoroughCode','IncGeo_BoroughName','ProperCase','IncGeo_WardCode','IncGeo_WardName','IncGeo_WardNameNew'
                            ,'Easting_m','Northing_m','Easting_rounded','Northing_rounded','Latitude','Longitude','FRS','IncidentStationGround',
                            'FirstPumpArriving_AttendanceTime','FirstPumpArriving_DeployedFromStation','SecondPumpArriving_AttendanceTime',
                            'SecondPumpArriving_DeployedFromStation','NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount','PumpMinutesRounded',
                            'Notional Cost (£)','NumCalls']

mobilisation_expected_columns = ['IncidentNumber','CalYear','HourOfCall','ResourceMobilisationId','Resource_Code','PerformanceReporting',
                                'DateAndTimeMobilised','DateAndTimeMobile','DateAndTimeArrived','TurnoutTimeSeconds','TravelTimeSeconds',
                                'AttendanceTimeSeconds','DateAndTimeLeft','DateAndTimeReturned','DeployedFromStation_Code','DeployedFromStation_Name',
                                'DeployedFromLocation','PumpOrder','PlusCode_Code','PlusCode_Description','DelayCodeId','DelayCode_Description']

CalYear_incident = 'CalYear'
CalYear_mobilisation = 'CalYear'

BandWidth_speed_min = 5
BandWidth_speed_max = 35
BandWidth_AttendanceTimeSeconds_min = 193
BandWidth_AttendanceTimeSeconds_max = 539


def setup_logging():
    # Define the format for the log messages
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    # Create handlers for each logger with specific log files
    handlers = {
        'data_processing': logging.FileHandler(os.path.join(log_directory, 'data_processing.log'), mode='w'),
        'features_building': logging.FileHandler(os.path.join(log_directory, 'build_features.log'), mode='w'),
        'model_training': logging.FileHandler(os.path.join(log_directory, 'train_model.log'), mode='w'),
        'model_predict': logging.FileHandler(os.path.join(log_directory, 'predict_model.log'), mode='w'),
        'model_eval': logging.FileHandler(os.path.join(log_directory, 'eval_model.log'), mode='w')
    }

    # Configure each logger
    loggers = {
        'data_processing': logging.getLogger('data_processing'),
        'features_building': logging.getLogger('build_features'),
        'model_training': logging.getLogger('train_model'),
        'model_predict': logging.getLogger('predict_model'),
        'model_eval': logging.getLogger('eval_model')
    }

    for name, logger in loggers.items():
        logger.setLevel(logging.DEBUG)  # Set the default level
        handler = handlers[name]
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = True  # Prevent log messages from being propagated to the root logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set log level for console
    console_handler.setFormatter(formatter)

    # Rotating file handler
    rotating_handler = RotatingFileHandler(
        os.path.join(log_directory, 'history.log'), maxBytes=1024 * 1024 * 1, backupCount=5) # 1 MB per file, keep 5 old copies
    rotating_handler.setLevel(logging.DEBUG)  # Set log level for file
    rotating_handler.setFormatter(formatter)

    # Get the root logger and add the handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set the lowest log level to handle
    root_logger.addHandler(console_handler)
    root_logger.addHandler(rotating_handler)

    # Return the configured loggers
    return (logging.getLogger('data_processing'), logging.getLogger('build_features'),
            logging.getLogger('train_model'), logging.getLogger('predict_model'), logging.getLogger('eval_model'))

logger_data, logger_features, logger_train, logger_predict, logger_eval = setup_logging()
