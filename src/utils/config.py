import os
import logging
import warnings
from logging import getLogger
from logging.handlers import RotatingFileHandler

chemin_data = '../../data'
chemin_data_ref = '../../data/ref'
chemin_model = '../../models'
# Define the log directory and make sure it exists
log_directory = '../../logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# définition des urls
url_incident = "https://data.london.gov.uk/download/london-fire-brigade-incident-records/f5066d66-c7a3-415f-9629-026fbda61822/LFB%20Incident%20data%20from%202018%20onwards.csv.xlsx"
url_mobilisation = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%202021%20-%202024.xlsx"

# Définition des nom de fichier
fichier_incident = "incident_data.csv"
fichier_mobilisation = "mobilisation_data.csv"
fichier_stations = "stations.csv"
fichier_calendrier = "calendrier.csv"
fichier_vehicle = "vehicle.csv"
fichier_global = "global_data.csv"
fichier_model = "linear_regression_model.pkl"

# définition de la période d'analyse
years = [2022, 2023]

# définition des dimensions

incident_expected_columns = ['IncidentNumber', 'DateOfCall', 'CalYear', 'TimeOfCall', 'HourOfCall', 'IncidentGroup',
                             'StopCodeDescription',
                             'SpecialServiceType', 'PropertyCategory', 'PropertyType', 'AddressQualifier',
                             'Postcode_full', 'Postcode_district',
                             'UPRN', 'USRN', 'IncGeo_BoroughCode', 'IncGeo_BoroughName', 'ProperCase',
                             'IncGeo_WardCode', 'IncGeo_WardName', 'IncGeo_WardNameNew'
    , 'Easting_m', 'Northing_m', 'Easting_rounded', 'Northing_rounded', 'Latitude', 'Longitude', 'FRS',
                             'IncidentStationGround',
                             'FirstPumpArriving_AttendanceTime', 'FirstPumpArriving_DeployedFromStation',
                             'SecondPumpArriving_AttendanceTime',
                             'SecondPumpArriving_DeployedFromStation', 'NumStationsWithPumpsAttending',
                             'NumPumpsAttending', 'PumpCount', 'PumpMinutesRounded',
                             'Notional Cost (£)', 'NumCalls']

mobilisation_expected_columns = ['IncidentNumber', 'CalYear', 'HourOfCall', 'ResourceMobilisationId', 'Resource_Code',
                                 'PerformanceReporting',
                                 'DateAndTimeMobilised', 'DateAndTimeMobile', 'DateAndTimeArrived',
                                 'TurnoutTimeSeconds', 'TravelTimeSeconds',
                                 'AttendanceTimeSeconds', 'DateAndTimeLeft', 'DateAndTimeReturned',
                                 'DeployedFromStation_Code', 'DeployedFromStation_Name',
                                 'DeployedFromLocation', 'PumpOrder', 'PlusCode_Code', 'PlusCode_Description',
                                 'DelayCodeId', 'DelayCode_Description']

CalYear_incident = 'CalYear'
CalYear_mobilisation = 'CalYear'

BandWidth_speed_min = 5
BandWidth_speed_max = 35
BandWidth_AttendanceTimeSeconds_min = 193
BandWidth_AttendanceTimeSeconds_max = 539

LOG_MODE = 'a'
DEFAULT_LEVEL = logging.DEBUG
CONSOLE_LEVEL = logging.DEBUG
HISTORY_LEVEL = logging.DEBUG

def setup_logging():
    # Define the format for the log messages
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Create rotating handlers for each logger with specific log files
    handlers = {
        'data_preprocessing': RotatingFileHandler(os.path.join(log_directory, 'data_preprocessing.log'), LOG_MODE,
                                               maxBytes=1024 * 100, backupCount=5),  # 5 MB per file, keep 5 backups
        'build_features': RotatingFileHandler(os.path.join(log_directory, 'build_features.log'), LOG_MODE,
                                                 maxBytes=1024 * 100, backupCount=5),
        'train_model': RotatingFileHandler(os.path.join(log_directory, 'train_model.log'), LOG_MODE,
                                              maxBytes=1024 * 100, backupCount=5),
        'predict_model': RotatingFileHandler(os.path.join(log_directory, 'predict_model.log'), LOG_MODE,
                                             maxBytes=1024 * 100, backupCount=5),
        'eval_model': RotatingFileHandler(os.path.join(log_directory, 'eval_model.log'), LOG_MODE,
                                          maxBytes=1024 * 100, backupCount=5),
        'api': RotatingFileHandler(os.path.join(log_directory, 'api.log'), LOG_MODE,
                                          maxBytes=1024 * 100, backupCount=5)
    }

    # Configure each logger
    loggers = {
        'data_preprocessing': logging.getLogger('data_preprocessing'),
        'build_features': logging.getLogger('build_features'),
        'train_model': logging.getLogger('train_model'),
        'predict_model': logging.getLogger('predict_model'),
        'eval_model': logging.getLogger('eval_model'),
        'api' : logging.getLogger('api')
    }

    for name, logger in loggers.items():
        logger.setLevel(DEFAULT_LEVEL)  # Set the default level
        handler = handlers[name]
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = True  # Prevent log messages from being propagated to the root logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LEVEL)  # Set log level for console
    console_handler.setFormatter(formatter)

    # Rotating file handler for root logger history
    rotating_handler = RotatingFileHandler(
        os.path.join(log_directory, 'history.log'), LOG_MODE, maxBytes=1024 * 1024 * 1,
        backupCount=5)  # 1 MB per file, keep 5 old copies
    rotating_handler.setLevel(HISTORY_LEVEL)  # Set log level for file
    rotating_handler.setFormatter(formatter)

    # Get the root logger and add the handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(DEFAULT_LEVEL)  # Set the lowest log level to handle
    root_logger.addHandler(console_handler)
    root_logger.addHandler(rotating_handler)

    # Return the configured loggers
    return (logging.getLogger('data_preprocessing'), logging.getLogger('build_features'),
            logging.getLogger('train_model'), logging.getLogger('predict_model'), logging.getLogger('eval_model'),
            logging.getLogger('api'))

# Setup logging and get loggers
logger_data, logger_features, logger_train, logger_predict, logger_eval, logger_api = setup_logging()

# Define a custom warning handler
def custom_show_warning(message, category, filename, lineno, file=None, line=None):
    log_message = f"{filename}:{lineno}: {category.__name__}: {message}"

    # Example: Decide based on the filename which logger to use
    if 'features' in filename:
        logger = logger_features
    elif 'data' in filename:
        logger = logger_data
    elif 'train' in filename:
        logger = logger_train
    elif 'predict' in filename:
        logger = logger_predict
    elif 'eval' in filename:
        logger = logger_eval
    elif 'api' in filename:
        logger = logger_api
    else:
        logger = logging.getLogger()  # Default to the root logger if no specific logger is found

    # Log to the specific logger
    logger.warning(log_message)

    # Also log to the root logger
    root_logger = logging.getLogger()


# Set the custom warning handler
warnings.showwarning = custom_show_warning

# Generate a warning to test
# warnings.warn("This is a TEST warning", UserWarning)
