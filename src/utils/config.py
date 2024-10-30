import logging
import os
import uuid
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler


if os.environ.get('DOCKER') == '1':
    #pour docker
    chemin_data = 'data'
    chemin_data_ref = 'data/ref'
    chemin_model = 'models'
    # Define the log directory and make sure it exists
    log_directory = 'logs'
else:
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
from prometheus_client import CollectorRegistry

PUSH_GETAWAY_ENABLED = False
PUSHGATEWAY_URL = 'http://localhost:9091'

# Metrics definitions
registry = CollectorRegistry(auto_describe=True)

from prometheus_client import push_to_gateway, Gauge, Counter, Histogram, CollectorRegistry


class MetricsLogger:
    def __init__(self, logger, module, registry=None, pushgateway_enabled=False):
        self.logger = logger
        self.module = module
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.pushgateway_enabled = pushgateway_enabled
        self.metrics_dict = {}  # Dictionary to store metrics
        self.job_id = f"{datetime.now().strftime('%y%m%d-%H%M%S')}_{uuid.uuid4()}"  # Unique job ID for each instance

    def _create_metric(self, name, metric_type, description):
        """Create or retrieve a Prometheus metric, ensuring names are compliant with Prometheus conventions."""
        metric_name = f"{name}_{self.module}".lower()
        if metric_name not in self.metrics_dict:
            try:
                metric = metric_type(metric_name, description, ['module'], registry=self.registry)
                self.metrics_dict[metric_name] = metric
            except ValueError as e:
                if 'Duplicated' in str(e):
                    logging.warning(
                        f"Duplicate metric registration attempted for {metric_name}. Using existing metric.")
                else:
                    raise
        return self.metrics_dict[metric_name]

    def _push_metrics(self):
        push_to_gateway(PUSHGATEWAY_URL, job=f'{self.module}_{self.job_id}', registry=self.registry)

    def log(self, level, message, metrics=None, metric_types=None):
        getattr(self.logger, level.lower())(message)
        log_metric_name = f"{level.lower()}_logs"
        logs_metric = self._create_metric(log_metric_name, Counter, f"Number of {level.lower()} level logs").labels(
            module=self.module)
        logs_metric.inc()

        if metrics:
            if metric_types is None:
                metric_types = ['Gauge'] * len(metrics)  # Default to Gauge if no types provided
            for (metric_name, value), m_type in zip(metrics.items(), metric_types):
                metric_class = {
                    'Counter': Counter,
                    'Gauge': Gauge,
                    'Histogram': Histogram
                }.get(m_type, Gauge)  # Default to Gauge if unknown type
                metric = self._create_metric(metric_name, metric_class, f"{m_type} for {metric_name}")
                if m_type == 'Histogram':
                    metric.labels(module=self.module).observe(value)
                else:
                    metric.labels(module=self.module).set(value)

        if self.pushgateway_enabled:
            self._push_metrics()

    # Define convenience methods for each log level
    def info(self, message, metrics=None, metric_types=None):
        self.log('info', message, metrics, metric_types)

    def warning(self, message, metrics=None, metric_types=None):
        self.log('warning', message, metrics, metric_types)

    def error(self, message, metrics=None, metric_types=None):
        self.log('error', message, metrics, metric_types)

    def debug(self, message, metrics=None, metric_types=None):
        self.log('debug', message, metrics, metric_types)

    def critical(self, message, metrics=None, metric_types=None):
        self.log('critical', message, metrics, metric_types)


def custom_show_warning(message, category, filename, lineno, file=None, line=None):
    # Access the singleton instance of LoggingMetricsManager
    logging_manager = LoggingMetricsManager()  # This will fetch the existing initialized instance
    metrics_loggers = logging_manager.metrics_loggers

    log_message = f"{filename}:{lineno}: {category.__name__}: {message}"

    # Determine which logger to use based on the filename
    if 'features' in filename:
        logger_key = 'build_features'
    elif 'data' in filename:
        logger_key = 'data_preprocessing'
    elif 'train' in filename:
        logger_key = 'train_model'
    elif 'predict' in filename:
        logger_key = 'predict_model'
    elif 'eval' in filename:
        logger_key = 'eval_model'
    elif 'api' in filename:
        logger_key = 'api'
    else:
        logger_key = None  # Fall back to a default logger if none of the conditions match

    # Get the appropriate logger from the metrics_loggers dictionary
    logger = metrics_loggers.get(logger_key, logging.getLogger())  # Fallback to the root logger

    # Log the warning message using the appropriate MetricsLogger
    if logger:
        logger.warning(log_message)

    # Also log to the root logger
    root_logger = logging.getLogger()


# Set the custom warning handler globally for all warnings
warnings.showwarning = custom_show_warning


class SingletonMeta(type):
    """ A metaclass for creating a Singleton base class. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class LoggingMetricsManager(metaclass=SingletonMeta):
    def __init__(self):
        # clear_prometheus_registry()
        if not hasattr(self, 'initialized'):  # This check prevents reinitialization
            self.log_directory = log_directory
            self.registry = CollectorRegistry()
            self.metrics_loggers = self.setup_logging()
            self.initialized = True

    def setup_logging(self):
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format, datefmt=date_format)

        modules = ['train_model', 'data_preprocessing', 'api', 'eval_model', 'build_features', 'predict_model']
        loggers = {}

        for module in modules:
            file_handler = RotatingFileHandler(
                os.path.join(self.log_directory, f'{module}.log'), 'a', maxBytes=1024 * 1024 * 5, backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger = logging.getLogger(module)
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.propagate = True

            # Each module gets its own MetricsLogger instance with the module name passed
            loggers[module] = MetricsLogger(logger, module, registry=self.registry,
                                            pushgateway_enabled=PUSH_GETAWAY_ENABLED)

        # Configure console and history logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(CONSOLE_LEVEL)
        console_handler.setFormatter(formatter)

        rotating_handler = RotatingFileHandler(
            os.path.join(log_directory, 'history.log'), LOG_MODE, maxBytes=1024 * 1024, backupCount=5
        )
        rotating_handler.setLevel(HISTORY_LEVEL)
        rotating_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(DEFAULT_LEVEL)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(rotating_handler)

        return loggers
