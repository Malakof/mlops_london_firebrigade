import logging
import os
import uuid
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler

# TODO: Find a way to remove this
if os.environ.get('DOCKER') == '1':
    #pour docker
    chemin_data = 'data'
    chemin_data_ref = 'data/ref'
    chemin_model = 'models'
    # Define the log directory and make sure it exists
    log_directory = 'logs'
    DEFAULT_PUSHGATEWAY_URL = 'http://pushgateway_service:9091'
    DEFAULT_MLFLOW_TRACKING_URI = 'http://mlflow_service:9092'
else:
    chemin_data = '../../data'
    chemin_data_ref = '../../data/ref'
    chemin_model = '../../models'
    # Define the log directory and make sure it exists
    log_directory = '../../logs'
    DEFAULT_PUSHGATEWAY_URL = 'http://127.0.0.1:9091'
    DEFAULT_MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

def ascii_happy_dog_face():
    happy_dog_face_lines = [
        "  / \\__",
        " (    @\\___",
        " /         O",
        "/   (_____/",
        "/_____/   U"
    ]
    for line in happy_dog_face_lines:
        warnings.warn(line, UserWarning)


# définition des urls
#url_incident = "https://data.london.gov.uk/download/london-fire-brigade-incident-records/73728cf4-b70e-48e2-9b97-4e4341a2110d/LFB%20Incident%20data%20from%202009%20-%202017.csv"
url_incident = "https://data.london.gov.uk/download/london-fire-brigade-incident-records/f5066d66-c7a3-415f-9629-026fbda61822/LFB%20Incident%20data%20from%202018%20onwards.csv.xlsx"

#url_mobilisation = "https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%202021%20-%202024.xlsx"
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


# Environment variable handling for Push Gateway
PUSH_GATEWAY_ENABLED = os.getenv('PUSH_GATEWAY_ENABLED', 'False').lower() in ('true', 'True', '1', 't')
if PUSH_GATEWAY_ENABLED:
    PUSHGATEWAY_URL = os.getenv('PUSHGATEWAY_URL', DEFAULT_PUSHGATEWAY_URL)
else:
    logging.warning(f"PUSH_GATEWAY_ENABLED is False or not present and will be DEACTIVATED")

# Environment variable handling for MLflow
MLFLOW_ENABLED = os.getenv('MLFLOW_ENABLED', 'False').lower() in ('true', 'True', '1', 't')
if MLFLOW_ENABLED:
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', DEFAULT_MLFLOW_TRACKING_URI)
else:
    logging.warning(f"MLFLOW_ENABLED is False or not present and will be DEACTIVATED")
MLFLOW_EXPERIMENT_NAME = 'LFB_MLOPS'  # Name of the MLflow experiment

# Metrics definitions
registry = CollectorRegistry(auto_describe=True)

from prometheus_client import push_to_gateway, Gauge, Counter, Histogram, CollectorRegistry


class MetricsLogger:
    def __init__(self, logger, module, registry=None, pushgateway_enabled=False):
        """
        Initializes a MetricsLogger instance.

        Args:
            logger: The logger instance to be used for logging.
            module: The name of the module where metrics are being logged.
            registry: An optional CollectorRegistry instance for registering metrics.
                      If not provided, a new CollectorRegistry is created.
            pushgateway_enabled: A boolean indicating if the Pushgateway is enabled for pushing metrics to prometheus.
        """
        self.logger = logger
        self.module = module
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.pushgateway_enabled = pushgateway_enabled
        self.metrics_dict = {}  # Dictionary to store metrics
        self.job_id = f"{datetime.now().strftime('%y%m%d-%H%M%S')}_{uuid.uuid4()}"  # Unique job ID for each instance

    def _get_or_create_metric(self, name, metric_type, description):
        """
        Retrieves an existing metric or creates a new one if it does not exist.

        Args:
            name (str): The base name of the metric.
            metric_type (type): The class type of the metric (e.g., Gauge, Counter, Histogram).
            description (str): A brief description of the metric.

        Returns:
            Metric: The existing or newly created metric object.

        Raises:
            ValueError: If a non-duplicate error occurs during metric creation.
        """
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
        """
        Logs a message with the given level and metrics.

        Args:
            level (str): The log level (e.g. 'debug', 'info', 'warning', 'error', 'critical').
            message (str): The log message.
            metrics (dict[str, float], optional): A dictionary of metrics to log. Defaults to None.
            metric_types (list[str], optional): A list of metric types corresponding to the values in `metrics`. Defaults to None.

        Raises:
            ValueError: If a non-duplicate error occurs during metric creation.
        """
        getattr(self.logger, level.lower())(message)
        log_metric_name = f"{level.lower()}_logs"
        logs_metric = self._get_or_create_metric(log_metric_name, Counter, f"Number of {level.lower()} level logs").labels(
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
                metric = self._get_or_create_metric(metric_name, metric_class, f"{m_type} for {metric_name}")
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
    """
    Custom warning handler that logs warnings to a MetricsLogger instance.

    This function is called by the warnings module when a warning is raised. It logs the warning message to a
    MetricsLogger instance, which is determined based on the filename in which the warning was raised. The logger is
    obtained from a dictionary that maps filenames to logger instances.

    The warning message is logged at the WARNING level, and the logger is also configured to log to the root logger.

    Parameters
    ----------
    message : str
        The warning message
    category : warnings.WarningMessage
        The warning category
    filename : str
        The filename in which the warning was raised
    lineno : int
        The line number on which the warning was raised
    file : file-like object, optional
        The file object to write to, by default None
    line : str, optional
        The line of source code that triggered the warning, by default None

    Returns
    -------
    None
    """
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
        """
        Ensure that only one instance of a Singleton class is created.

        This method is called when an instance of the class is requested. If an instance does not exist, one is created
        and stored in the `_instances` dictionary. Subsequent calls to `__call__` will return the existing instance.

        :param args: Arguments to pass to the class constructor
        :param kwargs: Keyword arguments to pass to the class constructor
        :return: The instance of the class
        :rtype: cls
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class LoggingMetricsManager(metaclass=SingletonMeta):
    def __init__(self):
        """
        Initializes a LoggingMetricsManager instance.

        This method is called when an instance of the class is requested. If an instance does not exist, one is created
        and stored in the `_instances` dictionary. Subsequent calls to `__init__` will return the existing instance.

        This method sets up the logging configuration for each module in the list `modules`, including creating a
        rotating file handler and setting the log level to INFO. Additionally, a MetricsLogger is created for each
        module, which is used to record metrics for each module.

        The root logger is also configured to log to the console and to a history log file.

        :return: None
        """
        if not hasattr(self, 'initialized'):  # This check prevents reinitialization
            self.log_directory = log_directory
            self.registry = CollectorRegistry()
            self.metrics_loggers = self.setup_logging()
            self.initialized = True

    def setup_logging(self):
        """
        Sets up logging for each module in the list `modules`.

        This involves creating a rotating file handler for each module, setting the log level to INFO, and adding the
        handler to the logger. Additionally, a MetricsLogger is created for each module, which is used to record metrics
        for each module.

        The root logger is also configured to log to the console and to a history log file.

        :return: A dictionary of loggers, keyed by module name
        :rtype: dict[str, logging.Logger]
        """
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
                                            pushgateway_enabled=PUSH_GATEWAY_ENABLED)

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

# mlflow utils
import mlflow
import mlflow.sklearn
import joblib

def start_mlflow_session(experiment_name, tracking_uri):
    """
    Initialize the MLflow tracking environment by setting the tracking URI and selecting the experiment.

    Args:
        experiment_name (str): The name of the experiment under which to log runs.
        tracking_uri (str): The URI of the MLflow tracking server.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model(model, encoder, run_name, model_path, encoder_path, metrics):
    """
    Log the ML model, its encoder, and performance metrics to MLflow and register the model.

    Args:
        model: The trained model object.
        encoder: The encoder object used for preprocessing categorical variables.
        run_name (str): The name to give to the MLflow run.
        model_path (str): Path where the model is saved locally.
        encoder_path (str): Path where the encoder is saved locally.
        metrics (dict): A dictionary of performance metrics to log.
    """
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", run_name)  # Set a custom name for the MLflow run
        mlflow.log_params({"model_type": "linear_regression"})  # Log model parameters
        mlflow.log_metrics(metrics)  # Log performance metrics
        mlflow.sklearn.log_model(model, "model", registered_model_name=run_name)  # Log and register the model
        mlflow.log_artifact(encoder_path, "encoder")  # Log the encoder as an artifact
        mlflow.register_model(f"runs:/{run.info.run_id}/model", run_name)  # Register the model in MLflow

def load_model_and_encoder(run_name):
    """
    Load the machine learning model and encoder either from local disk or MLflow based on configuration.

    Args:
        run_name (str): The name of the MLflow run from which to load the model, used only if MLFLOW_ENABLED is True.

    Returns:
        tuple: A tuple containing the loaded model and encoder objects.
    """
    try:
        if MLFLOW_ENABLED:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            # Search for the latest run with the given name
            runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName='{run_name}'",
                                      order_by=["attribute.start_time DESC"], max_results=1)
            if runs.empty:
                raise Exception("No runs found for the specified tag.")

            # Extract the run ID
            run_id = runs.iloc[0]['run_id']

            # Construct the path to the model in the MLflow tracking server
            model_path = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_path)

            # Construct the path to the encoder within the MLflow artifacts
            encoder_artifact_path = f"encoder/onehot_encoder.pkl"
            encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=encoder_artifact_path)
            encoder = joblib.load(encoder_path)

            logging.info("Model and encoder loaded successfully from MLflow.")
        else:
            # Load the saved machine learning model from disk
            model = joblib.load(os.path.join(chemin_model, fichier_model))
            # Load the saved OneHotEncoder from disk
            encoder = joblib.load(os.path.join(chemin_model, 'onehot_encoder.pkl'))
            logging.info("Model and encoder loaded successfully from local storage.")
    except Exception as e:
        logging.error(f"Failed to load model and encoder from local storage: {str(e)}")
        raise

    return model, encoder