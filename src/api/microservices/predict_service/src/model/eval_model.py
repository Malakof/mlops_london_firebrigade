import warnings

from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['eval_model']
logging.info("eval_model Logger loaded")
# TESTING
# logging.error("TEST ERROR")
# logging.warning("TEST ERROR")
# logging.info("TEST INFO")
# logging.debug("TEST DEBUG")
# logging.critical("TEST CRITICAL")

# Generate a warning to test
warnings.warn("This is a eval_model TEST warning", UserWarning)
