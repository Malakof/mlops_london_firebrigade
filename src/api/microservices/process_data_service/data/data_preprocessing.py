import argparse
import os
import time
import urllib.parse
import warnings

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import config as cfg
from src.utils.config import LoggingMetricsManager

# Get the logger for model training
logging = LoggingMetricsManager().metrics_loggers['data_preprocessing']
logging.info("data_preprocessing Logger loaded")

# TESTING
# logging.error("TEST ERROR")
# logging.warning("TEST ERROR")
# logging.info("TEST INFO")
# logging.debug("TEST DEBUG")
# logging.critical("TEST CRITICAL")

# Generate a warning to test
# warnings.warn("This is a data_preprocessing TEST warning", UserWarning)

SUCCESS_PROCESSING_INCIDENT_DATA_METRIC = "success_incident"
SUCCESS_PROCESSING_MOBILISATION_DATA_METRIC = "success_mobilisation"
# Each constant below is used to create 2 identical metrics for incident and mobilisation data (x2)
DOWNLOAD_SIZE_METRIC = "download_size"
DOWNLOADED_BYTES_METRIC = "downloaded"
EXPECTED_BYTES_METRIC = "expected"
INITIAL_COUNT = "initial_count"
FILTERED_COUNT = "filtered_count"
RECORDS_SAVED = "record_saved"
DOWNLOAD_DURATION_METRIC = "download_duration"
FILTER_DURATION_METRIC = "filter_duration"


def _download_with_progress(url, output_path):
    """
    Downloads a file from the specified URL and saves it to the output path.

    Logs metrics for the download size, duration, and bytes downloaded.

    Args:
        url (str): URL of the file to download.
        output_path (str): Path to save the downloaded file to.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    start_time = time.time()  # Start timing the download process
    duration = 0
    # Derive filename without extension for logging purposes
    filename_without_extension = '_'.join(
        urllib.parse.unquote(url.split('/')[-1].split('.')[0]).replace(' ', '_').split('_')[:2])
    try:
        # Send a GET request to the URL to download the file
        response = requests.get(url, stream=True)
        # Get the total size of the file to download
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # Define block size for downloading in chunks

        # Log the start of the download with the expected file size
        logging.info(f"Starting download from {url}",
                     metrics={f"{DOWNLOAD_SIZE_METRIC}_{filename_without_extension}": total_size_in_bytes})
        # Initialize progress bar for visual feedback
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        # Open the output file path in write-binary mode
        with open(output_path, 'wb') as file:
            # Stream the content in chunks
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))  # Update progress bar with the chunk size
                file.write(data)  # Write the chunk to the file
        progress_bar.close()  # Close the progress bar after download completes
        duration = time.time() - start_time  # Calculate the total duration of the download

        # Log successful download completion with duration
        logging.info("Download completed successfully",
                     metrics={f"{DOWNLOAD_DURATION_METRIC}_{filename_without_extension}": duration})

        # Check if the downloaded size matches the expected size
        if progress_bar.n != total_size_in_bytes:
            # Log a warning if the download might be incomplete
            logging.warning("Download might be incomplete",
                            metrics={f"{DOWNLOADED_BYTES_METRIC}_{filename_without_extension}": progress_bar.n,
                                     f"{EXPECTED_BYTES_METRIC}_{filename_without_extension}": total_size_in_bytes})
            return False

        # Log the number of bytes successfully downloaded
        logging.info("Download completed successfully",
                     metrics={f"{DOWNLOADED_BYTES_METRIC}_{filename_without_extension}": progress_bar.n})
        return True
    except Exception as e:
        # Log any errors that occur during the download process
        logging.error(f"Error during download from {url}. Exception: {e}",
                      metrics={f"{DOWNLOAD_DURATION_METRIC}_{filename_without_extension}": duration})
        return False


def _read_and_filter_data(filepath, year_column, years=None):
    """
    Reads and filters data from a specified Excel file based on a given year column and optional year values.

    Args:
        filepath (str): Path to the Excel file to read and filter.
        year_column (str): Name of the column to filter by.
        years (list, optional): List of years to include in the filtered data. Defaults to None.

    Returns:
        DataFrame: The filtered data, or None if an error occurs.
    """
    logging.info(f"Reading data from {filepath}")
    start_time = time.time()  # Start timing
    duration = 0
    filename_without_extension = '_'.join(
        os.path.basename(filepath).split('.')[0].replace(' ', '_').split('_')[:2])
    try:
        data = pd.read_excel(filepath, engine='openpyxl')
        initial_count = len(data)
        if years:
            # Filter the data by the specified years
            data = data[data[year_column].isin(years)]
        duration = time.time() - start_time  # Calculate duration
        logging.info("Data filtering completed",
                     metrics={f"{INITIAL_COUNT}_{filename_without_extension}": initial_count,
                              f"{FILTERED_COUNT}_{filename_without_extension}": len(data),
                              f"{FILTER_DURATION_METRIC}_{filename_without_extension}": duration})
        return data
    except Exception as e:
        # Log any errors that occur during data filtering
        logging.error(f"Error reading or filtering data from {filepath}. Exception: {e}",
                      metrics={f"{FILTER_DURATION_METRIC}_{filename_without_extension}": duration})
        return None


def _import_and_save_data(url, filepath, year_column, years):
    """
    Downloads data from a specified URL to a local file path, reads and filters the data by year, and saves it to a CSV file.

    Args:
        url (str): URL of the Excel file to download and process.
        filepath (str): Path to save the downloaded Excel file to.
        year_column (str): Name of the column to filter by.
        years (list, optional): List of years to include in the filtered data. Defaults to None.

    Returns:
        bool: True if the data was successfully imported and saved, False otherwise.
    """
    # Get the filename without extension from the URL
    filename_without_extension = '_'.join(
        urllib.parse.unquote(url.split('/')[-1].split('.')[0]).replace(' ', '_').split('_')[:2])
    # Download the data from the URL with a progress bar
    if _download_with_progress(url, filepath):
        # Read and filter the downloaded data
        data = _read_and_filter_data(filepath, year_column, years)
        if data is not None:
            # Save the filtered data to a CSV file
            csv_path = filepath.replace('.xlsx', '.csv')
            data.to_csv(csv_path, index=False)
            logging.info("Data saved to CSV", metrics={f"{RECORDS_SAVED}_{filename_without_extension}": len(data)})
            # Return True if the data was saved successfully
            return True
    # Log an error if the data was not saved successfully
    logging.error(f"Failed to import and save data from {url}")
    return False


def _validate_file(file_path, expected_columns):
    """
    Checks if a CSV file exists, is not empty, and contains expected columns.

    Args:
        file_path (str): Path to the CSV file to validate.
        expected_columns (list): List of column names expected in the file.

    Returns:
        bool: True if the file passes validation, False otherwise.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"{file_path} does not exist.")
        return False
    # Check if the file is not empty
    if os.path.getsize(file_path) == 0:
        logging.error(f"{file_path} is empty.")
        return False

    try:
        # Attempt to read the file into a pandas DataFrame
        df = pd.read_csv(file_path)
    except Exception as e:
        # Log an error if there was a problem reading the file
        logging.error(f"Could not read {file_path}. Exception: {e}")
        return False

    # Get the columns that are expected in the file but are not present
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        # Log an error if there are missing columns
        logging.error(f"{file_path} is missing columns: {', '.join(missing_columns)}")
        return False

    # Log a success message if all validation steps passed
    logging.info(f"{file_path} is valid.")
    return True


def process_data(data_type):
    """
    Processes data based on the specified type (incident or mobilisation).
    Args:
        data_type (str): Type of the data to process ('incident' or 'mobilisation').

    Returns:
        bool: True if the process was successful.

    Raises:
        Exception: An exception with a detailed error message if the process fails.
    """
    # Mapping of data types to their respective configuration dictionaries
    config_map = {
        'incident': {
            'url': cfg.url_incident,
            'filepath': os.path.join(cfg.chemin_data, cfg.fichier_incident),
            'year_column': cfg.CalYear_incident,
            'expected_columns': cfg.incident_expected_columns
        },
        'mobilisation': {
            'url': cfg.url_mobilisation,
            'filepath': os.path.join(cfg.chemin_data, cfg.fichier_mobilisation),
            'year_column': cfg.CalYear_mobilisation,
            'expected_columns': cfg.mobilisation_expected_columns
        }
    }

    # Get the configuration for the specified data type
    config = config_map[data_type]

    # Process the data by downloading it and saving it to a CSV file
    # and then validating the file
    success = _import_and_save_data(config['url'], config['filepath'], config['year_column'], cfg.years) and \
              _validate_file(config['filepath'].replace('.xlsx', '.csv'), config['expected_columns'])

    # Log a message to indicate that the data has been processed
    logging.info(f"Processed {data_type} data")
    return success


def convert_to_pickle(file_paths, output_dir):
    """
    Converts specified CSV files to pickle format and saves them in a designated directory.
    Args:
        file_paths (list): List of file paths to convert.
        output_dir (str): Directory to save the converted pickle files.

    Returns:
        dict: A dictionary with file paths and corresponding error messages if errors occurred.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    errors = {}
    for file_path in file_paths:
        try:
            data = pd.read_csv(file_path)
            output_file = os.path.join(str(output_dir), str(os.path.basename(file_path).replace('.csv', '.pkl')))
            data.to_pickle(output_file)
            logging.info(f"Converted {file_path} to pickle at {output_file}")
        except Exception as e:
            errors[file_path] = str(e)
            logging.error(f"Error converting {file_path} to pickle. Exception: {e}")
    if errors:
        logging.error("Some files could not be converted to pickle:")
        for file, error in errors.items():
            logging.error(f"{file}: {error}")
        return errors
    logging.info("All files successfully converted to pickle.")
    return None


def main():
    """
    Downloads, processes, validates, and optionally converts data files to pickle format.

    Parses command line arguments for the type of data to process and logs the results of the
    processing.

    If a specific data type is provided, processes it and logs result. Otherwise, defaults to
    processing all data types and logs results.
    """
    parser = argparse.ArgumentParser(
        description="Download, process, validate, and optionally convert data files to pickle format.")
    parser.add_argument('--type', choices=['incident', 'mobilisation'],
                        help="Specify the type of data to download and process individually.")

    args = parser.parse_args()

    if args.type:
        # If a specific data type is provided, process it and log result
        success = process_data(args.type)
        logging.info(f"{'Success' if success else 'Failed'} processing {args.type} data.",
                     metrics={f"success_{args.type}_data": success})
    else:
        # Default to processing all data types if no specific type or conversion is provided
        # and log results
        success_incident = process_data('incident')
        success_mobilisation = process_data('mobilisation')
        if success_incident and success_mobilisation:
            logging.info("Successfully processed incident data.",
                         metrics={SUCCESS_PROCESSING_INCIDENT_DATA_METRIC: True})
            logging.info("Successfully processed mobilisation data.",
                         metrics={SUCCESS_PROCESSING_MOBILISATION_DATA_METRIC: True})
        elif success_incident:
            logging.info("Successfully processed incident data.",
                         metrics={SUCCESS_PROCESSING_INCIDENT_DATA_METRIC: True})
            logging.error("Failed to process mobilisation data.",
                          metrics={SUCCESS_PROCESSING_MOBILISATION_DATA_METRIC: False})
        elif success_mobilisation:
            logging.info("Successfully processed mobilisation data.",
                         metrics={SUCCESS_PROCESSING_MOBILISATION_DATA_METRIC: True})
            logging.error("Failed to process incident data.", metrics={SUCCESS_PROCESSING_INCIDENT_DATA_METRIC: False})
        else:
            logging.error("Failed to process incident data.", metrics={SUCCESS_PROCESSING_INCIDENT_DATA_METRIC: False})
            logging.error("Failed to process mobilisation data.",
                          metrics={SUCCESS_PROCESSING_MOBILISATION_DATA_METRIC: False})


if __name__ == "__main__":
    main()
