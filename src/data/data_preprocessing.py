import argparse
import logging
import os

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import config as cfg
from src.utils.config import logger_data as logging

logging.info("Logger loaded")


def _download_with_progress(url, output_path):
    """
    Downloads a file from the specified URL to a local path, showing progress.

    Args:
        url (str): URL of the file to download.
        output_path (str): Local path where the file will be saved.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        logging.info(f"Downloading from {url} to {output_path}")
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if progress_bar.n != total_size_in_bytes:
            logging.warning("WARNING: Downloaded file might be incomplete.")
            return False

        logging.info(f"Download complete: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error during download from {url}. Exception: {e}")
        return False


def _read_and_filter_data(filepath, year_column, years=None):
    """
    Reads data from an Excel file, filters it by specified years, and returns it.

    Args:
        filepath (str): Path to the Excel file.
        year_column (str): Column name that contains the year for filtering.
        years (list, optional): List of years to include in the filter.

    Returns:
        DataFrame: Filtered data.
    """
    logging.info(f"Reading and filtering data from {filepath} using {year_column} and {years}")
    try:
        data = pd.read_excel(filepath, engine='openpyxl')
        if years:
            data = data[data[year_column].isin(years)]
        logging.info(f"Data read and filtered from {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error reading or filtering data from {filepath}. Exception: {e}")
        return None


def _import_and_save_data(url, filepath, year_column, years):
    """
    Downloads data from a URL, saves it to a file, filters it, and writes it to a CSV file.

    Args:
        url (str): URL to download the data from.
        filepath (str): Path to save the downloaded data.
        year_column (str): Column name to filter the data by year.
        years (list): Years to include in the filtered data.

    Returns:
        bool: True if the data was successfully imported and saved, False otherwise.
    """
    if _download_with_progress(url, filepath):
        data = _read_and_filter_data(filepath, year_column, years)
        if data is not None:
            csv_path = filepath.replace('.xlsx', '.csv')
            data.to_csv(csv_path, index=False)
            logging.info(f"Data saved to CSV at {csv_path}")
            return True
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
    if not os.path.exists(file_path):
        logging.error(f"{file_path} does not exist.")
        return False
    if os.path.getsize(file_path) == 0:
        logging.error(f"{file_path} is empty.")
        return False

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Could not read {file_path}. Exception: {e}")
        return False

    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        logging.error(f"{file_path} is missing columns: {', '.join(missing_columns)}")
        return False

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

    config = config_map[data_type]

    try:
        # Ensure both data importation and file validation are successful
        if not _import_and_save_data(config['url'], config['filepath'], config['year_column'], cfg.years):
            raise ValueError(f"Failed to import and save data for {data_type}.")
        if not _validate_file(config['filepath'].replace('.xlsx', '.csv'), config['expected_columns']):
            raise ValueError(f"Validation failed for {data_type} data.")

        logging.info(f"Successfully processed {data_type} data.")
        return True

    except Exception as e:
        logging.error(f"Failed to process {data_type} data: {e}")
        raise Exception(f"Failed to process {data_type} data: {e}")


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
    Main function to handle command-line arguments and direct the data processing tasks.
    This includes downloading, processing, validating, and optionally converting data files to pickle format.
    """
    parser = argparse.ArgumentParser(
        description="Download, process, validate, and optionally convert data files to pickle format.")
    parser.add_argument('--type', choices=['incident', 'mobilisation'],
                        help="Specify the type of data to download and process individually.")
    parser.add_argument('--convert-to-pickle', action='store_true',
                        help="Convert data to pickle format instead of processing from URLs.")

    args = parser.parse_args()

    if args.convert_to_pickle:
        # Define file paths based on the configuration
        file_paths = [
            os.path.join(cfg.chemin_data, cfg.fichier_incident),
            os.path.join(cfg.chemin_data, cfg.fichier_mobilisation)
        ]
        output_dir = cfg.chemin_data_ref  # Directory to save pickle files
        errors = convert_to_pickle(file_paths, output_dir)
        if errors:
            logging.error("Some files could not be converted to pickle:")
            for file, error in errors.items():
                logging.error(f"{file}: {error}")
        else:
            logging.info("All files successfully converted to pickle.")
    elif args.type:
        success = process_data(args.type)
        logging.info(f"{'Success' if success else 'Failed'} processing {args.type} data.")
    else:
        # Default to processing all data types if no specific type or conversion is provided
        success_incident = process_data('incident')
        success_mobilisation = process_data('mobilisation')
        if success_incident and success_mobilisation:
            logging.info("Successfully processed all data.")
        else:
            logging.error("Failed to process some or all data types.")


if __name__ == "__main__":
    main()