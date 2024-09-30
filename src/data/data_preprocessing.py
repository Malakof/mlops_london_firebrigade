import os
import argparse
import pandas as pd
from tqdm import tqdm
from src.utils import config as cfg
import requests


def download_with_progress(url, output_path):
    """Downloads file from a given URL with progress indication."""
    try:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        print(f"Downloading from {url} to {output_path}")
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if progress_bar.n != total_size_in_bytes:
            print("WARNING: Downloaded file might be incomplete.")
            return False
        print(f"Download complete: {output_path}")
        return True
    except Exception as e:
        print(f"Error during download from {url}. Exception: {e}")
        return False


def read_and_filter_data(filepath, year_column, years=None):
    """Reads and filters Excel file based on specified year column and years list."""
    print(f"Reading data and filtering data from {filepath} using {year_column} and {years}")
    try:
        data = pd.read_excel(filepath, engine='openpyxl')
        if years is not None:
            data = data[data[year_column].isin(years)]
        print(f"Data read and filtered from {filepath}")
        return data
    except Exception as e:
        print(f"Error reading or filtering data from {filepath}. Exception: {e}")
        return None


def import_and_save_data(url, filepath, year_column, years):
    """Imports data from URL and saves it as CSV after filtering."""
    if download_with_progress(url, filepath):
        data = read_and_filter_data(filepath, year_column, years)
        if data is not None:
            csv_path = filepath.replace('.xlsx', '.csv')
            data.to_csv(csv_path, index=False)
            print(f"Data saved to CSV at {csv_path}")
            return True
    return False


def validate_file(file_path, expected_columns):
    """Validates the integrity of the CSV file based on expected columns."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return False
    if os.path.getsize(file_path) == 0:
        print(f"Error: {file_path} is empty.")
        return False
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: Could not read {file_path}. Exception: {e}")
        return False
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        print(f"Error: {file_path} is missing columns: {', '.join(missing_columns)}")
        return False
    print(f"{file_path} is valid.")
    return True


def process_data(data_type):
    """Processes data based on the specified type (incident or mobilisation)."""
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
    return import_and_save_data(config['url'], config['filepath'], config['year_column'], cfg.years) and validate_file(
        config['filepath'].replace('.xlsx', '.csv'), config['expected_columns'])


def convert_to_pickle(file_paths, output_dir):
    """Convert specified CSV files to pickle files and save them to a directory."""
    import pandas as pd
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    errors = {}
    for file_path in file_paths:
        try:
            data = pd.read_csv(file_path)
            output_file = os.path.join(str(output_dir), str(os.path.basename(file_path).replace('.csv', '.pkl')))
            data.to_pickle(output_file)
            print(f"Converted {file_path} to {output_file}")
        except Exception as e:
            errors[file_path] = str(e)
            print(f"Error converting {file_path} to pickle. Exception: {e}")

    return errors
def save_to_pickle(data, filename):
    """Convert data to a pickle file and save it to the specified directory."""
    import pandas as pd

    pickle_path = os.path.join(str(cfg.chemin_data_ref), filename + '.pkl')
    if not os.path.exists(cfg.chemin_data_ref):
        os.makedirs(cfg.chemin_data_ref)
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data.to_pickle(pickle_path)
        print(f"Data successfully saved to pickle at {pickle_path}")
    except Exception as e:
        print(f"Failed to save data to pickle. Error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download, process, validate, and optionally convert data files to pickle format.")
    parser.add_argument('--type', choices=['incident', 'mobilisation'], help="Specify the type of data to download and process individually.")
    parser.add_argument('--convert-to-pickle', action='store_true', help="Convert data to pickle format instead of processing from URLs.")

    args = parser.parse_args()

    if args.convert_to_pickle:
        # File paths need to be defined or calculated based on the cfg
        file_paths = [
            os.path.join(cfg.chemin_data, 'incident_data.csv'),  # Update these paths based on actual data paths
            os.path.join(cfg.chemin_data, 'mobilisation_data.csv')
        ]
        output_dir = cfg.chemin_data_ref  # Directory to save pickle files, defined in your cfg module
        errors = convert_to_pickle(file_paths, output_dir)
        if errors:
            print("Some files could not be converted to pickle:")
            for file, error in errors.items():
                print(f"{file}: {error}")
        else:
            print("All files successfully converted to pickle.")
    elif args.type:
        success = process_data(args.type)
        print(f"{'Success' if success else 'Failed'} processing {args.type} data.")
    else:
        # Default to processing all data types if no specific type or conversion is provided
        success_incident = process_data('incident')
        success_mobilisation = process_data('mobilisation')
        print(f"{'Success' if success_incident else 'Failed'} processing incident data.")
        print(f"{'Success' if success_mobilisation else 'Failed'} processing mobilisation data.")
        convert_to_pickle([os.path.join(cfg.chemin_data, cfg.fichier_incident), os.path.join(cfg.chemin_data, cfg.fichier_mobilisation)], cfg.chemin_data)
if __name__ == "__main__":
    main()
