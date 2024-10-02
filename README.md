# London Fire Brigade MLOPS DataScienceTest Project

## Overview

This MLOps project aims to deploy a machine learning model to predict response times for the London Fire Brigade. It
utilizes the London Fire Brigade Incident Records dataset. The focus is on demonstrating a viable framework for ML model
deployment rather than just the model's performance.

### Repository Structure

- **scripts/**
    - `tests_api.sh` - Shell script for testing API endpoints.
- **src/**
    - **api/**
        - `main.py` - FastAPI application setup and route definitions.
    - **data/**
        - `data_preprocessing.py` - Functions for downloading and preprocessing data.
    - **features/**
        - `build_features.py` - Functions to build features from preprocessed data.
    - **model/**
        - `eval_model.py` - Script for evaluating model metrics.
        - `predict_model.py` - Functions for making predictions using the trained model.
        - `train_model.py` - Functions for training the model.
    - **utils/**
        - `config.py` - Configuration settings and paths.

### Main Components

#### API (`src/api/main.py`)

- Authentication, data processing, feature building, model training, and prediction endpoints.

#### Data Processing (`src/data/data_preprocessing.py`)

- Downloads, reads, filters, and processes incident and mobilisation data. Converts data to CSV or pickle formats.

#### Feature Building (`src/features/build_features.py`)

- Loads, cleans, merges, and stores features for modeling.

#### Model Operations (`src/model/`)

- `train_model.py`: Trains and saves a linear regression model.
- `predict_model.py`: Predicts attendance times using the trained model.
- `eval_model.py`: Placeholder for model evaluation.

#### Configuration and Logging (`src/utils/config.py`)

- Central configuration for paths, URLs, and logging setup.

## Datas

[London Fire Brigade Incident Records @ Kaggle](https://www.kaggle.com/datasets/mexwell/london-fire-brigade-incident-records?resource=download)

[Incidents Datasource ](
https://data.london.gov.uk/download/london-fire-brigade-incident-records/f5066d66-c7a3-415f-9629-026fbda61822/LFB%20Incident%20data%20from%202018%20onwards.csv.xlsx)

[Mobilistaion Datasource ](
https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%202021%20-%202024.xlsx)

## Scripts

### Data Processing Script Usage Guide

This guide covers the usage of the data processing script, designed to download, process, validate incident and
mobilisation data files, and optionally convert them to pickle format for optimized Python usage. You can run this
script directly from the command line or integrate it into an API for automated tasks.

#### Command Line Usage

The script can be run directly from the command line to process incident and mobilisation data files. You have the
option to specify which type of data to process, to process both by default, or to convert existing data files to pickle
format.

##### Syntax

The basic syntax to run the script is as follows:

```bash
python data_preprocessing.py [options]
```

##### Options

- `--type {incident,mobilisation}`: Specifies the type of data to download and process. You can choose either `incident`
  or `mobilisation`. If no type is specified, the script will process both types by default.
- `--convert-to-pickle`: Converts downloaded or existing CSV data files to pickle format, saving them in a specified
  directory. This option triggers the conversion of data files into pickle format instead of the default CSV processing.
  If specified without `--type`, it will convert all available CSV files.

##### Examples

1. **Process Both Data Types (Default)**

   If no specific type is provided, the script will process both incident and mobilisation data and convert them to
   pickle:

   ```bash
   python data_preprocessing.py
   ```

2. **Process Specific Data Type**

   To process only incident data:

   ```bash
   python data_preprocessing.py --type incident
   ```

   To process only mobilisation data:

   ```bash
   python data_preprocessing.py --type mobilisation
   ```

3. **Convert Data to Pickle Format**

   To convert all available data to pickle format after processing:

   ```bash
   python data_preprocessing.py --convert-to-pickle
   ```

#### API Integration

##### `/process_data` Endpoint

This endpoint processes either incident or mobilisation data based on the input parameters, with an optional conversion
to pickle format. It leverages background tasks to handle processing without delaying the response to the client.

###### Endpoint Details

- **URL**: `/process_data`
- **Method**: `GET`
- **Auth Required**: Yes (Basic HTTP Authentication)
- **Parameters**:
    - `incident`: Boolean, default is `false`. If `true`, processes incident data.
    - `mobilisation`: Boolean, default is `false`. If `true`, processes mobilisation data.
    - `convert_to_pickle`: Boolean, default is `false`. If `true`, converts processed data to pickle format.

###### Usage

The endpoint can be called with HTTP GET requests, providing the necessary parameters for data processing.
Authentication is required to access this endpoint, ensuring that only authorized users can initiate data processing
tasks.

###### Examples

1. **Process Incident Data Only**

   ```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/process_data?incident=true' \
     -u 'admin:fireforce' \
     -H 'accept: application/json'
   ```

2. **Convert Processed Data to Pickle Format**

   ```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/process_data?incident=true&mobilisation=true&convert_to_pickle=true' \
     -u 'admin:fireforce' \
     -H 'accept: application/json'
   ```

This integration allows for seamless operation between script-based data handling and API-driven interactions,
supporting a more automated and flexible workflow.

### Features Building Script Usage Guide

This guide details the usage of the `build_features.py` script, designed to load data, clean it, merge different data
sources, and finally save the resulting dataset for modeling. This script is a crucial step in the data preparation
phase of machine learning workflows.

#### Command Line Usage

The script is typically run from the command line and does not require any command-line arguments, simplifying its
execution.

##### Syntax

The basic syntax to run the script is as follows:

```bash
python build_features.py
```

This command will execute the feature building process using predefined settings specified in the script.

#### API Integration

##### `/build_features` Endpoint

This endpoint triggers the feature building process which involves data cleaning, transformation, and merging to prepare
it for model training. It is designed to be used after data has been processed and is ready to be transformed into a
format suitable for machine learning.

###### Endpoint Details

- **URL**: `/build_features`
- **Method**: `GET`
- **Auth Required**: Yes (Basic HTTP Authentication)

###### Usage

To initiate the feature building process through the API, an authenticated GET request is made to the endpoint. This
method allows the process to be integrated into larger workflows, such as continuous integration pipelines or automated
data handling systems.

###### Examples

1. **Trigger Feature Building**

   ```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/build_features' \
     -u 'admin:fireforce' \
     -H 'accept: application/json'
   ```

This endpoint provides an automated way to build features through an API call, ensuring that data preparation can be
seamlessly integrated into broader MLOps practices.

##### Error Handling

Errors during the feature building process are logged and raised as exceptions, ensuring that any issues are documented
and can be addressed promptly. This robust error handling is crucial for maintaining data integrity and reliability in
automated systems.

### Model Training Script Usage Guide

This guide outlines the usage of the `train_model.py` script, designed for training a machine learning model. This
script handles the training of a linear regression model using the prepared features, evaluates its performance, and
saves the model along with its encoder for future predictions.

#### Command Line Usage

The script can be run from the command line, allowing you to specify paths to the dataset, model, and encoder. This
flexibility makes it suitable for different environments and dataset configurations.

##### Syntax

The basic syntax to run the script is as follows:

```bash
python train_model.py [options]
```

##### Options

- `--data_path {path}`: Specifies the path to the dataset CSV file. This is where the script will read the data to be
  used for training.
- `--model_path {path}`: Specifies the path where the trained model should be saved.
- `--encoder_path {path}`: Specifies the path where the encoder used for preprocessing categorical variables should be
  saved.

##### Examples

1. **Train Model with Custom Paths**

   To specify custom paths for the dataset, model, and encoder:

   ```bash
   python train_model.py --data_path '/path/to/data.csv' --model_path '/path/to/model.pkl' --encoder_path '/path/to/encoder.pkl'
   ```

#### API Integration

##### `/train_model` Endpoint

This endpoint handles the training of the model directly through an API call, allowing the parameters for the model
training to be specified through a POST request. It integrates seamlessly into a continuous deployment pipeline or any
automated machine learning workflow.

###### Endpoint Details

- **URL**: `/train_model`
- **Method**: `POST`
- **Auth Required**: Yes (Basic HTTP Authentication)
- **Request Body**:
    - `data_path`: Path to the dataset CSV file.
    - `model_path`: Path to save the trained model.
    - `encoder_path`: Path to save the encoder.

###### Usage

This endpoint is designed for users who wish to train the model directly via the API, providing flexibility in
specifying paths dynamically.

###### Examples

1. **Train Model Via API**

   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/train_model' \
     -u 'admin:fireforce' \
     -H 'Content-Type: application/json' \
     -d '{
       "data_path": "/path/to/data.csv",
       "model_path": "/path/to/model.pkl",
       "encoder_path": "/path/to/encoder.pkl"
     }'
   ```

This API endpoint facilitates the on-demand training of models, making it an integral part of MLOps strategies that
prioritize automation and flexibility.

##### Error Handling

The script and API endpoint are designed with robust error handling to ensure that any issues during the model training
process are logged and addressed, providing detailed error messages to aid in troubleshooting.

### Model Prediction Script Usage Guide

This guide explains the usage of the `predict_model.py` script, designed to make predictions using a pre-trained model.
This script loads the necessary model and encoder, prepares the input features, and performs predictions based on input
parameters.

#### Command Line Usage

The script allows command-line interactions where you can specify input parameters directly, ideal for testing or
one-off predictions.

##### Syntax

The basic syntax to run the script from the command line is as follows:

```bash
python predict_model.py [options]
```

##### Options

- `--distance {float}`: Specifies the distance to the incident in kilometers. This is a required input for making
  predictions.
- `--station {string}`: Specifies the name of the fire station. This is another required input for the prediction.

##### Examples

1. **Predict Attendance Time**

   To make a prediction using specific input parameters:

   ```bash
   python predict_model.py --distance 5.2 --station 'Acton'
   ```

#### API Integration

##### `/predict` Endpoint

This endpoint facilitates predictions using the trained model, directly through an API call. It is designed for dynamic
interaction, allowing users to specify the input parameters via a GET request, which are then used to return a
prediction.

###### Endpoint Details

- **URL**: `/predict`
- **Method**: `GET`
- **Auth Required**: Yes (Basic HTTP Authentication)
- **Parameters**:
    - `distance`: A float that represents the distance to the incident in kilometers.
    - `station`: A string that represents the name of the departing fire station.

###### Usage

This endpoint allows external systems to make predictions by submitting a simple GET request with the necessary
parameters.

###### Examples

1. **Make a Prediction via API**

   ```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/predict?distance=5.2&station=Acton' \
     -u 'admin:fireforce' \
     -H 'accept: application/json'
   ```

This allows for seamless integration into an operational environment, enabling real-time prediction capabilities for
systems interacting with the MLOps infrastructure.

##### Error Handling

Both the script and API endpoint include comprehensive error handling mechanisms to ensure robust operation. Errors
during prediction due to model loading failures, data preparation issues, or during the prediction itself are logged and
reported. This helps maintain high reliability and provides clarity in operational settings.

## Logging Framework Documentation

This guide provides an overview of the logging framework implemented within the project. The logging setup is designed
to capture detailed logs across different modules of the application, ensuring that all significant events, errors, and
system information are recorded for troubleshooting and monitoring purposes.

### Overview

The project utilizes Python's built-in `logging` library to set up a robust logging system. Each major component of the
application (data processing, feature building, model training, prediction, and API) has its own dedicated logger and
log file, which helps in isolating logs by functionality and simplifying troubleshooting.

### Configuration

#### Log Files

Each component of the system writes logs to a separate file. Here are the log files used in the project:

- `data_preprocessing.log` - Logs all events related to the data preprocessing tasks.
- `build_features.log` - Captures logs concerning the feature building processes.
- `train_model.log` - Stores logs related to model training sessions.
- `predict_model.log` - Logs details during the prediction operations.
- `eval_model.log` - Used exclusively for logging the model evaluation processes.
- `api.log` - Captures all logs generated from API interactions and operations.

#### Log File Location

All log files are stored in a directory specified by the `log_directory` configuration parameter. The default path is
set to `../../logs`, relative to the main application directory.

#### Log Rotation

To prevent log files from consuming excessive disk space, a rotating file handler is configured for each log file. The
rotation criteria are based on file size, with each log file allowed to grow up to 1 MB before being rotated. Up to five
old log files are kept as backups.

#### Logging Levels

The logging level for each component is configurable. The default level is set to `DEBUG` for all components, ensuring
that all debug, info, warning, error, and critical messages are captured.

### Log Format

The log messages are formatted to include the timestamp, logger name, log level, and the message. The format used is:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

This format provides clarity and consistency across all logs, making it easier to read and understand the chronological
sequence of events and actions taken by the system.

### Parameters

Key parameters configured in the `config.py` file include:

- `LOG_MODE`: Specifies the mode for log file opening, set to 'a' for appending to ensure logs are not overwritten.
- `DEFAULT_LEVEL`: The default logging level set for all loggers unless specifically overridden.
- `CONSOLE_LEVEL`: Determines the logging level for console outputs, useful during development or debugging sessions.
- `HISTORY_LEVEL`: Controls the logging level for a separate historical log that aggregates important events.

### Custom Warning Handler

The project also includes a custom warning handler that redirects all warnings generated by the application to the
appropriate logs, ensuring that they are not missed and are recorded in the same format as other log messages.

### Usage

The logging framework is integrated throughout the application code, with loggers instantiated and used in each major
component. Developers can easily add new log messages or adjust logging levels as needed to enhance diagnostics or
handle new features.

Here's a detailed documentation for the `config.py` script in your project, focusing on the parameters it manages. This guide is formatted in Markdown to facilitate integration into your project documentation.

## Configuration Parameters Guide

This guide outlines the parameters defined in the `config.py` file used throughout the project. The `config.py` script centralizes configuration settings, providing a single point of reference for managing paths, URLs, and other system-wide settings. This approach ensures that changes to the configuration are reflected across all components of the application.

### Overview of Configuration Parameters

The `config.py` file contains various parameters used by different modules for tasks such as data handling, logging, and API configuration. Below is a detailed explanation of each parameter:

#### Data and Model Paths

- `chemin_data`: Path to the directory where raw data files are stored. Default is `../../data`.
- `chemin_data_ref`: Path to the directory for reference data or processed data. Default is `../../data/ref`.
- `chemin_model`: Path to the directory where models and encoders are stored. Default is `../../models`.

#### URL Resources

- `url_incident`: URL to download the incident data from the London Fire Brigade records.
- `url_mobilisation`: URL for downloading mobilisation data from the London Fire Brigade records.

#### File Names

- `fichier_incident`: Name of the file for incident data, typically saved as `incident_data.csv`.
- `fichier_mobilisation`: Name of the file for mobilisation data, saved as `mobilisation_data.csv`.
- `fichier_stations`: File name for station data, referred to as `stations.csv`.
- `fichier_calendrier`: Calendar file name, not specifically detailed in usage.
- `fichier_vehicle`: Vehicle data file name, not detailed in usage.
- `fichier_global`: Global data file that may be used for combined datasets or outputs.
- `fichier_model`: The default file name for storing the trained model, typically a `.pkl` file like `linear_regression_model.pkl`.

#### Logging Configuration

- `log_directory`: Directory path where log files are stored. Default is `../../logs`.

#### Data Processing and Feature Engineering

- `years`: A list of years relevant for filtering or processing data. Example: `[2022, 2023]`.
- `incident_expected_columns`: A list defining the expected columns in the incident data to validate data integrity.
- `mobilisation_expected_columns`: Similar to `incident_expected_columns`, for mobilisation data.

#### Model Training

- `BandWidth_speed_min`: Minimum threshold for speed calculation in feature engineering.
- `BandWidth_speed_max`: Maximum threshold for the same.
- `BandWidth_AttendanceTimeSeconds_min`: Minimum attendance time in seconds for filtering in feature preparation.
- `BandWidth_AttendanceTimeSeconds_max`: Maximum attendance time for the same.

#### Logging Levels and Modes

- `LOG_MODE`: Defines the file mode for log files, typically set to 'a' for appending.
- `DEFAULT_LEVEL`, `CONSOLE_LEVEL`, and `HISTORY_LEVEL`: Define the logging levels used across different outputs (file, console, and historical logs).

### Usage

These parameters are utilized across various scripts to standardize the data paths, file names, and other operational settings, making it easier to maintain and modify the system as needed. For instance, changing the `chemin_data` will automatically update the data paths in all scripts that import this configuration, facilitating easy relocations of data storage without modifying each script individually.

