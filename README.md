# London Fire Brigade MLOPS DataScienceTest Project

[London Fire Brigade Incident Records](https://www.kaggle.com/datasets/mexwell/london-fire-brigade-incident-records?resource=download)

# Data Processing Script Usage Guide

This guide covers the usage of the data processing script, designed to download, process, validate incident and
mobilisation data files, and optionally convert them to pickle format for optimized Python usage. You can run this
script directly from the command line or integrate it into an API for automated tasks.

## Requirements

Before running the script, ensure that you have Python installed on your system along with the following packages:

- `pandas`
- `requests`
- `tqdm`

These packages can be installed via pip if they are not already installed:

```bash
pip install pandas requests tqdm
```

## Command Line Usage

The script can be run directly from the command line to process incident and mobilisation data files. You have the
option to specify which type of data to process, to process both by default, or to convert existing data files to pickle
format.

### Syntax

The basic syntax to run the script is as follows:

```bash
python script_name.py [options]
```

### Options

- `--type {incident,mobilisation}`: Specifies the type of data to download and process. You can choose either `incident`
  or `mobilisation`. If no type is specified, the script will process both types by default.
- `--convert-to-pickle`: Converts downloaded or existing CSV data files to pickle format, saving them in a specified
  directory. This option triggers the conversion of data files into pickle format instead of the default CSV processing.
  If specified without `--type`, it will convert all available CSV files.

### Examples

1. **Process Both Data Types (Default)**

   If no specific type is provided, the script will process both incident and mobilisation data and convert them to
   pickle:

   ```bash
   python script_name.py
   ```

2. **Process Specific Data Type**

   To process only incident data:

   ```bash
   python script_name.py --type incident
   ```

   To process only mobilisation data:

   ```bash
   python script_name.py --type mobilisation
   ```

3. **Convert Data to Pickle Format**

   To convert all available data to pickle format after processing:

   ```bash
   python script_name.py --convert-to-pickle
   ```

## API Integration

To integrate this script into an API, you would typically wrap the `process_data` function into a web framework such as
Flask or Django. Here’s a basic example using Flask:

### Example with Flask

First, install Flask if it is not already installed:

```bash
pip install Flask
```

Create a new Python file for your Flask application, for example, `app.py`, and set up routes to handle requests:

```python
from flask import Flask, jsonify
from script_name import process_data  # make sure to import process_data function

app = Flask(__name__)


@app.route('/process/<data_type>', methods=['GET'])
def process(data_type):
    if data_type not in ['incident', 'mobilisation']:
        return jsonify({'error': 'Invalid data type'}), 400
    result, message = process_data(data_type)
    return jsonify({'success': result, 'message': message})


if __name__ == '__main__':
    app.run(debug=True)
```

### Running the Flask API

To run the Flask application, use the following command:

```bash
python app.py
```

This will start a local server. You can access the API to process data by visiting:

- `http://localhost:5000/process/incident` for incidents.
- `http://localhost:5000/process/mobilisation` for mobilisations.

This guide should help users understand how to use the script effectively from the command line and how to integrate it
into an API for automated processing.

---
