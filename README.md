# London Fire Brigade MLOPS DataScienceTest Project

[London Fire Brigade Incident Records](https://www.kaggle.com/datasets/mexwell/london-fire-brigade-incident-records?resource=download)

# Data Processing Script Usage Guide

This guide covers the usage of the data processing script, designed to download, process, validate incident and
mobilisation data files, and optionally convert them to pickle format for optimized Python usage. You can run this
script directly from the command line or integrate it into an API for automated tasks.

## Command Line Usage

The script can be run directly from the command line to process incident and mobilisation data files. You have the
option to specify which type of data to process, to process both by default, or to convert existing data files to pickle
format.

### Syntax

The basic syntax to run the script is as follows:

```bash
python data_preprocessing.py [options]
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

## API Integration

To integrate this script into an API, you would typically wrap the `process_data` function into a web framework
