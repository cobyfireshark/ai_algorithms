# decision_tree/util.py
import os
import logging
from csv import reader

# Load a CSV file
def load_csv(filepath, has_header=False):
    with open(filepath, "rt") as file:
        lines = reader(file)
        dataset = list(lines)
        # Skip the first row (header) if has_header is True
        if has_header:
            dataset.pop(0)
    return dataset

# Convert string column to float
def string_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def get_loggable_json(obj, max_length=100):
    if isinstance(obj, dict):
        return {k: get_loggable_json(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        limited_list = obj[:5]  # Limit the list to the first 5 items
        return [get_loggable_json(item, max_length) for item in limited_list]
    elif isinstance(obj, str) and len(obj) > max_length:
        return obj[:max_length] + '...'  # Truncate long strings
    return obj

def initialize_logging(debug, log_path):
    # Set base log level
    log_level = logging.DEBUG if debug else logging.INFO

    # Set up console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure the root logger
    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(console_handler)

    directory_log_path = os.path.dirname(log_path)
    os.makedirs(directory_log_path, exist_ok=True) 

    # Set up file logging
    file_log_level = logging.DEBUG if debug else logging.INFO
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info("Debug mode is %s", "ON" if debug else "OFF")