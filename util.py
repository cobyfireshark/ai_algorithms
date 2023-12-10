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

def get_loggable_json(obj, max_string_length=100, max_list_length=5):
    if isinstance(obj, dict):
        truncated_dict = {}
        for k, v in obj.items():
            if k == 'groups':  # Special handling for 'groups' within 'root'
                truncated_dict[k] = [get_loggable_json(group, max_string_length, max_list_length) for group in v[:max_list_length]]
            elif isinstance(v, list):
                truncated_dict[k] = [get_loggable_json(item, max_string_length, max_list_length) for item in v[:max_list_length]]
            elif isinstance(v, str):
                truncated_dict[k] = v[:max_string_length] + '...' if len(v) > max_string_length else v
            else:
                truncated_dict[k] = v
        return truncated_dict
    elif isinstance(obj, list):
        return [get_loggable_json(item, max_string_length, max_list_length) for item in obj[:max_list_length]]
    elif isinstance(obj, str):
        return obj[:max_string_length] + '...' if len(obj) > max_string_length else obj
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