import json
import logging
import os

import yaml


# Load the YAML configuration file
def read_yaml_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


# Load .properties file, with key=value
# commented lines must start with #
def read_properties(path):
    properties = {}
    with open(path) as f:
        for line in f.readlines():
            if not line.startswith("#"):
                k, v = line.split("=")
                properties[k.strip()] = v.strip()
    return properties


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        dirname = os.path.dirname(path)
        logging.error(f"File not found in {dirname} with content:")
        for name in os.listdir(dirname):
            logging.error(f"\t{name}")
        raise
