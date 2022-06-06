import argparse

import numpy as np
import pandas as pd
import yaml

from urllib.request import urlopen
from zipfile import ZipFile
from scipy.io import arff
import pandas as pd
from io import BytesIO, TextIOWrapper


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_data(model_var, data_url, filename, target):
    """
    load data set from URL, unzip data and read only filename
    input: model_var, data_url, filename, target
    output:pandas dataframe
    """
    resp = urlopen(data_url)
    zipfile = ZipFile(BytesIO(resp.read()))
    in_mem = TextIOWrapper(zipfile.open(filename), encoding="ascii")
    data = arff.loadarff(in_mem)
    df = pd.DataFrame(data[0])
    df["target"] = df[target]
    df = df[model_var]
    return df


def load_raw_data(config_path):
    """
    load data from external location(data/external) to the raw folder(data/raw) with train and testing dataset
    input: config_path
    output: save train file in data/raw folder
    """
    config = read_params(config_path)
    data_url = config["raw_data_config"]["url_data"]
    filename = config["raw_data_config"]["filename_data"]
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    model_var = config["raw_data_config"]["model_var"]
    target = config["raw_data_config"]["target"]

    df = load_data(model_var, data_url, filename, target)
    df.to_csv(raw_data_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)
