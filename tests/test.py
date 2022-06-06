import pandas as pd
from src.train import get_feat_and_target
from src.load_data import read_params, load_data

def test_get_feat_and_target():
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    target = 'B'
    x, y = get_feat_and_target(df, target)
    assert all([a == b for a, b in zip(x.columns, ['A','C'])])
    assert all([a == b for a, b in zip(y.columns, ['B'])])

def test_load_data():
    config = read_params('params.yaml')
    data_url = config["raw_data_config"]["url_data"]
    filename = config["raw_data_config"]["filename_data"]
    model_var = config["raw_data_config"]["model_var"]
    target = config["raw_data_config"]["target"]
    df = load_data(model_var, data_url, filename,target)
    assert len(df) == 310