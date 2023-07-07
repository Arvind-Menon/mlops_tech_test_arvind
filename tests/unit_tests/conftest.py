import xgboost as xgb
import pandas as pd
import pytest

from config.config import xgb_test_config
from pre_processing.data_pre_processing import data_pre_processor

model_path = xgb_test_config["model_path"]
test_data_path = xgb_test_config["test_data_path"]
test_target_column = xgb_test_config["target_column"]

@pytest.fixture
def xgb_model():
    model = xgb.Booster()
    model.load_model(model_path)
    return model

@pytest.fixture
def test_input():
    test_df = pd.read_csv(test_data_path)
    X = test_df.drop(test_target_column, axis=1)
    X_processed = data_pre_processor(X)
    Dtest = xgb.DMatrix(X_processed)
    return Dtest

@pytest.fixture
def test_data():
    test_df = pd.read_csv(test_data_path)
    X = test_df.drop(test_target_column, axis=1)
    y = test_df[test_target_column]
    return X,y