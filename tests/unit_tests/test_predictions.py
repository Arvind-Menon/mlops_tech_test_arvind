import pandas as pd
import numpy as np
import os
import xgboost as xgb
import sys
os.chdir('../')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from config.config import xgb_test_config
from pre_processing.data_pre_processing import data_pre_processor


model_path = xgb_test_config["model_path"]
test_data_path = xgb_test_config["test_data_path"]

X = pd.read_csv(test_data_path)
X_processed = data_pre_processor(X)
Dtest = xgb.DMatrix(X_processed)

def test_prediction_output_shape():
    model = xgb.Booster()
    model.load_model(model_path)
    y_pred = model.predict(Dtest)
    expected_shape = (2,)
    assert y_pred.shape == expected_shape

def test_prediction_output_range():
    model = xgb.Booster()
    model.load_model(model_path)
    y_pred = model.predict(Dtest)
    assert np.all((y_pred >= 0) & (y_pred <= 1))

def test_prediction_dtype():
    model = xgb.Booster()
    model.load_model(model_path)
    y_pred = model.predict(Dtest)
    assert y_pred.dtype == np.float32
