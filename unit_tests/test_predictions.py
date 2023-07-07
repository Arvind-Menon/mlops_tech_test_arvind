import numpy as np
import pytest

def test_prediction_output_shape():
    model = xgb.Booster(model_file='model_path')
    X_test = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # Sample test input
    y_pred = predict(model, X_test)
    expected_shape = (2,)  # Assuming 2 samples in the test input
    assert y_pred.shape == expected_shape

def test_prediction_output_range():
    model = xgb.Booster(model_file='model_path')
    X_test = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # Sample test input
    y_pred = predict(model, X_test)
    assert np.all((y_pred >= 0) & (y_pred <= 1))

def test_prediction_dtype():
    model = xgb.Booster(model_file='model_path')
    X_test = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # Sample test input
    y_pred = predict(model, X_test)
    assert y_pred.dtype == np.float32
