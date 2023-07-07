import numpy as np
from pre_processing.data_pre_processing import data_pre_processor

def test_data_preprocessing(test_data):
    X,y = test_data
    X_processed, y_processed = data_pre_processor(X, y)

    expected_result_y = np.array([['Yes'],['No']])
    expected_result_X = np.array([[ 0. ,  -1.,  -1.,   0.5,  0.5,  0.5,  1.,   1.,   1.,   1.,   1.,   1.,   1. ],
                                [ 0.,  1.,   1.,   0.5,  0.5,  0.5,  1.,   1.,   1.,   1.,   1.,   1.,   1. ]])

    np.testing.assert_array_equal(X_processed, expected_result_X)
    np.testing.assert_array_equal(y_processed, expected_result_y)

def test_data_preprocessing_only_x(test_data):
    X,y = test_data
    X_processed = data_pre_processor(X)

    expected_result_X = np.array([[ 0. ,  -1.,  -1.,   0.5,  0.5,  0.5,  1.,   1.,   1.,   1.,   1.,   1.,   1. ],
                                [ 0.,  1.,   1.,   0.5,  0.5,  0.5,  1.,   1.,   1.,   1.,   1.,   1.,   1. ]])

    np.testing.assert_array_equal(X_processed, expected_result_X)

def test_prediction_output_shape(xgb_model, test_input):
    y_pred = xgb_model.predict(test_input)
    expected_shape = (2,)
    assert y_pred.shape == expected_shape

def test_prediction_output_range(xgb_model, test_input):
    y_pred = xgb_model.predict(test_input)
    assert np.all((y_pred >= 0) & (y_pred <= 1))

def test_prediction_dtype(xgb_model, test_input):
    y_pred = xgb_model.predict(test_input)
    assert y_pred.dtype == np.float32
