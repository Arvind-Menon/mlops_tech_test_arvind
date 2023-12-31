import logging
import os
import xgboost as xgb
import pandas as pd
import sys

parent_dir = os.getcwd()
sys.path.append(parent_dir)

from pre_processing.data_pre_processing import data_pre_processor
from config.config import xgb_classification_config, log_config
from service.logger import Logger


# initialise vars from config
mapping = xgb_classification_config["target_mapping"]
target_column = xgb_classification_config["target_column"]
model_path = xgb_classification_config["model_path"]
dataset_url = xgb_classification_config["model_predict_dataset_url"]
output_file_path = xgb_classification_config["output_file_path"]
log_file_path = log_config["log_file_path"]

# configure logger
prediction_logger = Logger(log_file=log_file_path, event_name="task_2")

# Load the model
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)
prediction_logger.info("Model loaded")

# load dataset
petfinder_df = pd.read_csv(dataset_url)
prediction_logger.info("Dataset loaded")

# Create a mapping dictionary
reverse_mapping = {v: k for k, v in mapping.items()}

# Apply the mapping to transform the column
petfinder_df[target_column] = petfinder_df[target_column].map(mapping)

# separate features columns from target columns
X = petfinder_df.drop(target_column, axis=1)
y = petfinder_df.Adopted

# data pre-processing
X_processed = data_pre_processor(X)

# transform into form feed-able to the model
dtest = xgb.DMatrix(X_processed)

# predict the outcomes and round the results
predictions = loaded_model.predict(dtest)
y_pred_labels = [round(value) for value in predictions]
prediction_logger.info("Predictions done!")

# Change from 0/1 -> No/Yes
petfinder_df[target_column] = petfinder_df[target_column].map(reverse_mapping)
petfinder_df[f'{target_column}_prediction'] = y_pred_labels
petfinder_df[f'{target_column}_prediction'] = petfinder_df[f'{target_column}_prediction'].map(reverse_mapping)

# save results
petfinder_df.to_csv(output_file_path, index=False)

logging.info(f"Output has been saved in: {output_file_path}\n")
print(f"Output has been saved in: {output_file_path}")
