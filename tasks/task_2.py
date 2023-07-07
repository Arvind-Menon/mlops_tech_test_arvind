import logging
import os
import xgboost as xgb
import pandas as pd
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from pre_processing.data_pre_processing import data_pre_processor
from config.config import xgb_classification_config
from service.logger import Logger

prediction_logger = Logger(log_file='../output/prediction_logs.log', event_name="task_2")


# initialise vars from config
mapping = xgb_classification_config["target_mapping"]
target_column = xgb_classification_config["target_column"]
model_path = xgb_classification_config["model_path"]
dataset_url = xgb_classification_config["model_predict_dataset_url"]
output_file_path = xgb_classification_config["output_file_path"]

# Load the model
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)
prediction_logger.info("Model loaded")

petfinder_df = pd.read_csv(dataset_url)
prediction_logger.info("Dataset loaded")

# Create a mapping dictionary
reverse_mapping = {v: k for k, v in mapping.items()}

# Apply the mapping to transform the column
petfinder_df[target_column] = petfinder_df[target_column].map(mapping)

X = petfinder_df.drop(target_column, axis=1)
y = petfinder_df.Adopted

X_processed = data_pre_processor(X)

dtest = xgb.DMatrix(X_processed)

predictions = loaded_model.predict(dtest)
y_pred_labels = [round(value) for value in predictions]
prediction_logger.info("Predictions done!")

petfinder_df[target_column] = petfinder_df[target_column].map(reverse_mapping)
petfinder_df[f'{target_column}_scored'] = y_pred_labels
petfinder_df[f'{target_column}_scored'] = petfinder_df[f'{target_column}_scored'].map(reverse_mapping)

petfinder_df.to_csv(output_file_path)

logging.info(f"Output has been saved in: {output_file_path}\n")
print(f"Output has been saved in: {output_file_path}")
