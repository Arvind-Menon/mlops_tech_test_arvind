import os
import sys
import logging
import pandas as pd
import xgboost as xgb

parent_dir = os.getcwd()
sys.path.append(parent_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from pre_processing.data_pre_processing import data_pre_processor
from config.config import xgb_classification_config, log_config
from service.logger import Logger

# initialise vars from config
mapping = xgb_classification_config["target_mapping"]
target_column = xgb_classification_config["target_column"]
train_size = xgb_classification_config["train_size"]
val_test_split = xgb_classification_config["val_test_split"]
xgboost_max_depth = xgb_classification_config["xgboost_max_depth"]
xgboost_num_boost_round = xgb_classification_config["xgboost_num_boost_round"]
xgboost_early_stopping_rounds = xgb_classification_config["xgboost_early_stopping_rounds"]
model_path = xgb_classification_config["model_path"]
dataset_url = xgb_classification_config["model_create_dataset_url"]
log_file_path = log_config["log_file_path"]

# configure logger
prediction_logger = Logger(log_file=log_file_path, event_name="task_1")

# load dataset
petfinder_df = pd.read_csv(dataset_url)
prediction_logger.info(f"Dataset loaded from: {dataset_url}")

# Apply the mapping to transform the column
petfinder_df[target_column] = petfinder_df[target_column].map(mapping)

# Separate feature columns and target column
X = petfinder_df.drop(target_column, axis=1)
y = petfinder_df[target_column]

# Pre-process features and targets
X_processed, y_processed = data_pre_processor(X, y)
prediction_logger.info(f"Data pre-processed")

# Split the data into train, val, and test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size = (1-train_size), stratify=y_processed, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size = val_test_split, stratify=y_test, random_state=42
)
prediction_logger.info(f"Train split: {train_size}, Val split: {(1-train_size)*val_test_split}, Test split: {(1-train_size)*(1-val_test_split)}")

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the hyperparameters for XGBoost
params = {
    'max_depth': xgboost_max_depth
}
prediction_logger.info(f"Training begins with Max depth: {xgboost_max_depth}, Max boost round: {xgboost_num_boost_round}, early_stopping_round: {xgboost_early_stopping_rounds}")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=xgboost_num_boost_round,
    early_stopping_rounds=xgboost_early_stopping_rounds,
    evals=[(dval, 'validation')],
    verbose_eval=False
)
prediction_logger.info(f"Training over")

# predict the target column
y_pred = model.predict(dtest)
y_pred_labels = [round(value) for value in y_pred]

# performance metrics
accuracy = accuracy_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)

# Calculate accuracy on the test set
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("F1 Score: %.2f" % f1)
print("Recall: %.2f" % recall)
prediction_logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
prediction_logger.info("F1 Score: %.2f" % f1)
prediction_logger.info("Recall: %.2f" % recall)

# Save the model
model.save_model(model_path)
logging.info(f"Model saved at {model_path}\n")
print(f"Model saved at {model_path}")