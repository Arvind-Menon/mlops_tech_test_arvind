xgb_classification_config = {
        "target_mapping": {'Yes': 1, 'No': 0},
        "model_create_dataset_url" : "https://storage.googleapis.com/cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
        "model_predict_dataset_url" : "https://storage.googleapis.com/cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
        "target_column": "Adopted",
        "train_size": 0.6,
        "val_test_split": 0.5,
        "xgboost_max_depth": 6,
        "xgboost_num_boost_round": 1000,
        "xgboost_early_stopping_rounds": 3,
        "model_path": "../artifacts/model/XGBoost_model.bin",
        "output_file_path": "../output/results.csv"
        }
