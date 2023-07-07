import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class FrequencyEncoder:
    def __init__(self):
        self.mapping = {}

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col in X.columns:
            frequencies = X[col].value_counts(normalize=True)
            self.mapping[col] = frequencies.to_dict()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = X_copy[col].map(self.mapping[col])
        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def data_pre_processor(X, y=None):

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("frequency-encode", FrequencyEncoder()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")),
               ("scale", StandardScaler())]
    )

    cat_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns

    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )

    # Apply preprocessing
    X_processed = full_processor.fit_transform(X)
    if y is not None:
        y_processed = SimpleImputer(strategy="most_frequent").fit_transform(y.values.reshape(-1, 1))
        return X_processed,y_processed
    else:
        return X_processed
