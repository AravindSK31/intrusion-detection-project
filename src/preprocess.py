"""
Preprocessing for NSL-KDD. Fit the encoder/scaler on ONE reference dataframe
(the actual model-training data), then transform any other split with that
same fitted pipeline -- never re-fit on validation or test data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_nslkdd import FEATURE_COLUMNS, CATEGORICAL_COLUMNS

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def fit_feature_pipeline(reference_df: pd.DataFrame):
    """Fit one-hot categories + scaler on the reference (training) data."""
    dummy_columns = pd.get_dummies(reference_df[CATEGORICAL_COLUMNS]).columns
    scaler = StandardScaler()
    scaler.fit(reference_df[NUMERIC_COLUMNS])
    return {"dummy_columns": dummy_columns, "scaler": scaler}


def transform_features(df: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    """Apply an already-fitted pipeline to any dataframe (val or test)."""
    dummies = pd.get_dummies(df[CATEGORICAL_COLUMNS])
    dummies = dummies.reindex(columns=pipeline["dummy_columns"], fill_value=0)

    numeric_scaled = pd.DataFrame(
        pipeline["scaler"].transform(df[NUMERIC_COLUMNS]),
        columns=NUMERIC_COLUMNS,
        index=df.index,
    )
    return pd.concat([numeric_scaled, dummies], axis=1)