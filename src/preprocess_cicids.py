"""
Preprocessing for CICIDS2017. Mirrors preprocess.py's fit-on-reference,
transform-elsewhere pattern. Drops the Destination Port leakage column,
imputes the Flow Bytes/s & Flow Packets/s NaNs (median, fit on reference
data only), and scales all remaining numeric features.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from load_cicids import LEAKAGE_COLUMNS

NON_FEATURE_COLUMNS = ["Label", "split_group"]


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c not in LEAKAGE_COLUMNS]


def fit_cicids_pipeline(reference_df: pd.DataFrame):
    feature_cols = get_feature_columns(reference_df)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(reference_df[feature_cols])

    imputed = imputer.transform(reference_df[feature_cols])
    scaler = StandardScaler()
    scaler.fit(imputed)

    return {"feature_cols": feature_cols, "imputer": imputer, "scaler": scaler}


def transform_cicids(df: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    imputed = pipeline["imputer"].transform(df[pipeline["feature_cols"]])
    scaled = pipeline["scaler"].transform(imputed)
    return pd.DataFrame(scaled, columns=pipeline["feature_cols"], index=df.index)