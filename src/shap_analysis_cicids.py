"""
SHAP analysis for the CICIDS2017 Random Forest model. Tests the hypothesis
that RF learned "attack = long duration, low rate" from the Hulk/GoldenEye
training attacks, and therefore systematically misses PortScan's opposite
"short duration, high rate" signature -- rather than just arguing this from
median feature values, we measure each feature's actual signed contribution
to the model's attack-probability output, per group.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier

from load_cicids import load_cicids_file, tag_cicids_rows
from preprocess_cicids import fit_cicids_pipeline, transform_cicids

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
SHAP_SAMPLE_SIZE = 2000  # rows sampled per group for SHAP computation (not the full dataset -- SHAP is expensive)

KEY_FEATURES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s"]


def build_splits(wed_df, fri_df):
    normal_df = wed_df[wed_df["split_group"] == "normal"]
    seen_df = wed_df[wed_df["split_group"] == "seen"]

    normal_val = normal_df.sample(frac=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    normal_train = normal_df.drop(normal_val.index)

    model_train_df = pd.concat([normal_train, seen_df]).sample(frac=1.0, random_state=RANDOM_STATE)

    unseen_same_df = wed_df[wed_df["split_group"] == "unseen_same_category"]
    test_df = pd.concat([unseen_same_df, fri_df], ignore_index=False)

    return model_train_df, test_df


def get_binary_labels(df):
    return (df["split_group"] != "normal").astype(int).values


def get_attack_class_shap_values(explainer, X_sample):
    """
    Handles both the classic list-of-arrays SHAP API and the newer unified
    ndarray API, returning SHAP values for the 'attack' class (class 1).
    """
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[..., 1]
    return shap_values


if __name__ == "__main__":
    wed_path, fri_path = sys.argv[1], sys.argv[2]
    wed_df = tag_cicids_rows(load_cicids_file(wed_path))
    fri_df = tag_cicids_rows(load_cicids_file(fri_path))

    model_train_df, test_df = build_splits(wed_df, fri_df)

    pipeline = fit_cicids_pipeline(model_train_df)
    X_train = transform_cicids(model_train_df, pipeline)
    y_train = get_binary_labels(model_train_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(clf)

    # Define the four groups we want to compare
    group_defs = {
        "seen_DoS (Hulk/GoldenEye, training)": model_train_df[model_train_df["split_group"] == "seen"],
        "unseen_same (slowloris/Slowhttptest)": test_df[test_df["split_group"] == "unseen_same_category"],
        "unseen_new (PortScan)": test_df[test_df["split_group"] == "unseen_new_category"],
        "normal (test)": test_df[test_df["split_group"] == "normal"],
    }

    print("Computing SHAP values per group (this takes a few minutes per group)...\n")
    summary_rows = []
    first_group_shap_for_plot = None
    first_group_X_for_plot = None

    for name, group_df in group_defs.items():
        sample = group_df.sample(n=min(SHAP_SAMPLE_SIZE, len(group_df)), random_state=RANDOM_STATE)
        X_sample = transform_cicids(sample, pipeline)

        sv = get_attack_class_shap_values(explainer, X_sample)
        mean_shap_per_feature = pd.Series(sv.mean(axis=0), index=X_sample.columns)

        print(f"--- {name} (n={len(sample)}) ---")
        print(mean_shap_per_feature[KEY_FEATURES])
        print()

        row = {"group": name}
        for feat in KEY_FEATURES:
            row[f"mean_shap_{feat}"] = mean_shap_per_feature[feat]
        summary_rows.append(row)

        if first_group_shap_for_plot is None:
            first_group_shap_for_plot = sv
            first_group_X_for_plot = X_sample

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("shap_summary_by_group.csv", index=False)
    print("Saved per-group mean SHAP values to shap_summary_by_group.csv")

    # Global feature importance bar plot, based on the first group (seen DoS)
    plt.figure()
    shap.summary_plot(first_group_shap_for_plot, first_group_X_for_plot, plot_type="bar", show=False)
    plt.title("Top Features Driving RF's Attack Predictions (seen DoS)")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=150)
    print("Saved global feature importance plot to shap_feature_importance.png")