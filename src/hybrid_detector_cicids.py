"""
Hybrid RF + GMM detector for CICIDS2017, using the OR rule: flag if EITHER
method's own independently-calibrated threshold (1% target FPR, set from
normal-validation data only) is exceeded. This tests whether combining two
methods with complementary strengths (RF good on seen DoS, GMM good on
novel-category PortScan) captures both advantages simultaneously.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from load_cicids import load_cicids_file, tag_cicids_rows
from preprocess_cicids import fit_cicids_pipeline, transform_cicids

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TARGET_FPR = 0.01
N_COMPONENTS = 5


def build_splits(wed_df, fri_df):
    normal_df = wed_df[wed_df["split_group"] == "normal"]
    seen_df = wed_df[wed_df["split_group"] == "seen"]

    normal_val = normal_df.sample(frac=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    normal_train = normal_df.drop(normal_val.index)

    model_train_df = pd.concat([normal_train, seen_df]).sample(frac=1.0, random_state=RANDOM_STATE)
    unseen_same_df = wed_df[wed_df["split_group"] == "unseen_same_category"]
    test_df = pd.concat([unseen_same_df, fri_df], ignore_index=False)

    return model_train_df, normal_train, normal_val, test_df


def get_binary_labels(df):
    return (df["split_group"] != "normal").astype(int).values


def threshold_from_normal_scores(normal_scores, target_fpr):
    return np.percentile(normal_scores, 100 * (1 - target_fpr))


if __name__ == "__main__":
    wed_path, fri_path = sys.argv[1], sys.argv[2]
    wed_df = tag_cicids_rows(load_cicids_file(wed_path))
    fri_df = tag_cicids_rows(load_cicids_file(fri_path))

    model_train_df, normal_train, normal_val, test_df = build_splits(wed_df, fri_df)
    normal_only_train = model_train_df[model_train_df["split_group"] == "normal"]
    y_test = get_binary_labels(test_df)

    # --- Random Forest ---
    pipeline_rf = fit_cicids_pipeline(model_train_df)
    X_train_rf = transform_cicids(model_train_df, pipeline_rf)
    X_val_rf = transform_cicids(normal_val, pipeline_rf)
    X_test_rf = transform_cicids(test_df, pipeline_rf)
    y_train = get_binary_labels(model_train_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_rf, y_train)
    rf_val_scores = clf.predict_proba(X_val_rf)[:, 1]
    rf_test_scores = clf.predict_proba(X_test_rf)[:, 1]

    # --- GMM (normal-only) ---
    pipeline_gmm = fit_cicids_pipeline(normal_only_train)
    X_train_gmm = transform_cicids(normal_only_train, pipeline_gmm)
    X_val_gmm = transform_cicids(normal_val, pipeline_gmm)
    X_test_gmm = transform_cicids(test_df, pipeline_gmm)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_gmm)
    gmm_val_scores = -gmm.score_samples(X_val_gmm)
    gmm_test_scores = -gmm.score_samples(X_test_gmm)

    # --- Combine via OR of each method's own threshold ---
    rf_threshold = threshold_from_normal_scores(rf_val_scores, TARGET_FPR)
    gmm_threshold = threshold_from_normal_scores(gmm_val_scores, TARGET_FPR)

    rf_test_flagged = rf_test_scores >= rf_threshold
    gmm_test_flagged = gmm_test_scores >= gmm_threshold
    combined_test_flagged = rf_test_flagged | gmm_test_flagged

    is_normal = test_df["split_group"].values == "normal"
    same_cat_mask = test_df["split_group"].values == "unseen_same_category"
    new_cat_mask = test_df["split_group"].values == "unseen_new_category"

    print("=== Individual methods (reference) ===")
    print(f"RF:  FPR={rf_test_flagged[is_normal].mean():.4f}  "
          f"same_cat={rf_test_flagged[same_cat_mask].mean():.4f}  "
          f"new_cat={rf_test_flagged[new_cat_mask].mean():.4f}")
    print(f"GMM: FPR={gmm_test_flagged[is_normal].mean():.4f}  "
          f"same_cat={gmm_test_flagged[same_cat_mask].mean():.4f}  "
          f"new_cat={gmm_test_flagged[new_cat_mask].mean():.4f}")

    print("\n=== Hybrid (OR rule) ===")
    print(f"actual_fpr_on_test_normal: {combined_test_flagged[is_normal].mean():.4f}")
    print(f"detection_rate_unseen_same_category: {combined_test_flagged[same_cat_mask].mean():.4f}")
    print(f"detection_rate_unseen_new_category: {combined_test_flagged[new_cat_mask].mean():.4f}")
    