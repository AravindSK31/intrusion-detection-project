"""
Hybrid RF + GMM detector for NSL-KDD. Each method's raw score is converted
to a percentile rank relative to normal-validation data (putting both
methods on the same [0,1] scale), then combined via max() -- flagging a
connection if EITHER method considers it unusually anomalous. Threshold is
still chosen purely from normal-validation data, same as every other method.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from load_nslkdd import load_split, get_undetectable_unseen_rows
from preprocess import fit_feature_pipeline, transform_features
from evaluation import (
    make_normal_validation_split, get_binary_labels, tag_seen_unseen,
    threshold_from_normal_scores, evaluate_at_threshold,
)

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TARGET_FPR = 0.01
N_COMPONENTS = 5


def percentile_rank(scores, reference_scores):
    """
    For each value in `scores`, what fraction of `reference_scores` does it
    exceed? Returns values in [0, 1]. Implemented via searchsorted for speed.
    """
    sorted_ref = np.sort(reference_scores)
    ranks = np.searchsorted(sorted_ref, scores, side="right")
    return ranks / len(sorted_ref)


if __name__ == "__main__":
    train_path, test_path = sys.argv[1], sys.argv[2]
    train_df = load_split(train_path)
    test_df = load_split(test_path)

    train_attack_labels = set(train_df["label"].unique())
    test_df = tag_seen_unseen(test_df, train_attack_labels)
    undetectable = get_undetectable_unseen_rows(train_df, test_df, "snmpgetattack")

    model_train_df, normal_val_df = make_normal_validation_split(train_df, VALIDATION_FRACTION, RANDOM_STATE)
    normal_only_train_df = model_train_df[model_train_df["label"] == "normal"]

    # --- Random Forest ---
    pipeline_rf = fit_feature_pipeline(model_train_df)
    X_train_rf = transform_features(model_train_df, pipeline_rf)
    X_val_rf = transform_features(normal_val_df, pipeline_rf)
    X_test_rf = transform_features(test_df, pipeline_rf)
    y_train = get_binary_labels(model_train_df)
    y_test = get_binary_labels(test_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_rf, y_train)
    rf_val_scores = clf.predict_proba(X_val_rf)[:, 1]
    rf_test_scores = clf.predict_proba(X_test_rf)[:, 1]

    # --- GMM (normal-only) ---
    pipeline_gmm = fit_feature_pipeline(normal_only_train_df)
    X_train_gmm = transform_features(normal_only_train_df, pipeline_gmm)
    X_val_gmm = transform_features(normal_val_df, pipeline_gmm)
    X_test_gmm = transform_features(test_df, pipeline_gmm)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_gmm)
    gmm_val_scores = -gmm.score_samples(X_val_gmm)
    gmm_test_scores = -gmm.score_samples(X_test_gmm)

       # --- Combine via logical OR of each method's own independently-calibrated threshold ---
    rf_threshold = threshold_from_normal_scores(rf_val_scores, TARGET_FPR)
    gmm_threshold = threshold_from_normal_scores(gmm_val_scores, TARGET_FPR)

    rf_val_flagged = rf_val_scores >= rf_threshold
    gmm_val_flagged = gmm_val_scores >= gmm_threshold
    combined_val_flagged = rf_val_flagged | gmm_val_flagged
    print(f"Validation-set combined flag rate (OR of both methods' own 1% thresholds): "
          f"{combined_val_flagged.mean():.4f}")
    print("(This is expected to exceed 1% individually, since it's an OR of two independent alarms "
          "-- reported here for transparency, same as we report actual_fpr_on_test_normal elsewhere.)\n")

    rf_test_flagged = rf_test_scores >= rf_threshold
    gmm_test_flagged = gmm_test_scores >= gmm_threshold
    combined_test_flagged = rf_test_flagged | gmm_test_flagged

    print(f"Hybrid overall test ROC-AUC (using max of raw scores as a continuous proxy): "
          f"{roc_auc_score(y_test, np.maximum(rf_test_scores, percentile_rank(gmm_test_scores, gmm_val_scores))):.4f}")
    print(f"  (for reference) RF alone:  {roc_auc_score(y_test, rf_test_scores):.4f}")
    print(f"  (for reference) GMM alone: {roc_auc_score(y_test, gmm_test_scores):.4f}")

    is_normal = test_df["label"].values == "normal"
    seen_mask = test_df["seen_flag"].values == "seen"
    unseen_mask = (test_df["seen_flag"].values == "unseen") & (~test_df.index.isin(undetectable))

    print(f"\n[Hybrid RF+GMM, OR rule] actual_fpr_on_test_normal: {combined_test_flagged[is_normal].mean():.4f}")
    print(f"  detection_rate_seen: {combined_test_flagged[seen_mask].mean():.4f}")
    print(f"  detection_rate_unseen: {combined_test_flagged[unseen_mask].mean():.4f}")