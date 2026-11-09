"""
Gaussian Mixture Model: trained ONLY on normal traffic. Anomaly score is the
negative log-likelihood under the fitted mixture (higher = more anomalous,
i.e. less likely to be normal), matching the convention evaluation.py expects.
"""

import sys
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
N_COMPONENTS = 5  # number of Gaussian components -- a design choice to justify in your report


if __name__ == "__main__":
    train_path, test_path = sys.argv[1], sys.argv[2]
    train_df = load_split(train_path)
    test_df = load_split(test_path)

    train_attack_labels = set(train_df["label"].unique())
    test_df = tag_seen_unseen(test_df, train_attack_labels)

    undetectable = get_undetectable_unseen_rows(train_df, test_df, "snmpgetattack")

    model_train_df, normal_val_df = make_normal_validation_split(
        train_df, VALIDATION_FRACTION, RANDOM_STATE
    )
    normal_only_train_df = model_train_df[model_train_df["label"] == "normal"]

    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train_normal = transform_features(normal_only_train_df, pipeline)
    X_val_normal = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    y_test = get_binary_labels(test_df)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_normal)

    # score_samples returns log-likelihood: higher = more normal.
    # Negate so higher = more anomalous, consistent with other methods.
    val_normal_scores = -gmm.score_samples(X_val_normal)
    test_scores = -gmm.score_samples(X_test)

    print(f"Overall test ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")

    threshold = threshold_from_normal_scores(val_normal_scores, TARGET_FPR)
    results = evaluate_at_threshold(test_df, test_scores, threshold, undetectable)

    print(f"\n[GMM] Target FPR: {TARGET_FPR:.0%}  (threshold = {threshold:.4f})")
    for k, v in results.items():
        print(f"  {k}: {v}")