"""
One-Class SVM: trained ONLY on normal traffic, using an RBF kernel (Week 5
material). RBF-kernel SVMs scale poorly with sample size, so we fit on a
random subsample of normal training data rather than the full ~99k rows --
an explicit, documented compute tradeoff, not a hidden shortcut.
"""

import sys
from sklearn.svm import OneClassSVM
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
SVM_TRAIN_SAMPLE_SIZE = 5000  # subsample size for OCSVM fitting -- documented compute tradeoff


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

    # Pipeline fit on the FULL normal training set, for consistent encoding
    # with the other methods -- only the SVM's own .fit() call is subsampled.
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train_normal_full = transform_features(normal_only_train_df, pipeline)
    X_val_normal = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    y_test = get_binary_labels(test_df)

    svm_train_sample = X_train_normal_full.sample(
        n=min(SVM_TRAIN_SAMPLE_SIZE, len(X_train_normal_full)), random_state=RANDOM_STATE
    )
    print(f"Fitting One-Class SVM on a subsample of {len(svm_train_sample)} normal rows "
          f"(out of {len(X_train_normal_full)} available).")

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(svm_train_sample)
    print(f"Number of support vectors: {len(ocsvm.support_)}")

    # decision_function: higher = more normal (further inside the boundary).
    # Negate so higher = more anomalous, consistent with other methods.
    val_normal_scores = -ocsvm.decision_function(X_val_normal)
    test_scores = -ocsvm.decision_function(X_test)

    print(f"Overall test ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")

    threshold = threshold_from_normal_scores(val_normal_scores, TARGET_FPR)
    results = evaluate_at_threshold(test_df, test_scores, threshold, undetectable)

    print(f"\n[One-Class SVM] Target FPR: {TARGET_FPR:.0%}  (threshold = {threshold:.4f})")
    for k, v in results.items():
        print(f"  {k}: {v}")