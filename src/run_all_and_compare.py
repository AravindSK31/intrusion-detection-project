"""
Runs all four detectors with identical preprocessing/splits, builds a
comparison table, and plots overlaid ROC curves -- the centerpiece figure
for the report.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, roc_curve

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
SVM_TRAIN_SAMPLE_SIZE = 5000


def run_supervised(model_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(model_train_df)
    X_train = transform_features(model_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)
    y_train = get_binary_labels(model_train_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    return -(-clf.predict_proba(X_val)[:, 1]), -(-clf.predict_proba(X_test)[:, 1])
    # (double negation kept for symmetry/clarity with "higher = more anomalous" convention;
    #  predict_proba's attack-probability is already higher = more anomalous)


def run_isolation_forest(normal_only_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train = transform_features(normal_only_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    clf = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train)
    return -clf.score_samples(X_val), -clf.score_samples(X_test)


def run_gmm(normal_only_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train = transform_features(normal_only_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train)
    return -gmm.score_samples(X_val), -gmm.score_samples(X_test)


def run_ocsvm(normal_only_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train_full = transform_features(normal_only_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    sample = X_train_full.sample(n=min(SVM_TRAIN_SAMPLE_SIZE, len(X_train_full)), random_state=RANDOM_STATE)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(sample)
    return -ocsvm.decision_function(X_val), -ocsvm.decision_function(X_test)


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
    y_test = get_binary_labels(test_df)

    methods = {
        "Random Forest": run_supervised(model_train_df, normal_val_df, test_df),
        "Isolation Forest": run_isolation_forest(normal_only_train_df, normal_val_df, test_df),
        "GMM": run_gmm(normal_only_train_df, normal_val_df, test_df),
        "One-Class SVM": run_ocsvm(normal_only_train_df, normal_val_df, test_df),
    }

    summary_rows = []
    plt.figure(figsize=(7, 6))

    for name, (val_scores, test_scores) in methods.items():
        auc = roc_auc_score(y_test, test_scores)
        fpr, tpr, _ = roc_curve(y_test, test_scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
        results = evaluate_at_threshold(test_df, test_scores, threshold, undetectable)
        summary_rows.append({
            "method": name,
            "roc_auc": auc,
            "target_fpr": TARGET_FPR,
            **results,
        })

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Supervised vs. Unsupervised Intrusion Detectors (NSL-KDD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_comparison.png", dpi=150)
    print("Saved ROC plot to roc_comparison.png")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_summary.csv", index=False)
    print("\n=== Summary (also saved to results_summary.csv) ===")
    print(summary_df.to_string(index=False))