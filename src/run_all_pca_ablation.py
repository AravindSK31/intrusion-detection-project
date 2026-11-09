"""
PCA ablation: rerun all four detectors on PCA-reduced features and compare
against the full-feature results. PCA is fit the same way each method
already fits its own preprocessing (on its own training reference), so
this adds one step to the existing pipeline rather than changing the
overall methodology.
"""

import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
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
N_COMPONENTS = 5  # GMM component count, unchanged
SVM_TRAIN_SAMPLE_SIZE = 5000
PCA_VARIANCE_TARGET = 0.95  # keep enough components to retain 95% of variance


def fit_pca(X_train_reference):
    pca = PCA(n_components=PCA_VARIANCE_TARGET, random_state=RANDOM_STATE)
    pca.fit(X_train_reference)
    return pca


def run_supervised_pca(model_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(model_train_df)
    X_train = transform_features(model_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    pca = fit_pca(X_train)
    X_train_p, X_val_p, X_test_p = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)

    y_train = get_binary_labels(model_train_df)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_p, y_train)
    return clf.predict_proba(X_val_p)[:, 1], clf.predict_proba(X_test_p)[:, 1], pca.n_components_


def run_isolation_forest_pca(normal_only_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train = transform_features(normal_only_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    pca = fit_pca(X_train)
    X_train_p, X_val_p, X_test_p = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)

    clf = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_p)
    return -clf.score_samples(X_val_p), -clf.score_samples(X_test_p), pca.n_components_


def run_gmm_pca(normal_only_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train = transform_features(normal_only_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    pca = fit_pca(X_train)
    X_train_p, X_val_p, X_test_p = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_p)
    return -gmm.score_samples(X_val_p), -gmm.score_samples(X_test_p), pca.n_components_


def run_ocsvm_pca(normal_only_train_df, normal_val_df, test_df):
    pipeline = fit_feature_pipeline(normal_only_train_df)
    X_train_full = transform_features(normal_only_train_df, pipeline)
    X_val = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    pca = fit_pca(X_train_full)
    X_train_full_p = pca.transform(X_train_full)
    X_val_p, X_test_p = pca.transform(X_val), pca.transform(X_test)

    sample_idx = pd.Series(range(len(X_train_full_p))).sample(
        n=min(SVM_TRAIN_SAMPLE_SIZE, len(X_train_full_p)), random_state=RANDOM_STATE
    ).values
    sample = X_train_full_p[sample_idx]

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(sample)
    return -ocsvm.decision_function(X_val_p), -ocsvm.decision_function(X_test_p), pca.n_components_


if __name__ == "__main__":
    train_path, test_path = sys.argv[1], sys.argv[2]
    train_df = load_split(train_path)
    test_df = load_split(test_path)

    train_attack_labels = set(train_df["label"].unique())
    test_df = tag_seen_unseen(test_df, train_attack_labels)
    undetectable = get_undetectable_unseen_rows(train_df, test_df, "snmpgetattack")

    model_train_df, normal_val_df = make_normal_validation_split(train_df, VALIDATION_FRACTION, RANDOM_STATE)
    normal_only_train_df = model_train_df[model_train_df["label"] == "normal"]
    y_test = get_binary_labels(test_df)

    runners = {
        "Random Forest": run_supervised_pca(model_train_df, normal_val_df, test_df),
        "Isolation Forest": run_isolation_forest_pca(normal_only_train_df, normal_val_df, test_df),
        "GMM": run_gmm_pca(normal_only_train_df, normal_val_df, test_df),
        "One-Class SVM": run_ocsvm_pca(normal_only_train_df, normal_val_df, test_df),
    }

    rows = []
    for name, (val_scores, test_scores, n_comp) in runners.items():
        auc = roc_auc_score(y_test, test_scores)
        threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
        results = evaluate_at_threshold(test_df, test_scores, threshold, undetectable)
        rows.append({"method": name, "n_pca_components": n_comp, "roc_auc": auc, **results})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("results_summary_pca.csv", index=False)
    print(summary_df.to_string(index=False))