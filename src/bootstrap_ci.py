"""
Bootstrap confidence intervals for detection rates, built on top of scores
computed once per method (we do NOT retrain inside the bootstrap loop --
that would be slow and unnecessary). We resample which test rows are
evaluated, using scores already computed, which is the standard and
correct way to bootstrap a fixed model's performance metrics.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

from load_nslkdd import load_split, get_undetectable_unseen_rows
from preprocess import fit_feature_pipeline, transform_features
from evaluation import (
    make_normal_validation_split, get_binary_labels, tag_seen_unseen,
    threshold_from_normal_scores,
)

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TARGET_FPR = 0.01
N_COMPONENTS = 5
SVM_TRAIN_SAMPLE_SIZE = 5000
N_BOOTSTRAP = 1000


def compute_all_scores(model_train_df, normal_val_df, test_df):
    """Train each method once, return their scores on val and test sets."""
    normal_only_train = model_train_df[model_train_df["label"] == "normal"]
    scores = {}

    # Random Forest
    pipeline = fit_feature_pipeline(model_train_df)
    X_train, X_val, X_test = (
        transform_features(model_train_df, pipeline),
        transform_features(normal_val_df, pipeline),
        transform_features(test_df, pipeline),
    )
    y_train = get_binary_labels(model_train_df)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    scores["Random Forest"] = (clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1])

    # Unsupervised methods share a normal-only pipeline
    pipeline_u = fit_feature_pipeline(normal_only_train)
    X_train_u, X_val_u, X_test_u = (
        transform_features(normal_only_train, pipeline_u),
        transform_features(normal_val_df, pipeline_u),
        transform_features(test_df, pipeline_u),
    )

    iso = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_u)
    scores["Isolation Forest"] = (-iso.score_samples(X_val_u), -iso.score_samples(X_test_u))

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_u)
    scores["GMM"] = (-gmm.score_samples(X_val_u), -gmm.score_samples(X_test_u))

    sample = X_train_u.sample(n=min(SVM_TRAIN_SAMPLE_SIZE, len(X_train_u)), random_state=RANDOM_STATE)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(sample)
    scores["One-Class SVM"] = (-ocsvm.decision_function(X_val_u), -ocsvm.decision_function(X_test_u))

    return scores


def bootstrap_detection_rate(test_scores, threshold, group_mask, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Bootstrap the detection rate on rows where group_mask is True.
    Resamples WITH replacement from those rows only, n_bootstrap times.
    """
    rng = np.random.RandomState(seed)
    group_scores = test_scores[group_mask]
    n = len(group_scores)
    if n == 0:
        return np.nan, np.nan, np.nan

    point_estimate = (group_scores >= threshold).mean()
    boot_rates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_rates[i] = (group_scores[idx] >= threshold).mean()

    ci_low, ci_high = np.percentile(boot_rates, [2.5, 97.5])
    return point_estimate, ci_low, ci_high


if __name__ == "__main__":
    train_path, test_path = sys.argv[1], sys.argv[2]
    train_df = load_split(train_path)
    test_df = load_split(test_path)

    train_attack_labels = set(train_df["label"].unique())
    test_df = tag_seen_unseen(test_df, train_attack_labels)
    undetectable = get_undetectable_unseen_rows(train_df, test_df, "snmpgetattack")

    model_train_df, normal_val_df = make_normal_validation_split(train_df, VALIDATION_FRACTION, RANDOM_STATE)

    print("Training all methods once (this takes a minute)...")
    all_scores = compute_all_scores(model_train_df, normal_val_df, test_df)

    seen_mask = test_df["seen_flag"].values == "seen"
    unseen_mask = (test_df["seen_flag"].values == "unseen") & (~test_df.index.isin(undetectable))

    rows = []
    for method, (val_scores, test_scores) in all_scores.items():
        threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)

        seen_point, seen_lo, seen_hi = bootstrap_detection_rate(test_scores, threshold, seen_mask)
        unseen_point, unseen_lo, unseen_hi = bootstrap_detection_rate(test_scores, threshold, unseen_mask)

        rows.append({
            "method": method,
            "seen_detection_rate": seen_point,
            "seen_ci_95": f"[{seen_lo:.3f}, {seen_hi:.3f}]",
            "unseen_detection_rate": unseen_point,
            "unseen_ci_95": f"[{unseen_lo:.3f}, {unseen_hi:.3f}]",
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results_bootstrap_ci.csv", index=False)
    print("\n=== Detection rates with 95% bootstrap CIs ===")
    print(results_df.to_string(index=False))