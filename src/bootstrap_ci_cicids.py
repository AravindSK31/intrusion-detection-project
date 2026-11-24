"""
Bootstrap confidence intervals for CICIDS2017 detection rates.
Self-contained: trains each method once, then bootstraps the evaluation
(resampling which test rows are scored), rather than importing functions
that don't exist in run_cicids_experiment.py's current structure.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

from load_cicids import load_cicids_file, tag_cicids_rows
from preprocess_cicids import fit_cicids_pipeline, transform_cicids

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TARGET_FPR = 0.01
N_COMPONENTS = 5
SVM_TRAIN_SAMPLE_SIZE = 5000
N_BOOTSTRAP = 1000


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


def bootstrap_detection_rate(test_scores, threshold, group_mask, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
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


def train_all_methods(model_train_df, normal_train, normal_val, test_df):
    normal_only_train = model_train_df[model_train_df["split_group"] == "normal"]
    scores = {}

    # Random Forest (supervised)
    pipeline = fit_cicids_pipeline(model_train_df)
    X_train = transform_cicids(model_train_df, pipeline)
    X_val = transform_cicids(normal_val, pipeline)
    X_test = transform_cicids(test_df, pipeline)
    y_train = get_binary_labels(model_train_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    scores["Random Forest"] = (clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1])

    # Unsupervised methods share a normal-only pipeline
    pipeline_u = fit_cicids_pipeline(normal_only_train)
    X_train_u = transform_cicids(normal_only_train, pipeline_u)
    X_val_u = transform_cicids(normal_val, pipeline_u)
    X_test_u = transform_cicids(test_df, pipeline_u)

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


if __name__ == "__main__":
    wed_path, fri_path = sys.argv[1], sys.argv[2]
    wed_df = tag_cicids_rows(load_cicids_file(wed_path))
    fri_df = tag_cicids_rows(load_cicids_file(fri_path))

    model_train_df, normal_train, normal_val, test_df = build_splits(wed_df, fri_df)

    print("Training all methods once (this will take a few minutes on this larger dataset)...")
    methods = train_all_methods(model_train_df, normal_train, normal_val, test_df)

    same_cat_mask = test_df["split_group"].values == "unseen_same_category"
    new_cat_mask = test_df["split_group"].values == "unseen_new_category"

    rows = []
    for name, (val_scores, test_scores) in methods.items():
        threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)

        same_point, same_lo, same_hi = bootstrap_detection_rate(test_scores, threshold, same_cat_mask)
        new_point, new_lo, new_hi = bootstrap_detection_rate(test_scores, threshold, new_cat_mask)

        rows.append({
            "method": name,
            "unseen_same_category_rate": same_point,
            "unseen_same_category_ci_95": f"[{same_lo:.3f}, {same_hi:.3f}]",
            "unseen_new_category_rate": new_point,
            "unseen_new_category_ci_95": f"[{new_lo:.3f}, {new_hi:.3f}]",
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results_bootstrap_ci_cicids.csv", index=False)
    print("\n=== CICIDS2017 Detection Rates with 95% Bootstrap CIs ===")
    print(results_df.to_string(index=False))