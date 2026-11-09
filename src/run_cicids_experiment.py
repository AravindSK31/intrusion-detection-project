"""
CICIDS2017 experiment: trains all four detectors on Wednesday's BENIGN +
seen DoS subtypes, evaluates on two held-out tiers (unseen-same-category
DoS subtypes, and unseen-new-category PortScan from Friday).
Mirrors the NSL-KDD methodology exactly: thresholds chosen only from
normal-validation data, same target FPR, same model configurations.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

from load_cicids import load_cicids_file, tag_cicids_rows
from preprocess_cicids import fit_cicids_pipeline, transform_cicids

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TARGET_FPR = 0.01
N_COMPONENTS = 5
SVM_TRAIN_SAMPLE_SIZE = 5000


def build_splits(wed_df, fri_df):
    normal_df = wed_df[wed_df["split_group"] == "normal"]
    seen_df = wed_df[wed_df["split_group"] == "seen"]

    normal_val = normal_df.sample(frac=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    normal_train = normal_df.drop(normal_val.index)

    # Model-training data: remaining normal + all seen-attack rows
    model_train_df = pd.concat([normal_train, seen_df]).sample(frac=1.0, random_state=RANDOM_STATE)

    # Test set: unseen-same-category (from Wednesday) + ALL of Friday
    # (Friday's own 'normal' rows serve as an independent check on whether
    # test-normal drift, as found in NSL-KDD, replicates here too)
    unseen_same_df = wed_df[wed_df["split_group"] == "unseen_same_category"]
    test_df = pd.concat([unseen_same_df, fri_df], ignore_index=False)

    return model_train_df, normal_train, normal_val, test_df


def get_binary_labels(df):
    return (df["split_group"] != "normal").astype(int).values


def threshold_from_normal_scores(normal_scores, target_fpr):
    return np.percentile(normal_scores, 100 * (1 - target_fpr))


def evaluate(test_df, test_scores, threshold):
    is_normal = test_df["split_group"].values == "normal"
    flagged = test_scores >= threshold

    results = {"actual_fpr_on_test_normal": flagged[is_normal].mean()}
    for group in ["unseen_same_category", "unseen_new_category"]:
        mask = test_df["split_group"].values == group
        results[f"detection_rate_{group}"] = flagged[mask].mean() if mask.sum() > 0 else np.nan
    return results


def run_all_methods(model_train_df, normal_train, normal_val, test_df):
    normal_only_train = model_train_df[model_train_df["split_group"] == "normal"]

    results_rows = []

    # --- Random Forest (supervised) ---
    pipeline = fit_cicids_pipeline(model_train_df)
    X_train = transform_cicids(model_train_df, pipeline)
    X_val = transform_cicids(normal_val, pipeline)
    X_test = transform_cicids(test_df, pipeline)
    y_train = get_binary_labels(model_train_df)
    y_test = get_binary_labels(test_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    val_scores, test_scores = clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]
    threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
    row = {"method": "Random Forest", "roc_auc": roc_auc_score(y_test, test_scores)}
    row.update(evaluate(test_df, test_scores, threshold))
    results_rows.append(row)

    # --- Isolation Forest, GMM, One-Class SVM (unsupervised, normal-only) ---
    pipeline_u = fit_cicids_pipeline(normal_only_train)
    X_train_u = transform_cicids(normal_only_train, pipeline_u)
    X_val_u = transform_cicids(normal_val, pipeline_u)
    X_test_u = transform_cicids(test_df, pipeline_u)

    iso = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_u)
    val_scores, test_scores = -iso.score_samples(X_val_u), -iso.score_samples(X_test_u)
    threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
    row = {"method": "Isolation Forest", "roc_auc": roc_auc_score(y_test, test_scores)}
    row.update(evaluate(test_df, test_scores, threshold))
    results_rows.append(row)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_u)
    val_scores, test_scores = -gmm.score_samples(X_val_u), -gmm.score_samples(X_test_u)
    threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
    row = {"method": "GMM", "roc_auc": roc_auc_score(y_test, test_scores)}
    row.update(evaluate(test_df, test_scores, threshold))
    results_rows.append(row)

    sample = X_train_u.sample(n=min(SVM_TRAIN_SAMPLE_SIZE, len(X_train_u)), random_state=RANDOM_STATE)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(sample)
    val_scores, test_scores = -ocsvm.decision_function(X_val_u), -ocsvm.decision_function(X_test_u)
    threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
    row = {"method": "One-Class SVM", "roc_auc": roc_auc_score(y_test, test_scores)}
    row.update(evaluate(test_df, test_scores, threshold))
    results_rows.append(row)

    return pd.DataFrame(results_rows)


if __name__ == "__main__":
    wed_path, fri_path = sys.argv[1], sys.argv[2]
    wed_df = tag_cicids_rows(load_cicids_file(wed_path))
    fri_df = tag_cicids_rows(load_cicids_file(fri_path))

    model_train_df, normal_train, normal_val, test_df = build_splits(wed_df, fri_df)

    print(f"Model training rows: {len(model_train_df)}, normal-validation rows: {len(normal_val)}")
    print(f"Test rows: {len(test_df)} "
          f"(unseen_same_category: {(test_df['split_group']=='unseen_same_category').sum()}, "
          f"unseen_new_category: {(test_df['split_group']=='unseen_new_category').sum()}, "
          f"normal: {(test_df['split_group']=='normal').sum()})")

    results_df = run_all_methods(model_train_df, normal_train, normal_val, test_df)
    results_df.to_csv("results_summary_cicids.csv", index=False)
    print("\n=== CICIDS2017 Results ===")
    print(results_df.to_string(index=False))