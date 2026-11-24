"""
Three-way CICIDS2017 generalization test: trains on Wednesday's BENIGN +
seen DoS (Hulk/GoldenEye), evaluates on three held-out conditions:
  - unseen_same_category: slowloris/Slowhttptest (withheld DoS subtypes)
  - unseen_new_category_portscan: Friday PortScan (low-volume novel category)
  - unseen_new_category_ddos: Friday DDoS (volumetric novel category)
Runs RF, Isolation Forest, GMM, One-Class SVM, and the RF+GMM hybrid (OR rule)
across all three conditions in one consistent run.
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


def load_and_tag_ddos(path):
    df = load_cicids_file(path)
    df = df.copy()
    df["split_group"] = df["Label"].apply(
        lambda l: "normal" if l == "BENIGN" else "unseen_new_category_ddos"
    )
    return df


def load_and_tag_portscan(path):
    df = tag_cicids_rows(load_cicids_file(path))
    # Relabel PortScan's existing 'unseen_new_category' tag to be specific,
    # so it doesn't collide with DDoS's tag once we concatenate everything.
    df = df.copy()
    df.loc[df["split_group"] == "unseen_new_category", "split_group"] = "unseen_new_category_portscan"
    return df


def get_binary_labels(df):
    return (df["split_group"] != "normal").astype(int).values


def threshold_from_normal_scores(normal_scores, target_fpr):
    return np.percentile(normal_scores, 100 * (1 - target_fpr))


if __name__ == "__main__":
    wed_path, fri_portscan_path, fri_ddos_path = sys.argv[1], sys.argv[2], sys.argv[3]

    wed_df = tag_cicids_rows(load_cicids_file(wed_path))
    fri_portscan_df = load_and_tag_portscan(fri_portscan_path)
    fri_ddos_df = load_and_tag_ddos(fri_ddos_path)

    normal_df = wed_df[wed_df["split_group"] == "normal"]
    seen_df = wed_df[wed_df["split_group"] == "seen"]
    unseen_same_df = wed_df[wed_df["split_group"] == "unseen_same_category"]

    normal_val = normal_df.sample(frac=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    normal_train = normal_df.drop(normal_val.index)
    model_train_df = pd.concat([normal_train, seen_df]).sample(frac=1.0, random_state=RANDOM_STATE)

    # Test set: unseen-same (Wednesday) + ALL of Friday PortScan + ALL of Friday DDoS
    test_df = pd.concat([unseen_same_df, fri_portscan_df, fri_ddos_df], ignore_index=False)

    print(f"Model training rows: {len(model_train_df)}, normal-validation rows: {len(normal_val)}")
    print(f"Test set breakdown:\n{test_df['split_group'].value_counts()}\n")

    normal_only_train = model_train_df[model_train_df["split_group"] == "normal"]
    y_test = get_binary_labels(test_df)

    is_normal = test_df["split_group"].values == "normal"
    same_cat_mask = test_df["split_group"].values == "unseen_same_category"
    portscan_mask = test_df["split_group"].values == "unseen_new_category_portscan"
    ddos_mask = test_df["split_group"].values == "unseen_new_category_ddos"

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

    # --- Unsupervised methods (normal-only pipeline) ---
    pipeline_u = fit_cicids_pipeline(normal_only_train)
    X_train_u = transform_cicids(normal_only_train, pipeline_u)
    X_val_u = transform_cicids(normal_val, pipeline_u)
    X_test_u = transform_cicids(test_df, pipeline_u)

    iso = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_u)
    iso_val_scores, iso_test_scores = -iso.score_samples(X_val_u), -iso.score_samples(X_test_u)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE, covariance_type="diag")
    gmm.fit(X_train_u)
    gmm_val_scores, gmm_test_scores = -gmm.score_samples(X_val_u), -gmm.score_samples(X_test_u)

    sample = X_train_u.sample(n=min(SVM_TRAIN_SAMPLE_SIZE, len(X_train_u)), random_state=RANDOM_STATE)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(sample)
    ocsvm_val_scores, ocsvm_test_scores = -ocsvm.decision_function(X_val_u), -ocsvm.decision_function(X_test_u)

    methods = {
        "Random Forest": (rf_val_scores, rf_test_scores),
        "Isolation Forest": (iso_val_scores, iso_test_scores),
        "GMM": (gmm_val_scores, gmm_test_scores),
        "One-Class SVM": (ocsvm_val_scores, ocsvm_test_scores),
    }

    rows = []
    for name, (val_scores, test_scores) in methods.items():
        threshold = threshold_from_normal_scores(val_scores, TARGET_FPR)
        flagged = test_scores >= threshold
        rows.append({
            "method": name,
            "fpr": flagged[is_normal].mean(),
            "unseen_same_category": flagged[same_cat_mask].mean(),
            "unseen_new_portscan": flagged[portscan_mask].mean(),
            "unseen_new_ddos": flagged[ddos_mask].mean(),
        })

    # --- Hybrid RF + GMM, OR rule ---
    rf_threshold = threshold_from_normal_scores(rf_val_scores, TARGET_FPR)
    gmm_threshold = threshold_from_normal_scores(gmm_val_scores, TARGET_FPR)
    hybrid_flagged = (rf_test_scores >= rf_threshold) | (gmm_test_scores >= gmm_threshold)
    rows.append({
        "method": "Hybrid RF+GMM (OR)",
        "fpr": hybrid_flagged[is_normal].mean(),
        "unseen_same_category": hybrid_flagged[same_cat_mask].mean(),
        "unseen_new_portscan": hybrid_flagged[portscan_mask].mean(),
        "unseen_new_ddos": hybrid_flagged[ddos_mask].mean(),
    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results_three_way_cicids.csv", index=False)
    print("=== Three-Way CICIDS2017 Generalization Results ===")
    print(results_df.to_string(index=False))