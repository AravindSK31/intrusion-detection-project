"""
Supervised baseline (Random Forest) as a continuous anomaly scorer.
Threshold fixed at TARGET_FPR, chosen from normal-validation data only.
"""

import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from load_nslkdd import load_split, get_undetectable_unseen_rows
from preprocess import fit_feature_pipeline, transform_features
from evaluation import (
    make_normal_validation_split, get_binary_labels, tag_seen_unseen,
    threshold_from_normal_scores, evaluate_at_threshold,
)

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TARGET_FPR = 0.01  # locked in


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

    pipeline = fit_feature_pipeline(model_train_df)
    X_train = transform_features(model_train_df, pipeline)
    X_val_normal = transform_features(normal_val_df, pipeline)
    X_test = transform_features(test_df, pipeline)

    y_train = get_binary_labels(model_train_df)
    y_test = get_binary_labels(test_df)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    val_normal_scores = clf.predict_proba(X_val_normal)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]

    print(f"Overall test ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")

    threshold = threshold_from_normal_scores(val_normal_scores, TARGET_FPR)
    results = evaluate_at_threshold(test_df, test_scores, threshold, undetectable)

    print(f"\n[Random Forest] Target FPR: {TARGET_FPR:.0%}  (threshold = {threshold:.4f})")
    for k, v in results.items():
        print(f"  {k}: {v}")