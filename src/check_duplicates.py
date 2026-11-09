"""
Duplicate and leakage sanity checks.
Run this before any modeling -- if duplicates exist between train and test,
your test accuracy will be inflated by memorization, not generalization.
"""

import sys
from load_nslkdd import load_split, FEATURE_COLUMNS


def check_duplicates(train_df, test_df):
    # Duplicates within each split (on feature columns only, ignoring label)
    train_dupes = train_df[FEATURE_COLUMNS].duplicated().sum()
    test_dupes = test_df[FEATURE_COLUMNS].duplicated().sum()
    print(f"Duplicate rows WITHIN train (feature columns only): {train_dupes}")
    print(f"Duplicate rows WITHIN test (feature columns only): {test_dupes}")

    # Overlap BETWEEN train and test -- this is the dangerous one.
    # If a test row's features exactly match a train row, the model could
    # have memorized it rather than generalized.
    merged = train_df[FEATURE_COLUMNS].merge(
        test_df[FEATURE_COLUMNS], how="inner"
    )
    print(f"Rows in TEST that exactly match a row in TRAIN (feature overlap): {len(merged)}")
    print(f"  -> as a % of test set: {100 * len(merged) / len(test_df):.2f}%")


if __name__ == "__main__":
    train_df = load_split(sys.argv[1])
    test_df = load_split(sys.argv[2])
    check_duplicates(train_df, test_df)

    print("\n--- What TRAIN label do the unseen-attack overlaps correspond to? ---")
    unseen_of_concern = ["snmpgetattack", "saint"]
    merge_cols = FEATURE_COLUMNS

    for lbl in unseen_of_concern:
        test_subset = test_df[test_df["label"] == lbl]
        matched = test_subset.merge(
            train_df[merge_cols + ["label"]],
            on=merge_cols,
            how="inner",
            suffixes=("_test", "_train"),
        )
        print(f"\nTest label = {lbl}: {len(test_subset)} total rows, {len(matched)} match a train row")
        if len(matched) > 0:
            print("  Matching TRAIN rows were labeled as:")
            print(matched["label_train"].value_counts())