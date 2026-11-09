"""
CICIDS2017 loader for our two Wednesday (DoS) + Friday (PortScan) files.
Builds a two-tier seen/unseen structure:
  - seen: DoS Hulk, DoS GoldenEye (used in training)
  - unseen_same_category: DoS slowloris, DoS Slowhttptest (withheld DoS subtypes)
  - unseen_new_category: PortScan (an entire category never trained on)
  - excluded: Heartbleed (only 11 rows -- too small to evaluate reliably)
"""

import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = ["Destination Port"]  # known label-leakage risk, dropped before modeling

SEEN_LABELS = {"DoS Hulk", "DoS GoldenEye"}
UNSEEN_SAME_CATEGORY_LABELS = {"DoS slowloris", "DoS Slowhttptest"}
EXCLUDED_LABELS = {"Heartbleed"}  # sample size too small (11 rows) to evaluate either way


def load_cicids_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # fix the leading-space quirk

    # Flow Bytes/s and Flow Packets/s can contain inf/-inf when flow duration
    # rounds to 0 -- a documented CICIDS2017 quirk. Replace with NaN, then
    # we'll impute (median) during preprocessing rather than silently drop rows.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def tag_cicids_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'split_group' column: 'normal', 'seen', 'unseen_same_category',
    'unseen_new_category', or 'excluded'. PortScan rows are tagged
    'unseen_new_category' regardless of label set membership above, since
    this function is applied per-file and Friday's file is entirely PortScan.
    """
    df = df.copy()

    def tag(label):
        if label == "BENIGN":
            return "normal"
        if label in EXCLUDED_LABELS:
            return "excluded"
        if label in SEEN_LABELS:
            return "seen"
        if label in UNSEEN_SAME_CATEGORY_LABELS:
            return "unseen_same_category"
        if label == "PortScan":
            return "unseen_new_category"
        raise ValueError(f"Unmapped CICIDS2017 label encountered: {label!r}")

    df["split_group"] = df["Label"].apply(tag)
    return df


if __name__ == "__main__":
    import sys
    wed_df = tag_cicids_rows(load_cicids_file(sys.argv[1]))
    fri_df = tag_cicids_rows(load_cicids_file(sys.argv[2]))

    print("Wednesday split_group counts:")
    print(wed_df["split_group"].value_counts())
    print("\nFriday split_group counts:")
    print(fri_df["split_group"].value_counts())

    for name, df in [("Wednesday", wed_df), ("Friday", fri_df)]:
        feature_cols = df.drop(columns=["Label", "split_group"]).columns
        nan_rows = df[feature_cols].isna().any(axis=1)
        print(f"\n--- {name}: NaN-affected rows ---")
        print(f"Total rows with at least one NaN: {nan_rows.sum()} ({100*nan_rows.mean():.3f}%)")
        print("Breakdown by split_group:")
        print(df.loc[nan_rows, "split_group"].value_counts())
        print("Which columns contain NaNs:")
        print(df[feature_cols].isna().sum()[df[feature_cols].isna().sum() > 0])