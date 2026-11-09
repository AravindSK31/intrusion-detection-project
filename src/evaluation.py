"""
Shared evaluation utilities used by every detector (supervised + unsupervised).
Keeping this in one place ensures every method is scored identically --
same validation-based thresholding, same seen/unseen split, same exclusions.
"""

import numpy as np
import pandas as pd


def make_normal_validation_split(train_df: pd.DataFrame, validation_fraction: float, random_state: int):
    normal_df = train_df[train_df["label"] == "normal"]
    attack_df = train_df[train_df["label"] != "normal"]

    normal_val = normal_df.sample(frac=validation_fraction, random_state=random_state)
    normal_train = normal_df.drop(normal_val.index)

    model_train_df = pd.concat([normal_train, attack_df]).sample(frac=1.0, random_state=random_state)
    return model_train_df, normal_val


def get_binary_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] != "normal").astype(int).values


def tag_seen_unseen(test_df: pd.DataFrame, train_attack_labels: set) -> pd.DataFrame:
    def flag(label):
        if label == "normal":
            return "normal"
        return "seen" if label in train_attack_labels else "unseen"
    test_df = test_df.copy()
    test_df["seen_flag"] = test_df["label"].apply(flag)
    return test_df


def threshold_from_normal_scores(normal_scores: np.ndarray, target_fpr: float) -> float:
    percentile = 100 * (1 - target_fpr)
    return np.percentile(normal_scores, percentile)


def evaluate_at_threshold(test_df, test_scores, threshold, undetectable_indices, higher_is_more_anomalous=True):
    """
    higher_is_more_anomalous: True for e.g. RF attack-probability or
    reconstruction error. Set False for methods where LOWER scores mean
    more anomalous (e.g. raw GMM log-likelihood) -- flip the sign before
    calling, or pass False and this will flip internally.
    """
    scores = test_scores if higher_is_more_anomalous else -test_scores
    thresh = threshold if higher_is_more_anomalous else -threshold

    is_normal = test_df["label"].values == "normal"
    flagged = scores >= thresh

    results = {}
    results["actual_fpr_on_test_normal"] = flagged[is_normal].mean()

    seen_mask = test_df["seen_flag"].values == "seen"
    unseen_mask = test_df["seen_flag"].values == "unseen"
    exclude_mask = test_df.index.isin(undetectable_indices)
    unseen_eval_mask = unseen_mask & ~exclude_mask

    results["detection_rate_seen"] = flagged[seen_mask].mean() if seen_mask.sum() > 0 else np.nan
    results["detection_rate_unseen"] = flagged[unseen_eval_mask].mean() if unseen_eval_mask.sum() > 0 else np.nan
    results["n_unseen_excluded_as_undetectable"] = exclude_mask.sum()
    return results