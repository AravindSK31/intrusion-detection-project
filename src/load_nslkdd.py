"""
NSL-KDD data loading utilities.

The raw KDDTrain+.txt / KDDTest+.txt files are headerless CSVs with 41 feature
columns, plus a label column (attack name) and a difficulty-score column.
Column names below follow the standard NSL-KDD / KDD'99 field documentation.
"""

import pandas as pd

# 41 feature columns, in order, as defined in the NSL-KDD documentation.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

ALL_COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Maps every individual attack label in NSL-KDD to its broad category.
# "normal" is its own category. Anything not listed here will raise a
# KeyError when mapped -- this is intentional, so an unmapped label
# (e.g. a new attack subtype you haven't accounted for) surfaces loudly
# instead of silently being dropped or mislabeled.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "apache2": "DoS",
    "worm": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (remote to local)
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L",
    # U2R (user to root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


def load_split(path: str) -> pd.DataFrame:
    """Load a single NSL-KDD file (KDDTrain+.txt or KDDTest+.txt)."""
    df = pd.read_csv(path, names=ALL_COLUMNS, header=None)
    # strip stray punctuation some mirrors leave on the label column (e.g. "normal.")
    df["label"] = df["label"].str.replace(".", "", regex=False).str.strip()
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP)
    unmapped = df[df["attack_category"].isna()]["label"].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"Found label(s) not in ATTACK_CATEGORY_MAP: {list(unmapped)}. "
            "Add them before proceeding -- silently dropping them would "
            "quietly bias your seen/unseen attack analysis."
        )
    return df


def seen_vs_unseen_labels(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Returns (seen_labels, unseen_labels): the individual attack labels
    (not categories) that do / do not appear in the training set.
    'normal' is excluded from both -- it's not an attack.
    """
    train_labels = set(train_df["label"].unique()) - {"normal"}
    test_labels = set(test_df["label"].unique()) - {"normal"}
    seen = train_labels & test_labels
    unseen = test_labels - train_labels
    return seen, unseen

def get_undetectable_unseen_rows(train_df, test_df, unseen_label: str):
    """
    Returns the indices (within test_df) of rows labeled `unseen_label`
    whose features exactly match a 'normal'-labeled row in train_df.
    These rows are indistinguishable from benign traffic under this feature
    set, so no detector can be expected to catch them -- excluding them
    from unseen-attack evaluation prevents an artificial ceiling on every
    method's reported detection rate.
    """
    normal_train = train_df[train_df["label"] == "normal"][FEATURE_COLUMNS].drop_duplicates()
    target_rows = test_df[test_df["label"] == unseen_label]
    matched = target_rows.merge(normal_train, on=FEATURE_COLUMNS, how="inner")
    # recover original test_df indices of the matched rows
    merged_with_index = target_rows.reset_index().merge(
        normal_train, on=FEATURE_COLUMNS, how="inner"
    )
    return set(merged_with_index["index"])

if __name__ == "__main__":
    import sys
    train_path, test_path = sys.argv[1], sys.argv[2]
    train_df = load_split(train_path)
    test_df = load_split(test_path)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print("\nTrain attack_category counts:\n", train_df["attack_category"].value_counts())
    print("\nTest attack_category counts:\n", test_df["attack_category"].value_counts())

    seen, unseen = seen_vs_unseen_labels(train_df, test_df)
    print(f"\nAttack labels seen in training ({len(seen)}):", sorted(seen))
    print(f"\nAttack labels UNSEEN in training, only in test ({len(unseen)}):", sorted(unseen))