"""
Compare key feature distributions across groups, to sanity-check whether
PortScan's near-total miss by RF/IF/OCSVM (but not GMM) reflects a real
'low-volume attack vs. high-volume attack' distinction, or something else.
"""

import sys
import pandas as pd

from load_cicids import load_cicids_file, tag_cicids_rows

KEY_FEATURES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s"]


if __name__ == "__main__":
    wed_df = tag_cicids_rows(load_cicids_file(sys.argv[1]))
    fri_df = tag_cicids_rows(load_cicids_file(sys.argv[2]))

    groups = {
        "Wednesday normal (train)": wed_df[wed_df["split_group"] == "normal"],
        "Wednesday seen (Hulk/GoldenEye)": wed_df[wed_df["split_group"] == "seen"],
        "Wednesday unseen-same (slowloris/Slowhttptest)": wed_df[wed_df["split_group"] == "unseen_same_category"],
        "Friday normal": fri_df[fri_df["split_group"] == "normal"],
        "Friday PortScan": fri_df[fri_df["split_group"] == "unseen_new_category"],
    }

    for name, g in groups.items():
        print(f"\n--- {name} (n={len(g)}) ---")
        print(g[KEY_FEATURES].median())