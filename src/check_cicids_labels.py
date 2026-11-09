"""
Print exact label values and counts for a CICIDS2017 file, after stripping
whitespace from column names (the known leading-space quirk) -- so we know
the precise attack subtype strings before designing the seen/unseen split.
"""

import sys
import pandas as pd

if __name__ == "__main__":
    path = sys.argv[1]
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print(f"Total rows: {len(df)}")
    print("\nLabel value counts:")
    print(df["Label"].value_counts())