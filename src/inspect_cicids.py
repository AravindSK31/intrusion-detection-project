"""
Quick inspection of CICIDS2017 columns before building a full loader --
checking for the leading-space column name quirk and the known leakage
columns (Destination Port, Flow ID, IPs, Timestamp).
"""

import sys
import pandas as pd

if __name__ == "__main__":
    path = sys.argv[1]
    df = pd.read_csv(path, nrows=5)  # just a peek, don't load everything yet

    print("Column names (note any leading/trailing whitespace):")
    for col in df.columns:
        print(f"  {repr(col)}")

    print(f"\nTotal columns: {len(df.columns)}")
    print(f"\nFirst few rows:\n{df.head()}")