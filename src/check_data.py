"""
Quick sanity check on Phase 1 outputs before moving to cleaning.
Run after data_merger.py to confirm row counts, label balance,
and catch obvious issues (nulls, duplicates) early.
"""
import pandas as pd

from src.config import INTERIM_DIR


def check_text_corpus():
    path = INTERIM_DIR / "merged_raw.csv"
    df = pd.read_csv(path)

    print("=== Text Corpus (merged_raw.csv) ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Null email_text rows: {df['email_text'].isna().sum()}")
    print(f"Exact duplicate emails: {df.duplicated(subset=['email_text']).sum()}")
    print()


def check_metadata_set():
    path = INTERIM_DIR / "ethancratchley_metadata_only.csv"
    df = pd.read_csv(path)

    print("=== Metadata-only set (ethancratchley) ===")
    print(f"Rows: {len(df)}")
    dist = df['label'].value_counts()
    print(f"Label distribution:\n{dist}")
    print(f"Phishing rate: {dist[1] / len(df):.2%}")
    print(f"Null rows (any column): {df.isna().any(axis=1).sum()}")
    print()


if __name__ == "__main__":
    check_text_corpus()
    check_metadata_set()