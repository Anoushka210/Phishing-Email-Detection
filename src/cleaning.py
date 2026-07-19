"""
Phase 2: Data Cleaning.

Text corpus:
  - drop exact duplicates
  - strip HTML tags (BeautifulSoup, handles malformed markup)
  - normalize whitespace
  - lowercase
  - remove non-printable/control characters
  - KEEP urls and punctuation intact - these feed Phase 3's structural
    feature extraction (URL counts, domain mismatch, etc). Stripping
    punctuation/stopwords happens later, only on the copy fed to TF-IDF,
    not on this canonical cleaned version.

Metadata-only set:
  - stratified sample to a submission-friendly size
  - full phishing minority class retained, legitimate class downsampled
"""
import re

import pandas as pd
from bs4 import BeautifulSoup

from src.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    METADATA_LEGIT_SAMPLE_SIZE,
)


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def clean_text_corpus() -> pd.DataFrame:
    path = INTERIM_DIR / "merged_raw.csv"
    df = pd.read_csv(path)
    print(f"[text corpus] loaded {len(df)} rows")

    before = len(df)
    df = df.drop_duplicates(subset=["email_text"], keep="first")
    print(f"[text corpus] dropped {before - len(df)} exact duplicates")

    df["email_text_clean"] = (
        df["email_text"]
        .apply(strip_html)
        .apply(remove_control_chars)
        .apply(normalize_whitespace)
        .str.lower()
    )

    # Drop rows that became empty after cleaning (e.g. HTML-only emails)
    before = len(df)
    df = df[df["email_text_clean"].str.len() > 0]
    print(f"[text corpus] dropped {before - len(df)} rows empty after cleaning")

    df["label"] = df["label"].astype(int)

    out_path = PROCESSED_DIR / "text_corpus_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[saved] {len(df)} rows -> {out_path}")
    return df


def clean_metadata_set() -> pd.DataFrame:
    path = INTERIM_DIR / "ethancratchley_metadata_only.csv"
    df = pd.read_csv(path)
    print(f"[metadata set] loaded {len(df)} rows")

    before = len(df)
    df = df.drop_duplicates(keep="first")
    print(f"[metadata set] dropped {before - len(df)} exact duplicates")

    phishing = df[df["label"] == 1]
    legit = df[df["label"] == 0]

    legit_sample = legit.sample(
        n=min(METADATA_LEGIT_SAMPLE_SIZE, len(legit)),
        random_state=RANDOM_SEED,
    )

    sampled = pd.concat([phishing, legit_sample], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"[metadata set] stratified sample: "
          f"{len(phishing)} phishing + {len(legit_sample)} legitimate "
          f"= {len(sampled)} rows ({len(phishing)/len(sampled):.1%} phishing)")

    out_path = PROCESSED_DIR / "metadata_only_clean.csv"
    sampled.to_csv(out_path, index=False)
    print(f"[saved] {len(sampled)} rows -> {out_path}")
    return sampled


if __name__ == "__main__":
    clean_text_corpus()
    print()
    clean_metadata_set()