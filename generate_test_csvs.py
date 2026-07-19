# generate_test_csvs.py
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/text_corpus_clean.csv")

Path("test_uploads").mkdir(exist_ok=True)

# Small file - quick smoke test, mix of phishing and legitimate
small = df.sample(n=20, random_state=1)[["email_text", "label"]]
small.to_csv("test_uploads/sample_small_20.csv", index=False)

# Medium file - realistic batch size, checks progress bar/UX
medium = df.sample(n=300, random_state=2)[["email_text", "label"]]
medium.to_csv("test_uploads/sample_medium_300.csv", index=False)

# Larger file - stress test, checks how slow the per-row loop actually is
large = df.sample(n=2000, random_state=3)[["email_text", "label"]]
large.to_csv("test_uploads/sample_large_2000.csv", index=False)

malformed = df.sample(n=10, random_state=4)[["email_text", "label"]].rename(columns={"email_text": "body"})
malformed.to_csv("test_uploads/sample_malformed_column.csv", index=False)

print("Generated: sample_small_20.csv, sample_medium_300.csv, sample_large_2000.csv")