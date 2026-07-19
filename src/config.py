"""Central config: paths and dataset registry."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"          # NEW

for d in [RAW_DIR, INTERIM_DIR, EXTERNAL_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_SOURCES = {
    "naserabdullahalam": RAW_DIR / "naserabdullahalam" / "phishing_email.csv",
    "ethancratchley": RAW_DIR / "ethancratchley" / "email_phishing_data.csv",
}

UNIFIED_COLUMNS = ["email_text", "subject", "sender", "label", "source_dataset"]

# Phase 2 params
RANDOM_SEED = 42
METADATA_LEGIT_SAMPLE_SIZE = 25000   # legitimate rows to keep in clean metadata set