"""Central config: paths and dataset registry for Phase 1."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EXTERNAL_DIR = DATA_DIR / "external"

for d in [RAW_DIR, INTERIM_DIR, EXTERNAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Registry of manually-downloaded raw files.
# Update the 'filename' values once you check what's actually inside
# each folder after extracting your Kaggle downloads.
RAW_SOURCES = {
    "naserabdullahalam": RAW_DIR / "naserabdullahalam" / "phishing_email.csv",
    "ethancratchley": RAW_DIR / "ethancratchley" / "email_phishing_data.csv",
}

UNIFIED_COLUMNS = ["email_text", "subject", "sender", "label", "source_dataset"]