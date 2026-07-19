"""Central config: paths and dataset registry."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"          
MODELS_DIR = ROOT_DIR / "models"          
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"      
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

for d in [RAW_DIR, INTERIM_DIR, EXTERNAL_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_SOURCES = {
    "naserabdullahalam": RAW_DIR / "naserabdullahalam" / "phishing_email.csv",
    "ethancratchley": RAW_DIR / "ethancratchley" / "email_phishing_data.csv",
}

UNIFIED_COLUMNS = ["email_text", "subject", "sender", "label", "source_dataset"]

# Phase 2 params
RANDOM_SEED = 42
METADATA_LEGIT_SAMPLE_SIZE = 25000   # legitimate rows to keep in clean metadata set

# Phase 4
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
STRUCTURAL_FEATURE_COLS = [
    "num_link_mentions", "num_fused_domain_tokens", "num_urgent_keywords",
    "capital_ratio", "exclamation_count", "digit_ratio",
    "num_words", "num_unique_words", "num_stopwords", "avg_word_length",
]

# Phase 5
METADATA_FEATURE_COLS = [
    "num_words", "num_unique_words", "num_stopwords", "num_links",
    "num_unique_domains", "num_email_addresses", "num_spelling_errors",
    "num_urgent_keywords",
]