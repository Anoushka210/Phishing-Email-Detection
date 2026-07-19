"""
Phase 3: Feature Engineering.

Builds structural/content-derived features from email_text_clean
(no HTML/header data survives Phase 2, so all features are text-derived,
not sender/domain-derived - see data_card.md limitations).

Also prepares a stopword-stripped, punctuation-stripped text column for
TF-IDF vectorization - but does NOT fit the vectorizer here. TF-IDF
fitting happens in Phase 4's training script, on the train split only,
to avoid leaking test-set vocabulary into the vectorizer's vocabulary.
"""
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import PROCESSED_DIR

FUSED_TLD_PATTERN = re.compile(
    r"\b\w{10,}(com|net|org|edu|gov|info|biz|nl|co|io)\b", re.IGNORECASE
)
LINK_TOKEN_PATTERN = re.compile(r"\b(http|https|www)\b", re.IGNORECASE)

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
}

# Curated urgency/phishing-trigger lexicon - extend this over time,
# document additions in data_card.md so your methodology stays traceable
URGENCY_KEYWORDS = [
    "verify your account", "account suspended", "click here", "act now",
    "urgent", "immediately", "confirm your", "unauthorized access",
    "limited time", "your account will be", "update your information",
    "security alert", "suspicious activity", "password expires",
    "final notice", "action required",
]


def extract_domain(url: str) -> str:
    match = re.search(r"https?://([^/\s]+)", url)
    if match:
        return match.group(1).lower()
    return url.lower()


def count_urls_features(text: str) -> dict:
    """
    Token-based, since punctuation (://,  .) is already stripped from
    this corpus. Counts http/www mentions as a proxy for link presence,
    and detects unusually long fused tokens ending in a TLD fragment as
    a proxy for squashed emails/domains (e.g. 'aokfbozza4pluscom').
    """
    link_mentions = len(LINK_TOKEN_PATTERN.findall(text))
    fused_tld_hits = len(FUSED_TLD_PATTERN.findall(text))

    return {
        "num_link_mentions": link_mentions,
        "num_fused_domain_tokens": fused_tld_hits,
    }


def count_email_addresses(text: str) -> int:
    """
    Standard @-based regex will find nothing here (@ is stripped).
    Approximate email mentions via fused-domain-token count instead -
    same signal as count_urls_features's fused_tld_hits, so this
    function is now a thin duplicate; kept separate for clarity in
    the feature table, values will mirror num_fused_domain_tokens.
    """
    return len(FUSED_TLD_PATTERN.findall(text))



def count_urgency_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(kw) for kw in URGENCY_KEYWORDS)


def formatting_features(text: str) -> dict:
    if len(text) == 0:
        return {"capital_ratio": 0.0, "exclamation_count": 0, "digit_ratio": 0.0}

    letters = [c for c in text if c.isalpha()]
    capital_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)

    return {
        "capital_ratio": round(capital_ratio, 4),
        "exclamation_count": text.count("!"),
        "digit_ratio": round(digit_ratio, 4),
    }


def word_stats(text: str) -> dict:
    words = text.split()
    unique_words = set(words)
    stopword_count = sum(1 for w in words if w in ENGLISH_STOP_WORDS)

    return {
        "num_words": len(words),
        "num_unique_words": len(unique_words),
        "num_stopwords": stopword_count,
        "avg_word_length": round(
            sum(len(w) for w in words) / len(words), 2
        ) if words else 0.0,
    }


def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for text in df["email_text_clean"]:
        text = text if isinstance(text, str) else ""
        feats = {}
        feats.update(count_urls_features(text))
        feats["num_urgent_keywords"] = count_urgency_keywords(text)
        feats.update(formatting_features(text))
        feats.update(word_stats(text))
        records.append(feats)

    return pd.DataFrame(records)


def prepare_tfidf_text(text: str) -> str:
    """
    Strip stopwords for the TF-IDF-ready column.
    Note: this corpus already has URLs/punctuation stripped into plain
    tokens (e.g. 'http www tcigp com'), so there's no intact URL syntax
    or punctuation left to remove here - link/domain signal is already
    captured separately as num_link_mentions / num_fused_domain_tokens
    in the structural features. This function just does stopword removal.
    """
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def run_feature_engineering() -> pd.DataFrame:
    path = PROCESSED_DIR / "text_corpus_clean.csv"
    df = pd.read_csv(path)
    print(f"[feature engineering] loaded {len(df)} rows")

    structural = build_structural_features(df)
    df = pd.concat([df.reset_index(drop=True), structural], axis=1)

    df["text_for_vectorization"] = df["email_text_clean"].apply(prepare_tfidf_text)

    out_path = PROCESSED_DIR / "text_corpus_features.csv"
    df.to_csv(out_path, index=False)
    print(f"[saved] {len(df)} rows, {len(df.columns)} columns -> {out_path}")

    print("\nFeature summary (mean by label):")
    feature_cols = list(structural.columns)
    print(df.groupby("label")[feature_cols].mean().T)

    return df


if __name__ == "__main__":
    run_feature_engineering()