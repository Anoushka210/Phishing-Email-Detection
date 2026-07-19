"""
Shared inference pipeline for the dashboard.
Applies the exact same cleaning + feature engineering steps used during
training (Phases 2-3) to a single new email, then runs it through all
four trained text models.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from src.config import MODELS_DIR, STRUCTURAL_FEATURE_COLS
from src.cleaning import strip_html, remove_control_chars, normalize_whitespace
from src.feature_engineering import (
    count_urls_features, count_urgency_keywords,
    formatting_features, word_stats, prepare_tfidf_text,
)

TEXT_MODEL_NAMES = [
    "logistic_regression", "random_forest", "naive_bayes", "hist_gradient_boosting",
]


def clean_single_email(raw_text: str) -> str:
    text = strip_html(raw_text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    return text.lower()


def build_single_email_features(clean_text: str) -> dict:
    feats = {}
    feats.update(count_urls_features(clean_text))
    feats["num_urgent_keywords"] = count_urgency_keywords(clean_text)
    feats.update(formatting_features(clean_text))
    feats.update(word_stats(clean_text))
    return feats


def load_text_models():
    """Loads vectorizer + all four text models once. Cache this in the app layer."""
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    models = {
        name: joblib.load(MODELS_DIR / f"{name}.joblib")
        for name in TEXT_MODEL_NAMES
    }
    return vectorizer, models


def predict_email(raw_text: str, vectorizer, models: dict) -> dict:
    """
    Runs one email through cleaning, feature engineering, TF-IDF, and all
    four models. Returns predictions + probabilities + the feature matrix
    (needed later for SHAP explanations on the same input).
    """
    clean_text = clean_single_email(raw_text)
    structural_feats = build_single_email_features(clean_text)
    tfidf_text = prepare_tfidf_text(clean_text)

    tfidf_vec = vectorizer.transform([tfidf_text])
    structural_vec = csr_matrix(
        np.array([[structural_feats[col] for col in STRUCTURAL_FEATURE_COLS]])
    )
    X = hstack([tfidf_vec, structural_vec]).tocsr()

    results = {}
    for name, model in models.items():
        X_input = X.toarray().astype("float32") if name == "hist_gradient_boosting" else X
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]
        results[name] = {"prediction": int(pred), "phishing_probability": float(proba)}

    return {
        "clean_text": clean_text,
        "structural_features": structural_feats,
        "feature_matrix": X,
        "model_results": results,
    }